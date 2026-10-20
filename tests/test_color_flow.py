"""Color-flow state, ODE, renderer, CLI, and optional CUDA regression tests.

Run: python -m unittest discover -s tests -p test_color_flow.py -v
CUDA tests skip automatically without a GPU. Set GROWFLOW_STATIC_SMOKE_CKPT
to also test reuse of an existing static checkpoint at its stored SH degree.
"""

import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from gsplat.strategy import DefaultStrategy

from commands._common import render_config, training_config
from configs.blender_config import Config as BlenderConfig
from configs.captured_config import Config as CapturedConfig
from helpers.gaussian_state import (
    pack_gaussian_state,
    unpack_gaussian_state,
    validate_boundary_trajectory,
    validate_gaussian_state,
)
from models.gaussian_model import Gaussians
from models.neural_ode import DynamicalModel


GEOMETRY = ["means", "quats", "scales"]
COLOR = GEOMETRY + ["shs"]


def make_splats(device="cpu", sh_degree=0):
    return {
        "means": torch.tensor([[-.15, 0, 2], [.15, 0, 2.1], [0, .15, 2.2], [0, -.15, 2.3]], device=device),
        "quats": torch.tensor([[1., 0, 0, 0]], device=device).repeat(4, 1),
        "scales": torch.full((4, 3), .08, device=device).log(),
        "opacities": torch.full((4,), .8, device=device).logit(),
        "sh0": (torch.tensor([[[.3, .6, .1]]], device=device).repeat(4, 1, 1) - .5) / .28209479177387814,
        "shN": torch.linspace(-.03, .03, max(1, 4 * ((sh_degree + 1) ** 2 - 1) * 3),
                              device=device)[:4 * ((sh_degree + 1) ** 2 - 1) * 3].reshape(4, -1, 3),
    }


def make_renderer(splats, color_flow):
    # Skip mesh loading/optimizer creation; exercise the production renderer methods.
    renderer = Gaussians.__new__(Gaussians)
    nn.Module.__init__(renderer)
    renderer.splats = nn.ParameterDict({name: nn.Parameter(value, requires_grad=False)
                                       for name, value in splats.items()})
    renderer.deformed_params_list = COLOR if color_flow else GEOMETRY
    renderer.deformed_params_dict = {"means": 3, "quats": 4, "scales": 3}
    if color_flow:
        renderer.deformed_params_dict["shs"] = 3 * (1 + splats["shN"].shape[1])
    renderer.scale_activation = "exp"
    renderer.packed = False
    renderer.sparse_grad = False
    renderer.camera_model = "pinhole"
    renderer.strategy = DefaultStrategy()
    return renderer


def make_raster_params(splats, color_flow, cameras=2):
    device = splats["means"].device
    params = {
        "opacities": splats["opacities"],
        "viewmats": torch.eye(4, device=device)[None].repeat(cameras, 1, 1),
        "Ks": torch.tensor([[40., 0, 16], [0, 40., 12], [0, 0, 1]], device=device)[None].repeat(cameras, 1, 1),
        "backgrounds": torch.zeros(cameras, 3, device=device),
        "width": 32, "height": 24, "sh_degree": math.isqrt(1 + splats["shN"].shape[1]) - 1,
        "near_plane": .01, "far_plane": 100., "render_mode": "RGB",
        "rasterize_mode": "classic", "packed": False,
    }
    if not color_flow:
        params["colors"] = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return params


def make_model(color_dim=3, encoding="hexplane", **overrides):
    args = dict(
        data_dim=10 + color_dim, color_dim=color_dim, augment_dim=0,
        encoding=encoding, hidden_dim=64, hidden_depth=2, scene_scale=1.,
        max_steps=100, resnet_init=True, adjoint=False,
        rtol=1e-6, atol=1e-7, multires=[1], spatial_temp_resolution=[4, 4, 4, 4],
    )
    args.update(overrides)
    return DynamicalModel(**args)


def color_output_layer(model):
    head = model.odeblock.odefunc.mlp_colors
    return head[-1] if isinstance(head, nn.Sequential) else head


class ColorStateTests(unittest.TestCase):
    def test_geometry_state_is_exactly_legacy_concatenation(self):
        splats = make_splats()
        actual = pack_gaussian_state(splats, GEOMETRY)
        self.assertTrue(torch.equal(actual, torch.cat([splats[k] for k in GEOMETRY], -1)))
        self.assertEqual(actual.shape, (4, 10))

    def test_color_state_preserves_ids_and_unbounded_sh_coefficients(self):
        splats = make_splats()
        splats["sh0"] = torch.arange(12.).reshape(4, 1, 3) - 4
        state = pack_gaussian_state(splats, COLOR)
        self.assertEqual(state.shape, (4, 13))
        fields = unpack_gaussian_state(state, 3)
        for name in GEOMETRY:
            self.assertTrue(torch.equal(fields[name], splats[name]))
        self.assertTrue(torch.equal(fields["colors"], splats["sh0"]))
        self.assertNotIn("opacities", fields)

    def test_temporal_and_foreground_only_shapes_and_gradients(self):
        state = pack_gaussian_state(make_splats(), COLOR)[None].repeat(3, 1, 1).requires_grad_()
        selected = state[:, [0, 2]]
        colors = unpack_gaussian_state(selected, 3)["colors"]
        self.assertEqual(colors.shape, (3, 2, 1, 3))
        colors.sum().backward()
        self.assertTrue(torch.equal(state.grad[:, [0, 2], 10:], torch.ones(3, 2, 3)))
        self.assertEqual(state.grad[:, [1, 3]].abs().sum(), 0)
        self.assertEqual(state.grad[..., :10].abs().sum(), 0)

    def test_cache_round_trip_and_mode_mismatch(self):
        initial = pack_gaussian_state(make_splats(), COLOR)
        trajectory = initial[None].repeat(3, 1, 1).numpy()
        trajectory[1:, :, 10:] += .15
        stream = io.BytesIO()
        np.save(stream, trajectory)
        stream.seek(0)
        restored = np.load(stream, allow_pickle=False)
        validate_boundary_trajectory(restored, initial, 3)
        np.testing.assert_array_equal(restored, trajectory)
        for bad in (restored[..., :10], restored[:, :2], restored[:2]):
            with self.assertRaisesRegex(ValueError, "Boundary trajectory shape"):
                validate_boundary_trajectory(bad, initial, 3)
        for width, color_dim in ((10, 3), (13, 0), (14, 3)):
            with self.assertRaisesRegex(ValueError, "matching --learn-shs"):
                validate_gaussian_state(torch.zeros(4, width), color_dim)

    def test_all_sh_degrees_round_trip_with_temporal_selection_and_gradients(self):
        for degree in range(5):
            with self.subTest(degree=degree):
                splats = make_splats(sh_degree=degree)
                color_dim = 3 * (degree + 1) ** 2
                state = pack_gaussian_state(splats, COLOR)
                expected = torch.cat((splats["sh0"], splats["shN"]), 1)
                self.assertEqual(state.shape, (4, 10 + color_dim))
                torch.testing.assert_close(unpack_gaussian_state(state, color_dim)["colors"],
                                           expected, rtol=0, atol=0)
                temporal = state[None].repeat(3, 1, 1).requires_grad_()
                colors = unpack_gaussian_state(temporal[:, [2, 0]], color_dim)["colors"]
                self.assertEqual(colors.shape, (3, 2, (degree + 1) ** 2, 3))
                torch.testing.assert_close(colors[0], expected[[2, 0]], rtol=0, atol=0)
                colors.sum().backward()
                self.assertTrue(torch.equal(temporal.grad[:, [0, 2], 10:],
                                            torch.ones(3, 2, color_dim)))
                self.assertEqual(temporal.grad[:, [1, 3]].abs().sum(), 0)
                self.assertEqual(temporal.grad[..., :10].abs().sum(), 0)

    def test_incomplete_sh_bases_and_degree_mismatches_are_rejected(self):
        for dim in (-3, 1, 6, 9, 24, 108):
            with self.subTest(color_dim=dim), self.assertRaisesRegex(ValueError, "SH degrees"):
                validate_gaussian_state(torch.zeros(4, max(0, 10 + dim)), dim)
        splats = make_splats()
        splats["shN"] = torch.zeros(4, 2, 3)
        with self.assertRaisesRegex(ValueError, "SH degrees"):
            pack_gaussian_state(splats, COLOR)
        for width, color_dim in ((13, 48), (58, 3)):
            with self.assertRaisesRegex(ValueError, "matching --learn-shs"):
                validate_gaussian_state(torch.zeros(4, width), color_dim)


class ColorODETests(unittest.TestCase):
    def test_higher_degree_heads_and_checkpoint_compatibility(self):
        for degree in range(1, 5):
            for encoding in ("hexplane", "freq", ""):
                with self.subTest(degree=degree, encoding=encoding):
                    color_dim = 3 * (degree + 1) ** 2
                    model = make_model(color_dim=color_dim, encoding=encoding)
                    state = pack_gaussian_state(make_splats(sh_degree=degree), COLOR)
                    derivative = model.odeblock.odefunc(torch.tensor(.2), state)
                    self.assertEqual(derivative.shape, state.shape)
                    self.assertEqual(derivative[:, 10:].abs().sum(), 0)
                    derivative[:, 10:].sum().backward()
                    self.assertTrue((color_output_layer(model).bias.grad != 0).all())
                    restored = make_model(color_dim=color_dim, encoding=encoding)
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(restored.odeblock.odefunc(torch.tensor(.2), state),
                                               derivative, rtol=0, atol=0)
                    with self.assertRaises(RuntimeError):
                        make_model(encoding=encoding).load_state_dict(model.state_dict(), strict=True)

    def test_sh3_integration_and_augmentation_layout(self):
        initial = pack_gaussian_state(make_splats(sh_degree=3), COLOR)
        model = make_model(color_dim=48)
        bias = torch.linspace(-.1, .1, 48)
        with torch.no_grad():
            color_output_layer(model).bias.copy_(bias)
        times = torch.tensor([0., .25, 1.])
        trajectory = model(initial, times)
        torch.testing.assert_close(trajectory[..., 10:],
                                   initial[None, :, 10:] + times[:, None, None] * bias,
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model(trajectory[-1].detach(), times.flip(0))[-1],
                                   initial, rtol=1e-5, atol=1e-6)
        trajectory[-1, :, 13:].sum().backward()
        self.assertTrue((color_output_layer(model).bias.grad[3:] != 0).all())
        augmented = make_model(color_dim=48, augment_dim=2)
        with torch.no_grad():
            color_output_layer(augmented).bias.fill_(.25)
            augmented.odeblock.odefunc.mlp_aug[-1].bias.fill_(.75)
        derivative = augmented.odeblock.odefunc(
            torch.tensor(.2), torch.cat((initial, torch.zeros(4, 2)), -1))
        torch.testing.assert_close(derivative[:, 10:58], torch.full((4, 48), .25))
        torch.testing.assert_close(derivative[:, 58:], torch.full((4, 2), .75))

    def test_default_has_no_color_head_or_extra_state_keys(self):
        torch.manual_seed(12)
        model = make_model(color_dim=0)
        self.assertFalse(any("mlp_colors" in key for key in model.state_dict()))
        state = pack_gaussian_state(make_splats(), GEOMETRY)
        actual = model.odeblock.odefunc(torch.tensor(.2), state)
        ode = model.odeblock.odefunc
        hidden = torch.relu(ode.feature_out(ode.encoding_network_pos(state[:, :3], torch.full((4, 1), .2))))
        legacy = torch.cat((ode.mlp_means(hidden), ode.mlp_quats(hidden), ode.mlp_scales(hidden)), -1)
        self.assertTrue(torch.equal(actual, legacy))
        self.assertEqual(actual.shape, (4, 10))

    def test_color_head_starts_at_zero_and_receives_gradients(self):
        for encoding in ("hexplane", "freq", ""):
            with self.subTest(encoding=encoding):
                model = make_model(encoding=encoding)
                state = pack_gaussian_state(make_splats(), COLOR)
                derivative = model.odeblock.odefunc(torch.tensor(.2), state)
                self.assertEqual(derivative.shape, (4, 13))
                self.assertTrue(torch.equal(derivative[:, 10:], torch.zeros(4, 3)))
                derivative[:, 10:].sum().backward()
                self.assertGreater(color_output_layer(model).bias.grad.abs().sum().item(), 0)

    def test_augmented_state_keeps_colors_before_augmentation(self):
        model = make_model(augment_dim=2)
        ode = model.odeblock.odefunc
        with torch.no_grad():
            color_output_layer(model).bias.fill_(.25)
            ode.mlp_aug[-1].bias.fill_(.75)
        state = torch.cat([pack_gaussian_state(make_splats(), COLOR), torch.zeros(4, 2)], -1)
        derivative = ode(torch.tensor(.2), state)
        self.assertEqual(derivative.shape, (4, 15))
        torch.testing.assert_close(derivative[:, 10:13], torch.full((4, 3), .25))
        torch.testing.assert_close(derivative[:, 13:], torch.full((4, 2), .75))

    def test_forward_backward_integration_and_color_gradient(self):
        model = make_model()
        bias = torch.tensor([.2, -.1, .05])
        with torch.no_grad():
            color_output_layer(model).bias.copy_(bias)
        state = pack_gaussian_state(make_splats(), COLOR)
        times = torch.tensor([0., .2, .7, 1.])
        trajectory = model(state, times)
        self.assertTrue(torch.equal(trajectory[0], state))
        torch.testing.assert_close(trajectory[..., :10], state[None, :, :10].expand(4, -1, -1), rtol=0, atol=0)
        torch.testing.assert_close(trajectory[..., 10:], state[None, :, 10:] + times[:, None, None] * bias, rtol=1e-5, atol=1e-6)
        reverse = model(trajectory[-1].detach(), times.flip(0))
        torch.testing.assert_close(reverse[-1], state, rtol=1e-5, atol=1e-6)
        trajectory[-1, :, 10:].sum().backward()
        self.assertGreater(color_output_layer(model).bias.grad.abs().sum().item(), 0)

    def test_dynamic_checkpoint_round_trip(self):
        model = make_model()
        with torch.no_grad():
            color_output_layer(model).bias.fill_(.1)
        state = pack_gaussian_state(make_splats(), COLOR)
        buffer = io.BytesIO()
        torch.save({"neural_ode": model.state_dict(), "init_params": state}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = make_model()
        restored.load_state_dict(checkpoint["neural_ode"], strict=True)
        times = torch.tensor([0., .5])
        torch.testing.assert_close(restored(checkpoint["init_params"], times), model(state, times), rtol=0, atol=0)


class ColorRendererTests(unittest.TestCase):
    @staticmethod
    def fake_rasterization(**params):
        colors = params["colors"]
        batch = colors.shape[:-3]
        rgb = colors.mean(dim=(-3, -2))
        rendered = rgb[..., None, None, None, :].expand(*batch, params["viewmats"].shape[-3], params["height"], params["width"], 3)
        return rendered, torch.ones_like(rendered[..., :1]), {}

    def test_all_dynamic_renderers_use_evolving_colors(self):
        for method in ("rasterize_with_dynamic_params", "rasterize_with_dynamic_params_batched", "rasterize_with_dynamic_params_custom"):
            for activated in (True, False):
                with self.subTest(method=method, activated=activated):
                    splats = make_splats()
                    renderer = make_renderer(splats, True)
                    state = pack_gaussian_state(splats, COLOR)[None].repeat(2, 1, 1)
                    state[1, :, 10:] += .2
                    state.requires_grad_()
                    params = make_raster_params(splats, True)
                    with patch("models.gaussian_model.rasterization", side_effect=self.fake_rasterization) as raster:
                        images, _ = getattr(renderer, method)(state, params, activate_params=activated)
                    self.assertEqual(images.shape, (2, 2, 24, 32, 3))
                    torch.testing.assert_close(images[:, 1] - images[:, 0], torch.full((2, 24, 32, 3), .2))
                    images.sum().backward()
                    self.assertGreater(state.grad[..., 10:].abs().sum().item(), 0)
                    self.assertNotIn("colors", params)  # Caller data remains immutable.
                    self.assertEqual(raster.call_args.kwargs["colors"].shape[-3:], (4, 1, 3))

    def test_disabled_mode_reuses_fixed_colors(self):
        splats = make_splats()
        renderer = make_renderer(splats, False)
        params = make_raster_params(splats, False)
        state = pack_gaussian_state(splats, GEOMETRY)[None].repeat(2, 1, 1)
        with patch("models.gaussian_model.rasterization", side_effect=self.fake_rasterization) as raster:
            images, _ = renderer.rasterize_with_dynamic_params_batched(state, params)
        self.assertTrue(torch.equal(raster.call_args.kwargs["colors"], params["colors"][None].expand(2, -1, -1, -1)))
        self.assertTrue(torch.equal(images[:, 0], images[:, 1]))

    def test_foreground_only_color_rendering_does_not_use_full_scene_count(self):
        splats = make_splats()
        renderer = make_renderer(splats, True)
        state = pack_gaussian_state(splats, COLOR)[None, :2]
        params = make_raster_params(splats, True)
        params["opacities"] = params["opacities"][:2]
        with patch("models.gaussian_model.rasterization", side_effect=self.fake_rasterization) as raster:
            renderer.rasterize_with_dynamic_params_batched(state, params)
        self.assertEqual(raster.call_args.kwargs["colors"].shape, (1, 2, 1, 3))

    def test_sh3_all_renderers_preserve_coefficient_order_and_gradients(self):
        for method in ("rasterize_with_dynamic_params", "rasterize_with_dynamic_params_batched",
                       "rasterize_with_dynamic_params_custom"):
            with self.subTest(method=method):
                splats = make_splats(sh_degree=3)
                renderer = make_renderer(splats, True)
                state = pack_gaussian_state(splats, COLOR)[None, [2, 0]].repeat(2, 1, 1)
                state[1, :, 13:] += .2
                state.requires_grad_()
                params = make_raster_params(splats, True)
                params["opacities"] = params["opacities"][[2, 0]]
                with patch("models.gaussian_model.rasterization",
                           side_effect=self.fake_rasterization) as raster:
                    images, _ = getattr(renderer, method)(state, params)
                passed_colors = raster.call_args.kwargs["colors"]
                self.assertEqual(passed_colors.shape[-3:], (2, 16, 3))
                expected = unpack_gaussian_state(state, 48)["colors"]
                if method == "rasterize_with_dynamic_params":
                    expected = expected[-1]
                torch.testing.assert_close(passed_colors, expected, rtol=0, atol=0)
                images.sum().backward()
                self.assertTrue((state.grad[..., 13:] != 0).all())
                self.assertNotIn("colors", params)


class ColorConfigTests(unittest.TestCase):
    def test_color_flags_work_for_both_datasets_and_command_types(self):
        for config_type in (BlenderConfig, CapturedConfig):
            with self.subTest(config=config_type.__name__):
                cfg = training_config(config_type, ["default", "--learn-shs", "--sh-degree", "0"])
                self.assertTrue(cfg.learn_shs)
                self.assertEqual(cfg.sh_degree, 0)
                cfg = render_config(config_type, ["--learn-shs", "--sh-degree", "0"])
                self.assertTrue(cfg.learn_shs)
                self.assertFalse(config_type().learn_shs)

    def test_unsupported_combinations_fail_explicitly(self):
        for overrides in ({"sh_degree": -1}, {"sh_degree": 5}, {"app_opt": True}, {"version": "4dgs"}, {"learn_opacities": True}, {"learn_pos": False}, {"learn_quat": False}, {"learn_scales": False}):
            args = dict(learn_shs=True, sh_degree=0)
            args.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "--learn-shs"):
                BlenderConfig(**args)

    def test_higher_degree_cli_flags(self):
        for config_type in (BlenderConfig, CapturedConfig):
            for degree in range(1, 5):
                with self.subTest(config=config_type.__name__, degree=degree):
                    cfg = training_config(config_type, ["default", "--learn-shs", "--sh-degree", str(degree)])
                    rendered = render_config(config_type, ["--learn-shs", "--sh-degree", str(degree)])
                    self.assertTrue(cfg.learn_shs and rendered.learn_shs)
                    self.assertEqual(cfg.sh_degree, degree)
                    self.assertEqual(rendered.sh_degree, degree)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA smoke tests need a GPU")
class ColorFlowCUDATests(unittest.TestCase):
    def test_actual_boundary_global_checkpoint_and_inference_path(self):
        self._run_stage_smoke(use_bounding_box=True)

    def test_custom_scene_all_gaussians_global_with_adjoint(self):
        self._run_stage_smoke(use_bounding_box=False)

    def test_sh3_actual_stages_and_adjoint_with_and_without_selection(self):
        for use_bounding_box in (True, False):
            with self.subTest(use_bounding_box=use_bounding_box):
                self._run_stage_smoke(use_bounding_box, sh_degree=3)

    def _run_stage_smoke(self, use_bounding_box, sh_degree=0):
        """Run the production training loops on a tiny generated three-frame scene."""
        from datasets.blender import Dynamic_Dataset
        from trainers.trainer import Trainer

        torch.manual_seed(42)
        moving_count = 3 if use_bounding_box else 4
        splats = make_splats("cuda", sh_degree=sh_degree)
        color_dim = 3 * (sh_degree + 1) ** 2
        # Three foreground Gaussians in the existing clematis test box; one fixed
        # background Gaussian below it. No production scene/bounding box is edited.
        splats["means"] = torch.tensor([[-.1, 0, 1.6], [.1, 0, 1.7], [0, .1, 1.75], [0, 0, 1.1]], device="cuda")
        renderer = make_renderer(splats, True)
        initial = pack_gaussian_state(splats, COLOR)
        target_states = initial[None].repeat(3, 1, 1)
        target_states[:, :moving_count, 10:] += torch.tensor([0., .5, 1.], device="cuda")[:, None, None] * torch.tensor([.2, -.1, .1], device="cuda").repeat(color_dim // 3)
        params = make_raster_params(splats, True, cameras=1)
        with torch.no_grad():
            target_images, _ = renderer.rasterize_with_dynamic_params_batched(target_states, params)

        # Reuse the real dataset batching/collation implementation without disk I/O.
        dataset = Dynamic_Dataset.__new__(Dynamic_Dataset)
        dataset.times = [0., .5, 1.]
        dataset.images = {"r_0": list(target_images[0].cpu().numpy())}
        dataset.image_ids_dict = {"r_0": ["00000.png", "00001.png", "00002.png"]}
        dataset.cam_to_worlds_dict = {"r_0": params["viewmats"][0].cpu()}
        dataset.intrinsics = params["Ks"][0].cpu()
        dataset.image_width, dataset.image_height = 32, 24
        dataset.cam_batch_size = -1
        dataset.return_mask = dataset.is_monocular = dataset.include_zero = dataset.half_normalize = False
        dataset.prepend_zero = True

        real_loader = torch.utils.data.DataLoader

        def inline_loader(*args, **kwargs):
            kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
            return real_loader(*args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="growflow-color-stages-") as directory:
            cfg = BlenderConfig(
                learn_shs=True, sh_degree=sh_degree,
                data_dir="clematis_color_flow_test_fixture" if use_bounding_box else "custom_color_flow_test_fixture",
                use_bounding_box=use_bounding_box,
                result_dir=directory, min_iterations_req=3, dynamic_max_steps=2,
                dynamic_save_steps=[], dynamic_eval_steps=[], debug=False, wandb_every=0,
                compute_tv_loss_ours=False, reset_adam=False, near_plane=.01, far_plane=100.,
                adjoint=False,
            )
            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = "cuda"
            trainer.gaussians = renderer
            trainer.trainset = trainer.testset = dataset
            trainer.dynamical_model = make_model(color_dim=color_dim).cuda()
            trainer.wandb_run = Mock()
            trainer.evaluator = Mock()
            trainer.ckpt_dir = str(Path(directory) / "ckpts")
            Path(trainer.ckpt_dir).mkdir()
            # Skip plotting/encoding movies; keep training, state packing, caching,
            # differentiation, and checkpoint I/O real.
            trainer.visualize_fixed_pc_traj = Mock(return_value={})
            with patch("trainers.trainer.DataLoader", side_effect=inline_loader), patch("torch.utils.data.DataLoader", side_effect=inline_loader):
                with self.assertRaises(SystemExit) as completion:
                    trainer.generate_trajectory()
                self.assertEqual(completion.exception.code, 1)  # Existing completion convention.
                cached = np.load(Path(directory) / "fixed_pc_traj_3/full_traj_6.npy", allow_pickle=False)
                self.assertEqual(cached.shape, (3, 4, 10 + color_dim))
                self.assertTrue(np.isfinite(cached).all())
                np.testing.assert_array_equal(cached[0], initial.cpu().numpy())
                if use_bounding_box:
                    np.testing.assert_array_equal(cached[:, 3], np.repeat(initial[None, 3].cpu().numpy(), 3, axis=0))
                self.assertGreater(np.abs(cached[-1, :3, 10:] - cached[0, :3, 10:]).max(), 0)

                # Global optimization starts a fresh ODE from the boundary states.
                trainer.cfg.adjoint = True
                encoding = "hexplane" if use_bounding_box else "freq"
                trainer.dynamical_model = make_model(color_dim=color_dim, adjoint=True, encoding=encoding).cuda()
                trainer.train_dynamic_blender_masked(full_trajectory=cached)
            checkpoint = torch.load(Path(trainer.ckpt_dir) / "neural_ode_1.pt", map_location="cuda", weights_only=False)
            self.assertEqual(checkpoint["init_params"].shape, (4, 10 + color_dim))
            restored = make_model(color_dim=color_dim, encoding=encoding).cuda()
            restored.load_state_dict(checkpoint["neural_ode"], strict=True)
            with torch.no_grad():
                # Same foreground integration/reassembly used by evaluation.
                trajectory = checkpoint["init_params"][None].repeat(3, 1, 1)
                trajectory[:, :moving_count] = restored(checkpoint["init_params"][:moving_count], torch.tensor(dataset.times, device="cuda"))
                images, _ = renderer.rasterize_with_dynamic_params_batched(trajectory, params)
            self.assertEqual(images.shape, (1, 3, 24, 32, 3))
            self.assertTrue(torch.isfinite(images).all())
            self.assertGreater((images[:, 1:] - images[:, :1]).abs().max().item(), 0)
            if use_bounding_box:
                self.assertTrue(torch.equal(trajectory[:, 3], initial[None, 3].expand(3, -1)))
            else:
                self.assertGreater((trajectory[-1, 3, 10:] - initial[3, 10:]).abs().max().item(), 0)
            print(f"CUDA stage smoke: boundary 3x4x{10 + color_dim} -> adjoint global checkpoint -> inference; bbox={use_bounding_box}, IDs preserved")

    def test_ode_render_backward_and_color_only_learning(self):
        for degree in (0, 3):
            with self.subTest(sh_degree=degree):
                self._run_color_fit(degree)

    def _run_color_fit(self, sh_degree):
        torch.manual_seed(21)
        splats = make_splats("cuda", sh_degree=sh_degree)
        color_dim = 3 * (sh_degree + 1) ** 2
        renderer = make_renderer(splats, True)
        initial = pack_gaussian_state(splats, COLOR)
        model = make_model(color_dim=color_dim).cuda()
        params = make_raster_params(splats, True)
        times = torch.tensor([0., .5, 1.], device="cuda")
        target_state = initial[None].repeat(2, 1, 1)
        target_state[..., 10:] += torch.tensor([.15, -.1, .08], device="cuda").repeat(color_dim // 3)[None, None] * times[1:, None, None]
        with torch.no_grad():
            target, _ = renderer.rasterize_with_dynamic_params_batched(target_state, params)
        # Isolate color learning to prove photometric gradients reach the new head.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("mlp_colors" in name)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            trajectory = model(initial, times)
            images, _ = renderer.rasterize_with_dynamic_params_batched(trajectory[1:], params)
            loss = (images - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(color_output_layer(model).bias.grad).all())
            if sh_degree > 0 and not losses:
                self.assertGreater(color_output_layer(model).bias.grad[3:].abs().sum().item(), 0)
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * .2)
        self.assertTrue(torch.equal(trajectory[0], initial))
        torch.testing.assert_close(trajectory[..., :10], initial[None, :, :10].expand(3, -1, -1), rtol=0, atol=0)
        print(f"CUDA SH{sh_degree} color-fit loss: {losses[0]:.8g} -> {losses[-1]:.8g}")

    def test_sh3_anchor_render_is_identical_and_higher_orders_are_view_dependent(self):
        splats = make_splats("cuda", sh_degree=3)
        geometry = pack_gaussian_state(splats, GEOMETRY)
        colored = pack_gaussian_state(splats, COLOR)
        with torch.no_grad():
            old, _ = make_renderer(splats, False).rasterize_with_dynamic_params_batched(
                geometry[None], make_raster_params(splats, False))
            new, _ = make_renderer(splats, True).rasterize_with_dynamic_params_batched(
                colored[None], make_raster_params(splats, True))
        self.assertTrue(torch.equal(old, new))
        from gsplat.cuda._wrapper import spherical_harmonics
        coeffs = unpack_gaussian_state(colored, 48)["colors"]
        directions = torch.nn.functional.normalize(torch.randn(4, 3, device="cuda"), dim=-1)
        forward = spherical_harmonics(3, directions, coeffs)
        reverse = spherical_harmonics(3, -directions, coeffs)
        self.assertGreater((forward - reverse).abs().max().item(), 0)

    def test_existing_static_checkpoint_reused_without_changing_anchor_render(self):
        path = os.environ.get("GROWFLOW_STATIC_SMOKE_CKPT")
        if not path:
            self.skipTest("Set GROWFLOW_STATIC_SMOKE_CKPT for real-checkpoint verification")
        splats = torch.load(path, map_location="cuda", weights_only=False)["splats"]
        # Use every Gaussian in the checkpoint, without creating/removing/reordering IDs.
        geometry = pack_gaussian_state(splats, GEOMETRY)
        colored = pack_gaussian_state(splats, COLOR)
        self.assertTrue(torch.equal(geometry, colored[:, :10]))
        with torch.no_grad():
            old_image, _ = make_renderer(splats, False).rasterize_with_dynamic_params_batched(geometry[None], make_raster_params(splats, False))
            new_image, _ = make_renderer(splats, True).rasterize_with_dynamic_params_batched(colored[None], make_raster_params(splats, True))
        self.assertTrue(torch.equal(old_image, new_image))
        self.assertGreater(old_image.abs().sum().item(), 0)
        print(f"Existing static checkpoint: {colored.shape[0]} persistent IDs; identical SH{math.isqrt(1 + splats['shN'].shape[1]) - 1} anchor render")


if __name__ == "__main__":
    unittest.main()
