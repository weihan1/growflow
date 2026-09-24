"""Regression tests for the public training configuration contract.

These tests intentionally avoid importing ``trainers.runner`` or initializing CUDA. They
protect the dataclass defaults and Tyro CLI used by the documented entry points so
the configuration modules can be reorganized without silently changing behavior.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from typing import Any, Type

import tyro
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from configs.blender_config import Config as BlenderConfig
from configs.captured_config import Config as CapturedConfig


EXPECTED_CONTRACTS = {
    "blender": {
        "field_count": 228,
        "field_names_sha256": (
            "327d7d6d2a3ab6596c1ef10315a9a74a52499cf612ea326ab8625c8f4a9851d0"
        ),
        "defaults_sha256": (
            "7b54eec5718e0e217826a7618bf8aa37f5c5d14472c85edd9527a0fc4e2bcef0"
        ),
    },
    "captured": {
        "field_count": 244,
        "field_names_sha256": (
            "92fd60e5cc5dd975374c72b0fbe277f2e205d81933e3dcb7d152573f110b3949"
        ),
        "defaults_sha256": (
            "b01431d4098f83e0f0054c8c09b46125f8d07c5338bbed6f4c99f4af8b44cd9e"
        ),
    },
}

EXPECTED_BLENDER_ONLY_FIELDS = {
    "adjoint_train_all",
    "atol_train_all",
    "chamfer_reg_box",
    "compute_masked_psnr",
    "half_normalize",
    "metric_scene",
    "render_foreground",
    "render_only_foreground",
    "rtol_train_all",
    "train_interp",
    "use_intersection",
    "use_mask_projection",
    "use_mesh_vertices",
}

EXPECTED_CAPTURED_ONLY_FIELDS = {
    "align_timesteps",
    "apply_mask",
    "crop_imgs",
    "dates",
    "debug_data_loading",
    "dilation_iters",
    "end_until",
    "feature_out_output_dim",
    "flip_x",
    "flip_y",
    "flip_z",
    "include_end",
    "interpolation_factor",
    "param_loss_reg",
    "render_demo_viz",
    "render_interpolation_frames",
    "render_spacetime_viz",
    "save_pc_imgs",
    "skip_pc",
    "start_from",
    "subsample_factor",
    "use_bg_masks",
    "use_crops",
    "use_dense",
    "use_mask_intersection",
    "use_mask_proj",
    "use_mask_psnr",
    "use_own_impl",
    "viz_mask",
}


def _normalize(value: Any) -> Any:
    """Convert defaults to a deterministic, JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if dataclasses.is_dataclass(value):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _normalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    raise TypeError(f"Unsupported config default type: {type(value)!r}")


def _sha256(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _field_names(config_type: Type[Any]) -> list[str]:
    return sorted(field.name for field in dataclasses.fields(config_type))


def _defaults(config_type: Type[Any]) -> dict[str, Any]:
    config = config_type()
    return {
        field.name: _normalize(getattr(config, field.name))
        for field in dataclasses.fields(config_type)
    }


def _entrypoint_cli(config_type: Type[Any], args: list[str]) -> Any:
    """Reproduce the ``default``/``mcmc`` interface in the four entry scripts."""
    configs = {
        "default": (
            "Gaussian splatting with the default densification strategy.",
            config_type(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Gaussian splatting with MCMC densification.",
            config_type(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    return tyro.extras.overridable_config_cli(
        configs,
        args=args,
        console_outputs=False,
    )


class ConfigSchemaRegressionTests(unittest.TestCase):
    def test_only_one_initial_condition_is_supported(self) -> None:
        for config_type in (BlenderConfig, CapturedConfig):
            with self.subTest(config=config_type.__module__):
                self.assertEqual(config_type().num_init_conditions, 1)
                with self.assertRaisesRegex(ValueError, "Only one initial condition"):
                    config_type(num_init_conditions=2)

    def test_complete_schema_and_defaults_are_unchanged(self) -> None:
        config_types = {
            "blender": BlenderConfig,
            "captured": CapturedConfig,
        }

        for name, config_type in config_types.items():
            with self.subTest(config=name):
                expected = EXPECTED_CONTRACTS[name]
                names = _field_names(config_type)
                defaults = _defaults(config_type)

                self.assertEqual(len(names), expected["field_count"])
                self.assertEqual(
                    _sha256(names),
                    expected["field_names_sha256"],
                    f"{name} config fields changed; current fields: {names}",
                )
                self.assertEqual(
                    _sha256(defaults),
                    expected["defaults_sha256"],
                    (
                        f"{name} config defaults changed. If intentional, review the "
                        "new defaults and update EXPECTED_CONTRACTS.\n"
                        f"Current normalized defaults:\n{json.dumps(defaults, indent=2, sort_keys=True)}"
                    ),
                )

    def test_shared_and_dataset_specific_fields_are_unchanged(self) -> None:
        blender_fields = set(_field_names(BlenderConfig))
        captured_fields = set(_field_names(CapturedConfig))

        self.assertEqual(len(blender_fields & captured_fields), 215)
        self.assertEqual(blender_fields - captured_fields, EXPECTED_BLENDER_ONLY_FIELDS)
        self.assertEqual(captured_fields - blender_fields, EXPECTED_CAPTURED_ONLY_FIELDS)

    def test_high_impact_defaults_are_readable(self) -> None:
        blender = BlenderConfig()
        captured = CapturedConfig()

        self.assertEqual(
            {
                "data_type": blender.data_type,
                "init_type": blender.init_type,
                "data_factor": blender.data_factor,
                "encoding": blender.encoding,
                "hidden_dim": blender.hidden_dim,
                "hidden_depth": blender.hidden_depth,
                "spatial_temp_resolution": blender.spatial_temp_resolution,
                "learn_masks": blender.learn_masks,
                "return_mask": blender.return_mask,
                "use_bounding_box": blender.use_bounding_box,
                "use_wandb": blender.use_wandb,
                "run_eval": blender.run_eval,
                "static_max_steps": blender.static_max_steps,
                "dynamic_max_steps": blender.dynamic_max_steps,
            },
            {
                "data_type": "blender",
                "init_type": "blender_pts",
                "data_factor": 4,
                "encoding": "hexplane",
                "hidden_dim": 64,
                "hidden_depth": 3,
                "spatial_temp_resolution": [64, 64, 64, 25],
                "learn_masks": False,
                "return_mask": False,
                "use_bounding_box": True,
                "use_wandb": False,
                "run_eval": True,
                "static_max_steps": 30_000,
                "dynamic_max_steps": 30_000,
            },
        )
        self.assertEqual(
            {
                "data_type": captured.data_type,
                "init_type": captured.init_type,
                "data_factor": captured.data_factor,
                "encoding": captured.encoding,
                "hidden_dim": captured.hidden_dim,
                "hidden_depth": captured.hidden_depth,
                "spatial_temp_resolution": captured.spatial_temp_resolution,
                "learn_masks": captured.learn_masks,
                "return_mask": captured.return_mask,
                "use_bounding_box": captured.use_bounding_box,
                "use_wandb": captured.use_wandb,
                "run_eval": captured.run_eval,
                "static_max_steps": captured.static_max_steps,
                "dynamic_max_steps": captured.dynamic_max_steps,
            },
            {
                "data_type": "colmap",
                "init_type": "sfm",
                "data_factor": 1,
                "encoding": "hexplane",
                "hidden_dim": 256,
                "hidden_depth": 8,
                "spatial_temp_resolution": [64, 64, 64, 150],
                "learn_masks": True,
                "return_mask": True,
                "use_bounding_box": True,
                "use_wandb": True,
                "run_eval": False,
                "static_max_steps": 30_000,
                "dynamic_max_steps": 30_000,
            },
        )


class ConfigCliRegressionTests(unittest.TestCase):
    def test_blender_default_cli_overrides(self) -> None:
        config = _entrypoint_cli(
            BlenderConfig,
            [
                "default",
                "--data-dir",
                "/tmp/synthetic-scene",
                "--no-adjoint",
                "--dynamic-max-steps",
                "123",
            ],
        )

        self.assertEqual(config.data_dir, "/tmp/synthetic-scene")
        self.assertFalse(config.adjoint)
        self.assertEqual(config.dynamic_max_steps, 123)
        self.assertIsInstance(config.strategy, DefaultStrategy)
        self.assertTrue(config.strategy.verbose)

    def test_captured_default_cli_overrides(self) -> None:
        config = _entrypoint_cli(
            CapturedConfig,
            [
                "default",
                "--data-dir",
                "/tmp/captured-scene",
                "--subsample-factor",
                "10",
                "--include-end",
                "--encoding",
                "freq",
                "--no-adjoint",
            ],
        )

        self.assertEqual(config.data_dir, "/tmp/captured-scene")
        self.assertEqual(config.subsample_factor, 10)
        self.assertTrue(config.include_end)
        self.assertEqual(config.encoding, "freq")
        self.assertFalse(config.adjoint)
        self.assertIsInstance(config.strategy, DefaultStrategy)

    def test_mcmc_preset_is_preserved(self) -> None:
        for config_type in (BlenderConfig, CapturedConfig):
            with self.subTest(config=config_type.__module__):
                config = _entrypoint_cli(config_type, ["mcmc"])
                self.assertIsInstance(config.strategy, MCMCStrategy)
                self.assertTrue(config.strategy.verbose)
                self.assertEqual(config.init_opa, 0.5)
                self.assertEqual(config.init_scale, 0.1)
                self.assertEqual(config.opacity_reg, 0.01)
                self.assertEqual(config.scale_reg, 0.01)


class ConfigStepScalingRegressionTests(unittest.TestCase):
    def test_adjust_steps_preserves_current_semantics(self) -> None:
        for config_type in (BlenderConfig, CapturedConfig):
            with self.subTest(config=config_type.__module__):
                config = config_type()
                original_dynamic_max_steps = config.dynamic_max_steps
                config.adjust_steps(0.5)

                self.assertEqual(config.static_max_steps, 15_000)
                self.assertEqual(config.static_eval_steps[:4], [0, 1_750, 3_500, 7_500])
                self.assertEqual(config.static_save_steps[:3], [0, 3_500, 15_000])
                self.assertEqual(config.static_ply_steps, [3_500, 15_000])
                self.assertEqual(config.sh_degree_interval, 500)
                self.assertEqual(config.strategy.refine_start_iter, 250)
                self.assertEqual(config.strategy.refine_stop_iter, 7_500)
                self.assertEqual(config.strategy.reset_every, 1_500)
                self.assertEqual(config.strategy.refine_every, 50)

                # ``adjust_steps`` currently scales static training and densification only.
                self.assertEqual(config.dynamic_max_steps, original_dynamic_max_steps)


if __name__ == "__main__":
    unittest.main()
