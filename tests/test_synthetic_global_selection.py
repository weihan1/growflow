"""Exercise the production selection branches without loading datasets or CUDA."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def selection_code(stage):
    filename = "trainer.py" if stage == "training" else "evaluate.py"
    method_name = {"training": "train_dynamic_blender_masked",
                   "evaluation": "dynamic_eval", "rendering": "full_dynamic_eval"}[stage]
    tree = ast.parse((ROOT / "trainers" / filename).read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == method_name)
    body = method.body
    if stage == "rendering":
        body = next(n.body for n in body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "not cfg.skip_test")
    if stage == "evaluation":
        branch = next(n for n in body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "cfg.train_interp and cfg.use_bounding_box")
        nodes = [body[body.index(branch) - 1], branch]
    else:
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test).startswith("not cfg.use_bounding_box"))
        end = next(i for i, n in enumerate(body[start:], start)
                   if isinstance(n, ast.If) and ast.unparse(n.test).startswith("cfg.use_bounding_box"))
        nodes = body[start:end + 1]
    return compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec")


class SyntheticGlobalSelectionTests(unittest.TestCase):
    def run_selection(self, stage, scene, use_box):
        initial = torch.arange(52, dtype=torch.float32).reshape(4, 13)
        mask = torch.tensor([True, False, True, False])
        prism = Mock(return_value=(None, mask))
        model = Mock(side_effect=lambda state, times: state[None].repeat(len(times), 1, 1) + 1)
        cfg = SimpleNamespace(data_dir=f"data/{scene}_transparent_final_small_vase_35_timesteps",
                              use_bounding_box=use_box, learn_masks=False, train_interp=True)
        scope = dict(torch=torch, os=os, cfg=cfg, scene=scene, means_t0=initial[:, :3],
                     fixed_init_params=initial, inp_t=torch.tensor([0., .5, 1.]),
                     select_points_in_prism=prism,
                     self=SimpleNamespace(dynamical_model=model,
                                          gaussians=SimpleNamespace(splats=SimpleNamespace(means=initial[:, :3]))))
        exec(selection_code(stage), scope)
        return scope, prism, model

    def test_no_box_keeps_all_ids_even_for_known_scene_names(self):
        for stage in ("training", "evaluation", "rendering"):
            for scene in ("non_monotonic3", "clematis"):
                with self.subTest(stage=stage, scene=scene):
                    scope, prism, model = self.run_selection(stage, scene, False)
                    prism.assert_not_called()
                    if stage == "evaluation":
                        self.assertEqual(model.call_args.args[0].shape, (4, 13))
                        torch.testing.assert_close(scope["pred_param"][0], scope["fixed_init_params"] + 1)
                        self.assertEqual(scope["scene"], scene)
                    else:
                        self.assertTrue(scope["bounding_box_mask"].all())
                        self.assertEqual(scope["bounding_box_mask"].dtype, torch.bool)
                        self.assertEqual(scope["bounding_box_mask"].device, scope["means_t0"].device)

    def test_existing_clematis_bounds_are_preserved(self):
        for stage in ("training", "evaluation", "rendering"):
            with self.subTest(stage=stage):
                scope, prism, model = self.run_selection(stage, "clematis", True)
                self.assertEqual(prism.call_count, 1)
                self.assertEqual(prism.call_args.args[1:], ([.015, 0., 1.678], (.350, .3, .5)))
                self.assertEqual(prism.call_args.kwargs["rotation_angles"], (0, 0, 0))
                if stage == "evaluation":
                    self.assertEqual(model.call_args.args[0].shape, (2, 13))
                    torch.testing.assert_close(scope["pred_param"][0, 1], scope["fixed_init_params"][1])

    def test_unknown_box_fails_with_actionable_message(self):
        for stage in ("training", "evaluation", "rendering"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "--no-use-bounding-box"):
                self.run_selection(stage, "non_monotonic3", True)


if __name__ == "__main__":
    unittest.main()
