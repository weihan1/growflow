"""Check that reorganized public commands keep the training/render dispatch."""

import unittest
from unittest.mock import patch
import sys

from commands import render, train, trajectory
from configs.blender_config import Config as SyntheticConfig
from configs.captured_config import Config as CapturedConfig
from scripts.render_all import run_command


class CommandEntrypointTests(unittest.TestCase):
    @patch("trainers.runner.Runner")
    @patch("commands.train.display_config")
    def test_training_uses_synthetic_config_and_run(self, _, runner_type):
        train.main(["synthetic", "default", "--data-dir", "/tmp/synthetic-scene"])
        cfg = runner_type.call_args.args[0]
        self.assertIsInstance(cfg, SyntheticConfig)
        self.assertEqual(cfg.data_dir, "/tmp/synthetic-scene")
        runner_type.return_value.run.assert_called_once_with()

    @patch("trainers.runner.Runner")
    @patch("commands.trajectory.display_config")
    def test_trajectory_uses_captured_config_and_dispatch(self, _, runner_type):
        trajectory.main(["captured", "default", "--subsample-factor", "10"])
        cfg = runner_type.call_args.args[0]
        self.assertIsInstance(cfg, CapturedConfig)
        self.assertEqual(cfg.subsample_factor, 10)
        runner_type.return_value.generate_trajectory.assert_called_once_with()

    @patch("trainers.runner.Runner")
    @patch("commands.render.display_config")
    def test_captured_render_uses_only_test_cameras(self, _, runner_type):
        render.main(["captured", "--dynamic-ckpt", "/tmp/model.pt"])
        self.assertIsInstance(runner_type.call_args.args[0], CapturedConfig)
        self.assertEqual(runner_type.call_args.kwargs, {"load_test_cameras_only": True})
        runner_type.return_value.full_eval.assert_called_once_with()

    @patch("trainers.runner.Runner")
    @patch("commands.render.display_config")
    def test_synthetic_render_and_ground_truth_dispatch(self, _, runner_type):
        render.main(["synthetic", "--dynamic-ckpt", "/tmp/model.pt"])
        self.assertIsInstance(runner_type.call_args.args[0], SyntheticConfig)
        self.assertEqual(runner_type.call_args.kwargs, {"load_test_cameras_only": False})
        runner_type.return_value.full_eval.assert_called_once_with()

        runner_type.reset_mock()
        render.main(["captured", "--ground-truth"])
        self.assertEqual(runner_type.call_args.kwargs, {"load_test_cameras_only": False})
        runner_type.return_value.generate_gt.assert_called_once_with()

    @patch("scripts.render_all.subprocess.run")
    def test_batch_render_uses_the_current_python_environment(self, subprocess_run):
        subprocess_run.return_value.returncode = 0
        subprocess_run.return_value.stdout = ""
        subprocess_run.return_value.stderr = ""
        self.assertEqual(
            run_command("python -m commands.render captured --data-dir ./data/captured/scene"),
            0,
        )
        self.assertEqual(
            subprocess_run.call_args.args[0],
            [sys.executable, "-m", "commands.render", "captured", "--data-dir", "./data/captured/scene"],
        )


if __name__ == "__main__":
    unittest.main()
