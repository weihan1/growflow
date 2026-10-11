import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.view_camera_poses import _rigid_rotation, discover_scenes, load_synthetic, load_thumbnail


class PoseViewerTests(unittest.TestCase):
    def test_discover_sibling_plants_from_scene_or_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "clematis_transparent_final_subsample_6"
            second = root / "rose_transparent_final"
            for scene in (first, second):
                scene.mkdir()
                (scene / "transforms_train.json").write_text("{}")
            from_scene, selected = discover_scenes(first, "synthetic")
            from_root, _ = discover_scenes(root, "synthetic")
            self.assertEqual(from_scene, from_root)
            self.assertEqual(selected, "clematis (subsample 6)")
            self.assertEqual(set(from_scene), {"clematis (subsample 6)", "rose (full)"})

    def test_discover_captured_plants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("pi_paperwhite_full_subset4", "pi_rose"):
                (root / name / "timelapse_20260101").mkdir(parents=True)
            scenes, selected = discover_scenes(root / "pi_rose", "captured")
            self.assertEqual(set(scenes), {"paperwhite full subset4", "rose"})
            self.assertEqual(selected, "rose")

    def test_synthetic_deduplicates_timesteps_and_converts_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "train" / "r_0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (200, 100)).save(image_dir / "00000.png")
            transform = np.eye(4)
            transform[:3, :3] *= 0.5
            frames = [
                {"file_path": f"./train/r_0/{frame}.png", "transform_matrix": transform.tolist()}
                for frame in ("00000", "00001")
            ]
            (root / "transforms_train.json").write_text(
                json.dumps({"camera_angle_x": 1.0, "frames": frames})
            )

            poses = load_synthetic(root, "train")
            self.assertEqual(len(poses), 1)
            self.assertEqual(poses[0].name, "r_0")
            self.assertEqual(poses[0].aspect, 2.0)
            np.testing.assert_allclose(poses[0].c2w[:3, :3], np.diag([0.5, -0.5, -0.5]))
            np.testing.assert_allclose(_rigid_rotation(poses[0].c2w), np.diag([1, -1, -1]))
            thumbnail, fov_y, aspect = load_thumbnail(poses[0])
            self.assertEqual(thumbnail.shape, (100, 200, 3))
            self.assertAlmostEqual(fov_y, 2 * np.arctan(np.tan(0.5) / 2))
            self.assertEqual(aspect, 2.0)

    def test_synthetic_rejects_changing_camera_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "train" / "r_0"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(image_dir / "00000.png")
            transforms = [np.eye(4), np.eye(4)]
            transforms[1][0, 3] = 1.0
            frames = [
                {"file_path": f"./train/r_0/{i:05d}.png", "transform_matrix": matrix.tolist()}
                for i, matrix in enumerate(transforms)
            ]
            (root / "transforms_train.json").write_text(
                json.dumps({"camera_angle_x": 1.0, "frames": frames})
            )
            with self.assertRaisesRegex(ValueError, "changing poses"):
                load_synthetic(root, "train")


if __name__ == "__main__":
    unittest.main()
