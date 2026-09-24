"""Regression tests for captured-data test-camera-only loading."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from datasets.colmap import Dynamic_Dataset, Dynamic_Datasetshared


class _FakeParser:
    num_timesteps = 2
    test_every = 2
    white_bkgd = False

    def __init__(self) -> None:
        self.timestep_data = []
        for timestep in range(self.num_timesteps):
            self.timestep_data.append(
                {
                    "camera_ids": [1] * 5,
                    "image_paths": [
                        f"/images/{timestep}_{camera}.png" for camera in range(5)
                    ],
                    "image_ids": np.array([3, 1, 4, 0, 2]),
                    "masks_paths": [
                        f"/masks/{timestep}_{camera}.png" for camera in range(5)
                    ],
                    "params_dict": {1: np.empty(0, dtype=np.float32)},
                    "mapx_dict": {},
                    "mapy_dict": {},
                    "roi_undist_dict": {},
                    "camtoworlds": np.repeat(
                        np.eye(4, dtype=np.float32)[None], 5, axis=0
                    ),
                    "Ks_dict": {1: np.eye(3, dtype=np.float32)},
                }
            )

    def get_timestep_data(self, timestep: int):
        return self.timestep_data[timestep]


def _fake_imread(path: str) -> np.ndarray:
    if path.startswith("/masks/"):
        return np.full((2, 3), 255, dtype=np.uint8)
    camera = int(path.rsplit("_", 1)[1].split(".", 1)[0])
    return np.full((2, 3, 3), camera, dtype=np.uint8)


class TestCameraOnlyLoadingTests(unittest.TestCase):
    @patch("datasets.colmap.imageio.imread", side_effect=_fake_imread)
    def test_filtered_loader_matches_full_loader_test_split(self, imread) -> None:
        parser = _FakeParser()
        full_shared = Dynamic_Datasetshared(parser, debug_data_loading=True)
        full_test = Dynamic_Dataset(
            parser=parser,
            shared_data=full_shared.get_shared_data(),
            split="test",
        )

        filtered_shared = Dynamic_Datasetshared(
            parser,
            debug_data_loading=True,
            load_test_cameras_only=True,
        )
        filtered_test = Dynamic_Dataset(
            parser=parser,
            shared_data=filtered_shared.get_shared_data(),
            split="test",
        )

        for timestep in range(parser.num_timesteps):
            self.assertEqual(full_test.camera_filter[timestep], [3, 4, 2])
            self.assertEqual(
                filtered_test.camera_filter[timestep],
                full_test.camera_filter[timestep],
            )
            self.assertEqual(
                list(filtered_test.timestep_images[timestep]), [3, 4, 2]
            )
            for camera in filtered_test.camera_filter[timestep]:
                np.testing.assert_array_equal(
                    filtered_test.timestep_images[timestep][camera],
                    full_test.timestep_images[timestep][camera],
                )

        # Full loading reads 2 files for each of 5 cameras at 2 timesteps;
        # filtered loading reads 2 files for each of 3 test cameras.
        self.assertEqual(imread.call_count, 20 + 12)

    @patch("datasets.colmap.imageio.imread", side_effect=_fake_imread)
    def test_filtered_shared_data_rejects_training_split(self, _imread) -> None:
        parser = _FakeParser()
        filtered_shared = Dynamic_Datasetshared(
            parser,
            debug_data_loading=True,
            load_test_cameras_only=True,
        )

        with self.assertRaisesRegex(ValueError, "cannot create a training split"):
            Dynamic_Dataset(
                parser=parser,
                shared_data=filtered_shared.get_shared_data(),
                split="train",
            )


if __name__ == "__main__":
    unittest.main()
