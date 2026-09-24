"""Regression tests for captured image-metric tensor layout."""

import unittest

import torch

from metrics_captured import image_to_metric_input


class MetricInputLayoutTests(unittest.TestCase):
    def test_hwc_is_converted_to_nchw_without_swapping_spatial_axes(self) -> None:
        image = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

        converted = image_to_metric_input(image)

        self.assertEqual(converted.shape, (1, 4, 2, 3))
        torch.testing.assert_close(converted[0, :, 1, 2], image[1, 2, :])


if __name__ == "__main__":
    unittest.main()
