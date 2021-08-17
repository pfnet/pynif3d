import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.aggregation.nerf_aggregator import NeRFAggregator


class TestNerfAggregator(TestCase):
    def test_forward(self):
        print("\nTesting NeRF aggregator...")

        aggregator = NeRFAggregator(background_color=torch.as_tensor((1, 1, 1)))

        file_path = os.path.join(
            os.path.dirname(__file__), "data/dummy_nerf_aggregator_data.npz"
        )
        dummy_output = np.load(file_path)

        dummy_rgb_map = torch.Tensor(dummy_output["rgb_map"])[None, :]
        dummy_depth_map = torch.Tensor(dummy_output["depth_map"])[None, :]
        dummy_disparity_map = torch.Tensor(dummy_output["disparity_map"])[None, :]
        dummy_alpha_map = torch.Tensor(dummy_output["alpha_map"])[None, :]
        dummy_weights = torch.Tensor(dummy_output["weights"])[None, :]
        dummy_raw = torch.Tensor(dummy_output["raw"])[None, :]
        dummy_z_vals = torch.Tensor(dummy_output["z_vals"])[None, :]
        dummy_rays_d = torch.Tensor(dummy_output["rays_d"])[None, :]

        rgb_map, depth_map, disparity_map, alpha_map, weights = aggregator(
            dummy_raw, dummy_z_vals, dummy_rays_d
        )

        atol = 1e-5
        np.testing.assert_allclose(rgb_map, dummy_rgb_map, atol=atol)
        np.testing.assert_allclose(depth_map, dummy_depth_map, atol=atol)
        np.testing.assert_allclose(disparity_map, dummy_disparity_map, atol=atol)
        np.testing.assert_allclose(alpha_map, dummy_alpha_map, atol=atol)
        np.testing.assert_allclose(weights, dummy_weights, atol=atol)
