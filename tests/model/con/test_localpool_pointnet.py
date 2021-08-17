from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import PointNet_LocalPool


class TestPointNetLocalPool(TestCase):
    def test_forward(self):
        plane = "xz"
        feature_channel = 32
        plane_resolution = 256
        model = PointNet_LocalPool(
            feature_grids=[plane],
            feature_grid_resolution=plane_resolution,
            feature_grid_channels=feature_channel,
        )

        n_points = 3000
        input_points = np.random.random((1, n_points, 3)) / 2 + 0.5
        input_points = torch.FloatTensor(input_points)

        out = model(input_points)

        self.assertTrue(plane in out)  # Only xz plane
        self.assertTrue(out[plane].shape[1] == feature_channel)
        self.assertTrue(out[plane].shape[2] == plane_resolution)
        self.assertTrue(out[plane].shape[3] == plane_resolution)
