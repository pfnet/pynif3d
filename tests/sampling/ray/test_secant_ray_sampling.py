import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.sampling.ray.secant import SecantRaySampler


class TestSecantRaySampler(TestCase):
    def test_forward(self):
        filename = "dummy_secant_ray_sampling.npz"
        dummy_data = np.load(os.path.join(os.path.dirname(__file__), "data", filename))

        rays_d = torch.as_tensor(dummy_data["rays_d"])
        rays_o = torch.as_tensor(dummy_data["rays_o"])
        rays_m = torch.as_tensor(dummy_data["object_mask"])
        zs_min = torch.as_tensor(dummy_data["zs_min"])
        zs_max = torch.as_tensor(dummy_data["zs_max"])

        def dummy_sdf_model(point):
            return point.sum(dim=-1)

        ray_sampler = SecantRaySampler(dummy_sdf_model)
        points, z_vals, mask = ray_sampler(rays_d, rays_o, rays_m, zs_min, zs_max)

        expected_points = torch.as_tensor(dummy_data["secant_points"])
        expected_z_vals = torch.as_tensor(dummy_data["secant_z_vals"])
        expected_mask = torch.as_tensor(dummy_data["secant_mask"])

        self.assertTrue(torch.allclose(points, expected_points))
        self.assertTrue(torch.allclose(z_vals, expected_z_vals))
        self.assertTrue(torch.equal(mask, expected_mask))
