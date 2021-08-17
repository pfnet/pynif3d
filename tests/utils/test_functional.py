import os
import unittest

import numpy as np
import torch

from pynif3d.utils.functional import ray_sphere_intersection, secant


class TestFunctional(unittest.TestCase):
    def test_ray_sphere_intersection_radius(self):
        rays_d = torch.tensor([[[-0.9742, 0.2220, 0.0410], [-0.9801, 0.1710, 0.1010]]])
        rays_o = torch.as_tensor([[[0.1, 0.3, 0.2], [0.1, 0.3, 0.2]]])

        z_vals, mask = ray_sphere_intersection(rays_d, rays_o, radius=1.0)
        z_vals = z_vals.clamp_min(0.0).transpose(-1, 0)

        expected_z_vals = torch.tensor([[[0.0, 0.9502576590], [0.0, 0.9542506933]]])
        expected_mask = torch.as_tensor([[True, True]])

        self.assertTrue(torch.allclose(z_vals, expected_z_vals))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_secant(self):
        filename = "dummy_secant.npz"
        dummy_data = np.load(os.path.join(os.path.dirname(__file__), "data", filename))

        xs_min = torch.as_tensor(dummy_data["xs_min"])
        xs_max = torch.as_tensor(dummy_data["xs_max"])
        rays_o = torch.as_tensor(dummy_data["rays_o"])
        rays_d = torch.as_tensor(dummy_data["rays_d"])

        def secant_fn(zs):
            points = rays_o + zs[..., None] * rays_d
            zs = points.sum(dim=-1)
            return zs

        prediction = secant(secant_fn, xs_min, xs_max)
        expected = torch.as_tensor(dummy_data["xs_pred"])
        self.assertTrue(torch.allclose(prediction, expected))
