import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.camera import SphereRayTracer


class TestSphereRayTracer(TestCase):
    def test_forward(self):
        filename = "dummy_sphere_ray_tracer.npz"
        dummy_data = np.load(os.path.join(os.path.dirname(__file__), "data", filename))

        rays_d = torch.as_tensor(dummy_data["rays_d"])
        rays_o = torch.as_tensor(dummy_data["rays_o"])

        def dummy_sdf_model(point):
            return point.sum(dim=-1)

        ray_tracer = SphereRayTracer(dummy_sdf_model)
        points, z_vals, mask_unfinished = ray_tracer(rays_d, rays_o)

        expected_points = torch.as_tensor(dummy_data["points"])
        expected_z_vals = torch.as_tensor(dummy_data["z_vals"])
        expected_mask_unfinished = torch.as_tensor(dummy_data["mask_unfinished"])

        self.assertTrue(torch.allclose(points, expected_points))
        self.assertTrue(torch.allclose(z_vals, expected_z_vals))
        self.assertTrue(torch.equal(mask_unfinished, expected_mask_unfinished))
