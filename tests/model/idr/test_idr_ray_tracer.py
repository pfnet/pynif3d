import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.models.idr.ray_tracer import IDRRayTracer


class TestIDRRayTracer(TestCase):
    def test_forward(self):
        filename = "idr_ray_tracer.npz"
        dummy_data = np.load(os.path.join(os.path.dirname(__file__), "data", filename))

        rays_d = torch.as_tensor(dummy_data["rays_d"])[None, :]
        rays_o = torch.as_tensor(dummy_data["rays_o"])[None, :]
        object_mask = torch.as_tensor(dummy_data["object_mask"])[None, :]

        def sdf_model(point):
            sdf_vals = point.sum(dim=-1)
            return sdf_vals

        model = IDRRayTracer(sdf_model, debug=True)

        points, z_vals, network_mask = model(rays_d, rays_o, object_mask)
        points = points.reshape(-1, 3)
        z_vals = z_vals.reshape(-1)
        network_mask = network_mask.reshape(-1)

        expected_points = torch.as_tensor(dummy_data["points"]).reshape(-1, 3)
        expected_z_vals = torch.as_tensor(dummy_data["z_vals"]).reshape(-1)
        expected_network_mask = torch.as_tensor(dummy_data["network_mask"]).reshape(-1)

        self.assertTrue(torch.allclose(points, expected_points, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(z_vals, expected_z_vals, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.equal(network_mask, expected_network_mask))
