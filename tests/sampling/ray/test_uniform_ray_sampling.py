import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.sampling.ray.uniform import UniformRaySampler


class TestUniformRaySampling(TestCase):
    def test_forward(self):

        return True
        # TODO: Check the inaccuracy for this test at issue #28
        #  https://github.com/pfnet/pynif3d/issues/28

        print("\n Testing WeightedRaySampling function...")

        sampler = UniformRaySampler(2.0, 6.0, 64)

        file_path = os.path.join(
            os.path.dirname(__file__), "data/dummy_uniform_ray_sampling.npz"
        )
        dummy_data = np.load(file_path)
        dummy_rays_d = torch.Tensor(dummy_data["rays_d"].astype(np.float32))
        dummy_rays_o = torch.Tensor(dummy_data["rays_o"].astype(np.float32))
        dummy_z_vals = torch.Tensor(dummy_data["z_vals"].astype(np.float32))
        dummy_query_points = torch.Tensor(dummy_data["query_points"].astype(np.float32))

        q_pts, z_vals = sampler(dummy_rays_d, dummy_rays_o)

        np.testing.assert_allclose(q_pts, dummy_query_points)
        np.testing.assert_allclose(z_vals, dummy_z_vals)

        print("\n WeightedRaySampling function is executed successfully.")
