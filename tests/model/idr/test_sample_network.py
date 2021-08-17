import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import IDRSampleNetwork


class TestIDRSampleNetwork(TestCase):
    def test_forward(self):
        file_path = os.path.join(os.path.dirname(__file__), "data/sample_network.npz")
        dummy_data = np.load(file_path, allow_pickle=True)

        rays_d = torch.as_tensor(dummy_data["rays_d"])
        rays_o = torch.as_tensor(dummy_data["rays_o"])
        z_vals = torch.as_tensor(dummy_data["z_vals"])
        z_pred = torch.as_tensor(dummy_data["z_pred"])

        mask_surface = torch.as_tensor(dummy_data["mask_surface"])
        z_at_surface = torch.as_tensor(dummy_data["z_at_surface"])
        grad_at_surface = torch.as_tensor(dummy_data["grad_at_surface"])
        expected = torch.as_tensor(dummy_data["points"])

        model = IDRSampleNetwork()
        prediction = model(
            rays_d, rays_o, z_vals, z_pred, mask_surface, z_at_surface, grad_at_surface
        )

        self.assertTrue(prediction.shape == expected.shape)
        self.assertTrue(prediction.dtype == torch.float32)
        self.assertTrue(torch.allclose(prediction, expected))
