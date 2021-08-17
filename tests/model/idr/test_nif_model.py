from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import IDRNIFModel


class TestIDRNIFModel(TestCase):
    def test_forward(self):
        batch_size = 5
        n_rays = 3000
        input_channels = 3
        output_channels = 513

        model = IDRNIFModel(
            input_channels=input_channels, output_channels=output_channels
        )
        points = np.random.randn(batch_size, n_rays, input_channels)
        points = torch.FloatTensor(points)

        prediction = model(points)
        sdf_values = prediction["sdf_values"]
        features = prediction["features"]

        self.assertTrue(sdf_values.shape == (batch_size, n_rays))
        self.assertTrue(sdf_values.dtype == torch.float32)
        self.assertTrue(features.shape == (batch_size, n_rays, output_channels - 1))
        self.assertTrue(features.dtype == torch.float32)
