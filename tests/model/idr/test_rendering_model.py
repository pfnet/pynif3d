from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import IDRRenderingModel


class TestIDRRenderingModel(TestCase):
    def test_forward(self):
        batch_size = 5
        n_rays = 3000

        input_channel_points = 3
        input_channel_features = 256
        input_channel_normals = 3
        input_channel_view_dirs = 3
        output_channels = 3

        model = IDRRenderingModel(
            input_channel_points=input_channel_points,
            input_channel_features=input_channel_features,
            input_channel_normals=input_channel_normals,
            input_channel_view_dirs=input_channel_view_dirs,
            output_channels=output_channels,
        )

        points = np.random.randn(batch_size, n_rays, input_channel_points)
        points = torch.FloatTensor(points)

        features = np.random.randn(batch_size, n_rays, input_channel_features)
        features = torch.FloatTensor(features)

        view_dirs = np.random.randn(batch_size, n_rays, input_channel_view_dirs)
        view_dirs = torch.FloatTensor(view_dirs)

        normals = np.random.randn(batch_size, n_rays, input_channel_normals)
        normals = torch.FloatTensor(normals)

        prediction = model(points, features, normals, view_dirs)

        self.assertTrue(prediction.shape == (batch_size, n_rays, output_channels))
        self.assertTrue(prediction.dtype == torch.float32)
