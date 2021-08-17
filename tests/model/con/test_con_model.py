from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import ConvolutionalOccupancyNetworksModel


class TestCONModel(TestCase):
    def test_forward(self):
        batch_size = 5
        n_points = 3000
        pts_size = 3
        feat_size = 64

        model = ConvolutionalOccupancyNetworksModel(linear_channels=feat_size)
        q_pts = np.random.randn(batch_size, n_points, pts_size)
        q_pts = torch.FloatTensor(q_pts)

        feats = np.random.randn(batch_size, n_points, feat_size)
        feats = torch.FloatTensor(feats)

        out = model(q_pts, feats)

        self.assertTrue(out.shape, (batch_size, n_points))
