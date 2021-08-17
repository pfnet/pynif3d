from unittest import TestCase

import torch

from pynif3d.loss import eikonal_loss


class TestLoss(TestCase):
    def test_eikonal_loss(self):
        x = torch.as_tensor(
            [
                [0.2936261892, -1.0289776325, 0.1445489526],
                [-0.2577984035, -0.7820385098, 0.3506951332],
                [-0.4243153632, 0.8669579029, -0.6295363903],
            ]
        )
        loss = float(eikonal_loss(x))
        expected_loss = 0.0135356029
        self.assertAlmostEqual(loss, expected_loss, places=5)
