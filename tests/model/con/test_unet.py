from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import UNet


class TestUNetModel(TestCase):
    def test_forward(self):
        model = UNet(
            1,
            network_depth=6,
            merge_mode="concat",
            input_channels=1,
            first_layer_channels=32,
        )

        reso = 256
        x = np.zeros((1, 1, reso, reso))
        x[:, :, int(reso / 2 - 1), int(reso / 2 - 1)] = np.nan
        x = torch.FloatTensor(x)

        out = model(x)
        all_nan_count = torch.sum(torch.isnan(out)).detach().cpu().numpy()
        self.assertTrue(all_nan_count == (reso * reso))
