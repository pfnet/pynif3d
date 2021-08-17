from unittest import TestCase

import numpy as np
import torch

from pynif3d.models.con.unet3d import UNet3D


class TestUNet3DModel(TestCase):
    def test_forward(self):
        reso = 64
        in_ch = 16
        out_ch = 32
        model = UNet3D(
            output_channels=out_ch,
            input_channels=in_ch,
            feature_maps=8,
            num_levels=3,
            is_segmentation=False,
        )

        x = np.zeros((1, in_ch, reso, reso, reso))
        x[:, :, reso // 2 - 1, reso // 2 - 1, reso // 2 - 1] = np.nan
        x = torch.FloatTensor(x)

        out = model(x)
        all_nan_count = torch.sum(torch.isnan(out)).detach().cpu().numpy()
        self.assertTrue(all_nan_count == (reso ** 3 * out_ch))
