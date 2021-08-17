import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.pipeline.nerf import NeRF


class TestNeRF(TestCase):
    def test_forward(self):
        # Create dummy data
        # NeRF takes camera pose as input
        # It gets the image as GT signal
        file_path = os.path.join(os.path.dirname(__file__), "data/dummy_nerf_data.npz")
        data = np.load(file_path)
        height = int(data["height"]) * 2
        width = int(data["width"])
        focal_x = float(data["focal_x"])
        focal_y = float(data["focal_y"])

        # Load model
        model = NeRF((height, width), (focal_x, focal_y))
        self.assertIsNotNone(model)

        # Test training
        pose = torch.as_tensor(data["pose"][None, :])
        ret = model(pose)

        self.assertTrue("rgb_map" in ret)
        self.assertTrue("depth_map" in ret)
        self.assertTrue("disparity_map" in ret)
        self.assertTrue("alpha_map" in ret)
