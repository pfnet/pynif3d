import unittest

import numpy as np

from pynif3d.datasets import Blender


class TestBlender(unittest.TestCase):
    def test_blender(self):
        data_directory = "/tmp/datasets/blender"
        scene = "lego"
        sizes = {"train": 1, "val": 1, "test": 1}

        image_shape_full = (800, 800, 4)
        image_shape_half = (400, 400, 4)

        for mode in ["train", "val", "test"]:
            # Test full resolution
            dataset_full = Blender(data_directory, mode, scene, half_resolution=False)
            self.assertTrue(len(dataset_full) == sizes[mode])

            image_full, camera_pose = dataset_full[0]
            self.assertTrue(image_full.dtype == np.float32)
            self.assertTrue(image_full.min() >= 0)
            self.assertTrue(image_full.max() <= 1)
            self.assertTrue(image_full.shape == image_shape_full)

            # Test half resolution
            dataset_half = Blender(data_directory, mode, scene, half_resolution=True)
            self.assertTrue(len(dataset_half) == sizes[mode])

            image_half, camera_pose = dataset_half[0]
            self.assertTrue(image_half.dtype == np.float32)
            self.assertTrue(image_half.min() >= 0)
            self.assertTrue(image_half.max() <= 1)
            self.assertTrue(image_half.shape == image_shape_half)

            self.assertTrue(camera_pose.dtype == np.float32)
