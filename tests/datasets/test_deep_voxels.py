import unittest

import numpy as np

from pynif3d.datasets import DeepVoxels


class TestDeepVoxels(unittest.TestCase):
    def test_deep_voxels(self):
        data_directory = "/tmp/datasets/deep_voxels"
        scene = "cube"
        sizes = {"train": 1, "val": 1, "test": 1}
        image_shape = (512, 512, 3)

        for mode in ["train", "val", "test"]:
            dataset = DeepVoxels(data_directory, mode, scene)
            self.assertTrue(len(dataset) == sizes[mode])

            image, pose = dataset[0]
            self.assertTrue(image.dtype == np.float32)
            self.assertTrue(image.min() >= 0)
            self.assertTrue(image.max() <= 1)
            self.assertTrue(image.shape == image_shape)
            self.assertTrue(pose.dtype == np.float32)
