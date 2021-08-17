import unittest

import numpy as np

from pynif3d.datasets import LLFF


class TestLLFF(unittest.TestCase):
    def test_blender(self):
        data_directory = "/tmp/datasets/llff"
        scene = "flower"
        sizes = {"train": 34}
        image_shape = (3024, 4032, 3)

        for mode in ["train"]:
            dataset = LLFF(data_directory, mode, scene, factor=None)
            self.assertTrue(len(dataset) == sizes[mode])

            image, pose = dataset[0]
            self.assertTrue(image.dtype == np.float32)
            self.assertTrue(image.min() >= 0)
            self.assertTrue(image.max() <= 1)
            self.assertTrue(image.shape == image_shape)
            self.assertTrue(pose.dtype == np.float32)
