import unittest

import numpy as np

from pynif3d.datasets import DTUMVSIDR


class TestDTUMVSIDR(unittest.TestCase):
    def test_dtu_mvs_idr(self):
        # Load the dataset.
        data_directory = "/tmp/datasets/dtu_mvs_idr"
        dataset = DTUMVSIDR(data_directory, mode="train", scan_id=65)

        # Check the dataset size.
        self.assertEqual(len(dataset), 49)

        # Check whether the data was loaded properly.
        identity = np.eye(3, dtype=np.float32)
        for batch in dataset:
            image, mask, intrinsics, camera_pose = batch
            rotation = camera_pose[:3, :3]

            self.assertTrue(image.dtype == np.float32)
            self.assertTrue(np.all(np.logical_and(image >= -1, image <= 1)))
            self.assertTrue(mask.dtype == bool)

            self.assertTrue(intrinsics.dtype == np.float32)
            self.assertTrue(intrinsics.shape == (4, 4))

            self.assertTrue(camera_pose.dtype == np.float32)
            self.assertTrue(camera_pose.shape == (4, 4))

            self.assertTrue(np.allclose(rotation @ rotation.T, identity, atol=1e-6))
