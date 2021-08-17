import os
import unittest

import numpy as np

from pynif3d.datasets import DTUMVSPixelNeRF


class TestDTUMVSPixelNeRF(unittest.TestCase):
    def test_dtu_mvs_pixel_nerf(self):
        # Load the dataset.
        data_directory = "/tmp/datasets/dtu_mvs_idr"
        mode = "train"
        scan_ids_file = os.path.join(data_directory, "DTU", "train.lst")
        scan_ids = [65]

        with open(scan_ids_file, "w") as stream:
            stream.writelines(["scan{}\n".format(scan_id) for scan_id in scan_ids])

        dataset = DTUMVSPixelNeRF(data_directory, mode, scan_ids_file)
        n_scans = 1

        self.assertEqual(len(dataset), n_scans)
        self.assertEqual(dataset.images[0].dtype, np.float32)
        self.assertEqual(dataset.intrinsics[0].dtype, np.float32)
        self.assertEqual(dataset.camera_poses[0].dtype, np.float32)
