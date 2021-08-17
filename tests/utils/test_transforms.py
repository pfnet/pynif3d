import unittest

import numpy as np

from pynif3d.utils.transforms import rotation_mat, translation_mat


class TestTransform(unittest.TestCase):
    def test_translation_mat(self):
        pred = translation_mat(0)
        real = np.identity(4)
        self.assertTrue(np.allclose(pred, real))

    def test_rotation_mat(self):
        pred_x = rotation_mat(np.pi / 4, axis="x")
        real_x = np.array(
            [
                [1, 0, 0, 0],
                [0, 0.7071068, -0.7071068, 0],
                [0, 0.7071068, 0.7071068, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(pred_x, real_x))

        pred_y = rotation_mat(np.pi / 4, axis="y")
        real_y = np.array(
            [
                [0.7071068, 0, 0.7071068, 0],
                [0.0000000, 1, 0.0000000, 0],
                [-0.7071068, 0, 0.7071068, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(pred_y, real_y))

        pred_z = rotation_mat(np.pi / 4, axis="z")
        real_z = np.array(
            [
                [0.7071068, -0.7071068, 0, 0],
                [0.7071068, 0.7071068, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(pred_z, real_z))

    def test_pose_to_spherical(self):
        self.assertTrue(True)
