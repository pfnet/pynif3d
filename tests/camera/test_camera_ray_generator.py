from unittest import TestCase

import torch

from pynif3d.camera.camera_ray_generator import CameraRayGenerator


class TestCameraRayGenerator(TestCase):
    def test_forward(self):
        height = 400
        width = 800
        focal_x = focal_y = 555.555

        generator = CameraRayGenerator(
            height,
            width,
            focal_x,
            focal_y,
        )
        pose = torch.tensor(
            [
                [
                    [6.5707326e-01, -5.1133215e-01, 5.5388999e-01, 2.2328019e00],
                    [7.5382674e-01, 4.4570285e-01, -4.8279837e-01, -1.9462224e00],
                    [-1.4901161e-08, 7.3477095e-01, 6.7831522e-01, 2.7343762e00],
                ]
            ],
            dtype=torch.float32,
        )

        rays_d, rays_o, view_dirs = generator(pose)

        self.assertTrue(rays_d.shape == (1, 3, height, width))
        self.assertTrue(rays_o.shape == (1, 3, height, width))
        self.assertTrue(view_dirs.shape == (1, 3, height, width))
