from unittest import TestCase

import torch

from pynif3d.pipeline.idr import IDR


class TestIDR(TestCase):
    def test_forward(self):
        # Create dummy data
        batch_size = 2
        height = 600
        width = 800
        focal_x = 555.555
        focal_y = 555.555
        center_x = 400
        center_y = 300

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        images = torch.rand(
            (batch_size, 3, height, width), dtype=torch.float32, device=device
        )
        images = images * 2.0 - 1.0
        object_masks = torch.rand((batch_size, 1, height, width), device=device)
        object_masks = object_masks > 0.5

        intrinsics = torch.zeros((batch_size, 4, 4), dtype=torch.float32, device=device)
        intrinsics[:, 0, 0] = focal_x
        intrinsics[:, 1, 1] = focal_y
        intrinsics[:, 0, 2] = center_x
        intrinsics[:, 1, 2] = center_y

        camera_poses = torch.eye(4, dtype=torch.float32, device=device)
        camera_poses[:3, 3] = 0.5
        camera_poses = camera_poses.repeat(batch_size, 1, 1)

        # Load model
        model = IDR((height, width))
        model.train()
        model.to(device)
        self.assertIsNotNone(model)

        # Test training
        prediction = model(images, object_masks, intrinsics, camera_poses)

        self.assertTrue("points" in prediction)
        self.assertTrue("rgb_vals" in prediction)
        self.assertTrue("mask" in prediction)
        self.assertTrue("z_pred" in prediction)
        self.assertTrue("gradient_theta" in prediction)
        self.assertTrue("sample_coordinates" in prediction)
