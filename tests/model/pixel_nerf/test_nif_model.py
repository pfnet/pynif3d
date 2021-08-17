from unittest import TestCase

import numpy as np
import torch

from pynif3d.models import PixelNeRFNIFModel
from pynif3d.utils.transforms import rotation_mat, translation_mat


class TestPixelNeRFNIFModel(TestCase):
    def test_forward(self):
        print("\nTesting forward pass of PixelNeRF NIF model...")

        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")

        n_samples = 8192
        n_views = 3
        feature_size = 512

        phis = torch.rand(n_views) * np.pi
        ts = torch.rand(n_views)
        rotations = np.stack([rotation_mat(phi, "x")[:3, :] for phi in phis])
        translations = np.stack([translation_mat(t)[:3, :] for t in ts])

        camera_poses = rotations
        camera_poses[:, :, 3] = translations[:, :, 3]
        camera_poses = torch.as_tensor(camera_poses, device=device)

        ray_points = torch.rand(n_samples, 3, device=device)
        view_dirs = torch.rand(n_samples, 3, device=device)
        features = torch.rand(n_views, n_samples, feature_size, device=device)

        model = PixelNeRFNIFModel(hidden_channels=feature_size)
        model.to(device)
        model.eval()
        self.assertIsNotNone(model)

        prediction = model(ray_points, camera_poses, features, view_dirs)
        self.assertEqual(prediction.shape, (n_samples, 4))
        self.assertEqual(prediction.dtype, torch.float32)

        print("Successfully executed the forward pass of SpatialEncoder.\n")
