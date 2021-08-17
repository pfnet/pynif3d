from unittest import TestCase

import torch

from pynif3d.models import SpatialEncoder


class TestSpatialEncoder(TestCase):
    def test_forward(self):
        print("\nTesting forward pass of PixelNeRF local encoder...")

        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")

        batch_size = 3
        image_height = 300
        image_width = 400

        dummy_input = torch.rand((batch_size, 3, image_height, image_width))
        dummy_input = dummy_input.to(device)

        model = SpatialEncoder()
        model.to(device)
        model.eval()
        self.assertIsNotNone(model)

        latent = model(dummy_input)
        expected_shape = (batch_size, 512, image_height // 2, image_width // 2)

        self.assertEqual(latent.shape, expected_shape)
        self.assertEqual(latent.dtype, torch.float32)

        print("Successfully executed the forward pass of SpatialEncoder.\n")
