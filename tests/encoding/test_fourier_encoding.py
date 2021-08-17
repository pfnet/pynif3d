from unittest import TestCase

import numpy as np
import torch

from pynif3d.encoding.fourier import FourierEncoding


class TestFourierEncoding(TestCase):
    def test_forward(self):
        print("\nTesting fourier encoding...")
        encoding = FourierEncoding(3, 256)

        self.assertIsNotNone(encoding)

        # Create dummy data
        dummy_input = np.broadcast_to(np.linspace(-1, 1, 3)[None, :], (1, 3))
        dummy_input = torch.tensor(dummy_input, dtype=torch.float32)

        res = encoding(dummy_input)
        torch.testing.assert_allclose(res.shape[-1], 512)

        print("Fourier encoding is executed successfully. \n")
