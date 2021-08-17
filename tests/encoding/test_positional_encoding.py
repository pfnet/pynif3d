import os
from unittest import TestCase

import numpy as np
import torch

from pynif3d.encoding.positional import PositionalEncoding


class TestPositionalEncoding(TestCase):
    def test_forward(self):
        print("\nTesting positional encoding...")
        encoding = PositionalEncoding()
        self.assertIsNotNone(encoding)

        # Create dummy data
        dummy_input = np.broadcast_to(np.linspace(-1, 1, 3)[None, :], (1, 3))
        dummy_input = torch.tensor(dummy_input, dtype=torch.float32)

        file_path = os.path.join(
            os.path.dirname(__file__), "data/dummy_positional_encoding_output.npy"
        )
        dummy_output = np.load(file_path)

        res = encoding(dummy_input)
        torch.testing.assert_allclose(res, dummy_output)

        print("Positional encoding is executed successfully. \n")
