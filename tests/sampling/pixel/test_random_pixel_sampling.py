from unittest import TestCase

import numpy as np
import torch
from torch.testing import assert_allclose

from pynif3d.sampling.pixel.random_pixel_sampler import RandomPixelSampler


class TestRandomSampling(TestCase):
    def test_forward(self):
        print("\nTesting forward pass of random sampling...")

        width = 800
        height = 400
        channel = 3
        n_sample = 1024
        batch_size = 2

        sampler = RandomPixelSampler(height, width)

        # Generate dummy data
        dummy_input = torch.tensor(
            np.random.random((batch_size, channel, height, width)), dtype=torch.float32
        )

        res_dict = sampler(n_sample, data=dummy_input)
        data = res_dict["data"]
        sample_coordinates = res_dict["sample_coordinates"]

        xs = sample_coordinates[..., 0]
        ys = sample_coordinates[..., 1]

        self.assertTrue(data.shape == (batch_size, n_sample, 3))
        self.assertTrue(sample_coordinates.shape == (batch_size, n_sample, 2))
        assert_allclose(data, dummy_input[torch.arange(batch_size)[:, None], :, ys, xs])

        print("Random sampling is executed successfully.\n")
