from unittest import TestCase

import numpy as np
import torch

from pynif3d import logger
from pynif3d.models import NeRFModel


class TestNeRFModel(TestCase):
    def test_forward(self):
        print("\nTesting forward pass of NeRF model...")
        model = NeRFModel(is_use_view_directions=False)
        self.assertIsNotNone(model)

        # Generate dummy data
        batch_size = 2
        dummy_input = torch.tensor(
            np.random.random((batch_size, 1024, 12, 3)), dtype=torch.float32
        )
        fw_res = model(dummy_input)

        self.assertEqual(fw_res.shape, (batch_size, 1024, 12, 4))
        self.assertEqual(fw_res.dtype, torch.float32)
        print("Forward pass of NeRF model is executed properly.\n")

    def test_forward_view_directions(self):
        print("\nTesting forward pass with view directions of NeRF model...")
        model = NeRFModel(is_use_view_directions=True)

        # Generate dummy data
        batch_size = 2
        dummy_input = torch.tensor(
            np.random.random((batch_size, 2048, 64, 3)), dtype=torch.float32
        )
        dummy_views = torch.tensor(
            np.random.random((batch_size, 2048, 3)), dtype=torch.float32
        )

        # Try to run model without supplying view directions
        # It shall throw exception
        is_error = False
        try:
            model(dummy_input, None)
        except Exception:
            is_error = True
        self.assertTrue(is_error)

        # Try to run model as expected
        fw_res = model(dummy_input, dummy_views)
        self.assertEqual(fw_res.shape, (batch_size, 2048, 64, 4))
        self.assertEqual(fw_res.dtype, torch.float32)
        print(
            "Forward pass of NeRF model with view directions is executed "
            "successfully.\n"
        )

    def test_convergence(self):
        logger.debug("\nTesting convergence of NeRF model...")
        model = NeRFModel(is_use_view_directions=False)
        model.train(True)

        # Generate dummy data
        batch_size = 2
        dummy_input = torch.tensor(
            np.random.random((batch_size, 12, 12, 3)), dtype=torch.float32
        )
        dummy_output = torch.tensor(
            np.random.random((batch_size, 12, 12, 4)), dtype=torch.float32
        )

        # Generate training info
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = np.inf
        is_converged = False

        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            dummy_output = dummy_output.cuda()

        for i in range(1000):
            fw_res = model(dummy_input, dummy_input)
            loss = torch.mean(torch.sqrt(torch.sum((fw_res - dummy_output) ** 2, -1)))

            if i == 0:
                initial_loss = loss

            if loss < initial_loss * 0.05:
                is_converged = True
                break

            logger.debug(loss)
            # Do one step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.assertTrue(is_converged)

        logger.debug("NeRF model is converged. \n")
