import numpy as np
import torch
import torch.nn as nn

from pynif3d.log.log_funcs import func_logger


class FourierEncoding(nn.Module):
    """

    The implementation of the paper "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains". This class encodes input
    N-dimensional coordinates into M-dimensional gaussian distribution and applies
    trigonometric encoding.

    For more details, please check https://arxiv.org/abs/2006.10739.

    Usage:

    .. code-block:: python

        encoder = FourierEncoding()
        encoded_points = encoder(points)
    """

    @func_logger
    def __init__(
        self,
        input_dimensions=3,
        output_dimensions=256,
        scale=10,
    ):
        super().__init__()
        self.output_dimensions = output_dimensions
        self.distributions = torch.randn((output_dimensions, input_dimensions)) * scale

    def get_dimensions(self):
        return self.output_dimensions * 2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Its shape is ``(number_of_samples,
                input_dimensions)``.

        Returns:
            torch.Tensor: Tensor with shape
                ``(number_of_samples, 2 * output_dimensions)``.
        """
        x_proj = (2.0 * np.pi * x) @ self.distributions.T.to(x.device)
        res = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return res
