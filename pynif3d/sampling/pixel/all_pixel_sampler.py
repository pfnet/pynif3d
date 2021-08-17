import torch
import torch.nn as nn

from pynif3d.common.verification import check_pos_int
from pynif3d.log.log_funcs import func_logger


class AllPixelSampler(nn.Module):
    """
    The function that is used to sample all the elements of the given input 2D array.
    This function simply flattens a 2D array based on the [y, x] coordinates and returns
    each pixel as result.

    Usage:

    .. code-block:: python

        sampler = AllPixelSampler(image_height, image_width)
        sampled_data = sampler(rays_directions=rays_d, rays_origins=rays_o)

        rays_d = sampled_data["ray_directions"]
        rays_o = sampled_data["ray_origins"]
    """

    @func_logger
    def __init__(self, height, width):
        """
        Args:
            height (int): The height of the 2D array to be sampled. Positive integer.
            width (int): The width of the 2D array to be sampled. Positive integer.
        """
        super().__init__()

        self.height = height
        self.width = width

        check_pos_int(height, "height")
        check_pos_int(width, "width")

        self.coords = torch.stack(
            torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
            ),
            -1,
        )
        self.coords = torch.reshape(self.coords, [-1, 2])

    def forward(self, image=None, **kwargs):
        """

        Args:
            image (torch.Tensor): (Optional) The input 2D array to be sampled. Its shape
                is ``(1, 3, image_height, image_width)``.

        Returns:
            dict: Dictionary containing the sampled colors (RGB values) and pixel
            coordinates. The sampled colors are represented as `torch.Tensor` with
            shape ``(n_pixels, 3)``, while the sampled coordinates are represented
            as a `torch.Tensor` with shape ``(n_pixels, 2)``.
        """
        sample_coordinates = self.coords.long()

        ys = sample_coordinates[..., 0]
        xs = sample_coordinates[..., 1]

        sample_colors = None
        if image is not None:
            sample_colors = image[:, :, ys, xs]
            sample_colors = sample_colors.permute(0, 2, 1)

        output = {
            "sample_coordinates": sample_coordinates,
            "sample_colors": sample_colors,
        }

        for key, value in kwargs.items():
            output[key] = value[:, :, ys, xs].permute(0, 2, 1)

        return output
