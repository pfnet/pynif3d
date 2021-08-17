import torch
import torch.nn as nn

from pynif3d.common.verification import check_equal, check_pos_int
from pynif3d.log.log_funcs import func_logger


class RandomPixelSampler(nn.Module):
    """
    Randomly samples N elements from a given 2D array as input.
    """

    @func_logger
    def __init__(self, height, width):
        """
        Args:
            height (int): Positive integer defining the height of the 2D array to be
                sampled.
            width (int): Positive integer defining the width of the 2D array to be
                sampled.

        Usage:

        .. code-block:: python

            sampler = RandomPixelSampler(image_height, image_width)
            sampled_data = sampler(rays_directions=rays_d, rays_origins=rays_o)

            rays_d = sampled_data["ray_directions"]
            rays_o = sampled_data["ray_origins"]
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

    def forward(self, n_sample, **kwargs):
        """
        Args:
            n_sample (int): Positive integer defining the number of samples to be
                queried.
            kwargs (dict): The (key, value) pairs for multiple values to be sampled at
                once.

        Returns:
            dict: Dictionary containing the same (key, value) pairs as `**kwargs`.
            It also contains the random sampling locations.
        """
        batch_size = 1
        if len(kwargs) > 0:
            values = list(kwargs.values())
            batch_size = len(values[0])
            for value in values[1:]:
                check_equal(len(value), batch_size, "len(value)", "batch_size")

        indices = torch.randint(0, len(self.coords), (batch_size, n_sample))
        sample_coordinates = self.coords[indices].long()

        output = {
            "indices": indices,
            "sample_coordinates": sample_coordinates,
        }

        ys = sample_coordinates[..., 0]
        xs = sample_coordinates[..., 1]

        for key, value in kwargs.items():
            output[key] = value[torch.arange(batch_size)[:, None], :, ys, xs]

        return output
