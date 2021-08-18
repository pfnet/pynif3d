import torch

from pynif3d import logger
from pynif3d.common.verification import check_equal, check_in_options, check_pos_int


def normalize_coordinate(point, padding=0.1, plane="xz", eps=1e-5):
    """
    The function to normalize coordinates to [0, 1], considering input within the limits
    of `[-0.5 * (1 + padding), +0.5 * (1 + padding)]`. The input limits are not strongly
    enforced.

    Args:
        point (Tensor): The tensor of the points to be normalized within given interval.
                        It has to have shape (batch_size, n_points, 3)

        padding (float): The ratio of padding to be applied within [lim_min, lim_max].
                         Default is 0.1.

        plane (str): The plane to apply the normalization.
                     Options are ("xy", "xz", "yz", "grid"). Default is "xz".

        eps: The epsilon to prevent zero-division. Default is 1e-5.

    Returns:
        Tensor of the points scaled and shifted to [lim_min, lim_max]
    """

    check_equal(len(point.shape), 3, "point.shape", "3")

    # Define plane to index coordinates
    coords = {"xz": [0, 2], "xy": [0, 1], "yz": [1, 2], "grid": [0, 1, 2]}

    # Check input values
    check_in_options(plane, coords.keys(), "plane")
    sub_points = point[:, :, coords[plane]].clone()

    # Apply padding
    sub_points = sub_points / (1 + padding + eps)
    sub_points += 0.5

    # Clip outliers
    sub_points = torch.clip(sub_points, 0, 1 - eps)

    return sub_points


def coordinate2index(coordinates, resolution):
    """
    The function to convert 2D / 3D coordinates into 1D indices. It supports only the
    square / cubical resolution.

    Args:
        coordinates (Tensor): 2D or 3D coordinates. axis=2 shall contain the coordinate
                              information.

        resolution (int): The spatial resolution of the coordinates to be parsed.

    Returns:
        Tensor containing 1D locations indices of the input coordinates.
    """

    if len(coordinates.shape) <= 2:
        msg = "coordinates has to have shape larger than 2."
        logger.error(msg)
        raise ValueError(msg)

    check_pos_int(resolution, "resolution")

    # Resize coordinates x = [0, 1] to x = [0, reso - 1]
    x1 = (coordinates * (resolution - 1)).long()

    # Iterate each elements in the axis to define the 1-dimensional indices
    index = 0
    for idx in range(1, coordinates.shape[2])[::-1]:
        index = resolution * (x1[:, :, idx] + index)

    # Add the x dimension value
    index += x1[:, :, 0]

    index = index[:, None, :]
    return index


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat and interleave a tensor along a given dimension. The current PyTorch
    implementation of `repeat_interleave` is slow and there is an open ticket for it:
    https://github.com/pytorch/pytorch/issues/31980
    Args:
        input (torch.Tensor): Tensor containing the input data.
        repeats (int): The number of repetitions for each element.
        dim (int). The dimension along which to repeat values.

    Returns:
        torch.Tensor: Repeated tensor which has the same shape as input, except along
            the given axis.
    """
    output = input.unsqueeze(dim + 1).expand(-1, repeats, *input.shape[1:])
    if dim == 0:
        output_shape = (-1,) + input.shape[1:]
    else:
        output_shape = input.shape[:dim] + (-1,) + input.shape[dim:]
    return output.reshape(output_shape)
