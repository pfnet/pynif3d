import torch

from pynif3d.common.verification import check_true


def eikonal_loss(x):
    """
    Computes the eikonal loss for a given set of points.

    Args:
        x (torch.Tensor): Tensor containing the point coordinates. Its shape is
            ``(batch_size, n_samples, 3)`` or ``(n_samples, 3)``.

    Returns:
        torch.Tensor: Tensor containing the eikonal loss. Its shape is ``(1,)``.
    """
    valid_shape = (x.ndim == 2 or x.ndim == 3) and (x.shape[-1] == 3)
    check_true(valid_shape, "valid_shape")

    loss = torch.tensor(0.0, device=x.device)
    if len(x) != 0:
        loss = ((x.reshape(-1, 3).norm(2, dim=-1) - 1) ** 2).mean()

    return loss
