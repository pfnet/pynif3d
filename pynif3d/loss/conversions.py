import math

import torch


def mse_to_psnr(mse_loss, max_intensity=1.0):
    """
    Converts a mean-squared error (MSE) loss to peak signal-to-noise ratio (PSNR).

    Args:
        mse_loss (torch.Tensor): MSE loss. Its shape is ``(1,)``.
        max_intensity (float): The maximum pixel intensity. Default value is 1.0.

    Returns:
        torch.Tensor: Tensor with the corresponding PSNR loss.
    """
    device = mse_loss.device

    if mse_loss == 0:
        psnr_loss = float("inf")
    else:
        psnr_loss = 20 * math.log10(max_intensity / math.sqrt(mse_loss))

    psnr_loss = torch.as_tensor([psnr_loss], device=device)
    return psnr_loss
