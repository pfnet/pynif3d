import torch
import torch.nn as nn

from pynif3d.common.verification import check_not_none, check_true
from pynif3d.log.log_funcs import func_logger


class PointRenderer(nn.Module):
    """
    The function that is used for rendering each sampled point using the NIF model.

    Usage:

    .. code-block:: python

        # Assume that a NIF model (torch.nn.Module) is given.
        renderer = PointRenderer(chunk_size=256)
        prediction = renderer(nif_model)
    """

    @func_logger
    def __init__(self, chunk_size=None):
        """
        Args:
            chunk_size (int): The chunk size of the tensor that is passed for NIF
                prediction.
        """
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, nif_model, *args):
        """
        Args:
            nif_model (torch.nn.Module): NIF model for outputting the prediction.
            args (list, tuple): Tuple or list containing the tensors that are passed
                through the NIF model.

        Returns:
            torch.Tensor: Tensor containing the concatenated predictions.
        """
        # TODO: Add CUDA optimization for inference

        check_not_none(args, "args")
        n_rays = [a.shape[1] for a in args if isinstance(a, torch.Tensor)]
        same_batch_size = all(n == n_rays[0] for n in n_rays)
        check_true(same_batch_size, "same_batch_size")

        chunk_size = self.chunk_size
        if self.chunk_size is None:
            chunk_size = n_rays[0]

        nif_prediction = []
        for i in range(0, n_rays[0], chunk_size):
            chunk = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    chunk.append(a[:, i : (i + chunk_size), ...])
                else:
                    chunk.append(a)
            prediction = nif_model(*chunk)
            nif_prediction.append(prediction)

        nif_prediction = torch.cat(nif_prediction, 1)
        return nif_prediction
