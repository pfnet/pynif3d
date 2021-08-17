import torch

from pynif3d.common import init_Linear
from pynif3d.common.verification import check_pos_int
from pynif3d.log.log_funcs import func_logger


class ResnetBlockFC(torch.nn.Module):
    """
    Implementation of the fully-connected ResNet block for Convolutional Occupancy
    Networks (CON), as described in:
    https://arxiv.org/abs/2003.04618

    .. note::

        This implementation is based on the original one, which can be found at:
        https://github.com/autonomousvision/convolutional_occupancy_networks

    It replaces convolutional layers of vanilla ResNet blocks with linear layers.

    Usage:

    .. code-block:: python

        input_channels = 32
        hidden_channels = 32
        output_channels = 32

        model = ResnetBlockFC(input_channels, output_channels, hidden_channels)
        features = model(x)
    """

    @func_logger
    def __init__(
        self,
        size_in: int,
        size_out: int,
        size_inner: int,
        activation_fn: torch.nn.Module = None,
        init_fc_0_kwargs: dict = None,
        init_fc_1_kwargs: dict = None,
        init_fc_s_kwargs: dict = None,
    ):
        super().__init__()
        """
        Args:
            size_in (int): The input dimensions of the ResNetFC block.
            size_out (int): The output dimensions of the ResNetFC block.
            size_inner (int): The inner dimensions of the ResNetFC block.
            activation_fn (torch.nn.Module): The activation function. If set to None, it
                will default to `torch.nn.ReLU`. Default is None.
            init_fc_0_kwargs (dict): Initialization parameters for the first linear
                layer. If set to None, no initialization parameters will be used.
                Default is None.
            init_fc_1_kwargs (dict): Initialization parameters for the second linear
                layer. If set to None, no initialization parameters will be used.
                Default is None.
            init_fc_s_kwargs (dict): Initialization parameters for the skip. If set to
                None, no initialization parameters will be used. Default is None.
        """

        check_pos_int(size_in, "size_in")
        check_pos_int(size_out, "size_out")
        check_pos_int(size_inner, "size_inner")

        if activation_fn is None:
            activation_fn = torch.nn.ReLU()
        self.activation_fn = activation_fn

        if init_fc_0_kwargs is None:
            init_fc_0_kwargs = {}
        self.fc_0 = init_Linear(size_in, size_inner, **init_fc_0_kwargs)

        if init_fc_1_kwargs is None:
            init_fc_1_kwargs = {}
        self.fc_1 = init_Linear(size_inner, size_out, **init_fc_1_kwargs)

        self.fc_s = None
        if size_in != size_out:
            if init_fc_s_kwargs is None:
                init_fc_s_kwargs = {
                    "bias": False,
                }
            self.fc_s = init_Linear(size_in, size_out, **init_fc_s_kwargs)

    def forward(self, x):
        """
        Args:
             x (torch.Tensor): Tensor with shape ``(batch_size, n_points, size_in)``.

        Returns:
            torch.Tensor: Tensor with shape ``(batch_size, n_points, size_out)``.
        """
        h = self.activation_fn(self.fc_0(x))
        h = self.activation_fn(self.fc_1(h))

        if self.fc_s:
            x_s = self.fc_s(x)
        else:
            x_s = x

        res = x_s + h
        return res
