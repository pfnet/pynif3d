import math

import numpy as np
import torch
from torch.nn.functional import softplus

from pynif3d import logger
from pynif3d.common.layer_generator import init_Linear
from pynif3d.common.verification import (
    check_callable,
    check_iterable,
    check_pos_int,
    check_true,
)
from pynif3d.encoding import PositionalEncoding
from pynif3d.log.log_funcs import func_logger
from pynif3d.models.idr.hyperparams import IDRHyperParams


class IDRNIFModel(torch.nn.Module):
    """
    The multi-layer MLP model for NIF representation. If provided, it applies positional
    encoding to the inputs and overrides the input channel information accordingly.

    .. note::

        Please check the paper for more information: https://arxiv.org/abs/2003.09852

    Usage:

    .. code-block:: python

        model = IDRNIFModel()
        pred_dict = model(points)
    """

    @func_logger
    def __init__(
        self,
        input_channels=3,
        output_channels=257,
        base_network_depth=8,
        base_network_channels=512,
        skip_layers=None,
        encoding_fn=None,
        is_encoding_active=True,
        normalize_weights=True,
        geometric_init=True,
        **kwargs
    ):
        super().__init__()

        hyperparams = kwargs.get("hyperparams", None)
        if hyperparams is None:
            hyperparams = IDRHyperParams()

        self.hyperparams = hyperparams

        self.input_channels = input_channels
        check_pos_int(input_channels, "input_channels")

        self.base_network_depth = base_network_depth
        check_pos_int(base_network_depth, "base_network_depth")

        self.base_network_channels = base_network_channels
        check_pos_int(base_network_channels, "base_network_channels")

        self.output_channels = output_channels
        check_pos_int(output_channels, "output_channels")

        self.skip_layers = skip_layers
        if self.skip_layers is None:
            self.skip_layers = [4]
        check_iterable(self.skip_layers, "skip_layers")

        self.is_encoding_active = is_encoding_active
        if is_encoding_active:
            logger.info("Input encoding is active. Overriding input channel size.")
            if self.is_encoding_active:
                self.encoding_fn = encoding_fn
                if self.encoding_fn is None:
                    self.encoding_fn = PositionalEncoding(
                        input_dimensions=input_channels,
                        num_frequency=6,
                        max_frequency=5,
                    )
                check_callable(self.encoding_fn, "get_dimensions", "encoding_fn")
                input_channels = self.encoding_fn.get_dimensions()
                self.input_channels = input_channels

        n_layers = base_network_depth + 2
        for layer_index in range(n_layers - 1):
            size_in = base_network_channels
            if layer_index == 0:
                size_in = input_channels

            size_out = base_network_channels
            if layer_index == n_layers - 2:
                size_out = output_channels

            if layer_index + 1 in self.skip_layers:
                size_out = size_out - input_channels

            if geometric_init:
                layer = self.init_geometric(size_in, size_out, layer_index, n_layers)
            else:
                layer = init_Linear(size_in, size_out)

            if normalize_weights:
                layer = torch.nn.utils.weight_norm(layer)

            setattr(self, "linear_" + str(layer_index), layer)

    def init_geometric(self, size_in, size_out, layer_index, n_layers):
        w_init_fn = torch.nn.init.normal_
        b_init_fn = torch.nn.init.constant_

        if layer_index == n_layers - 2:
            mean = np.sqrt(np.pi) / np.sqrt(size_in)
            std = 1e-4
            bias = -self.hyperparams.geometric_init.bias
        else:
            mean = 0.0
            std = np.sqrt(2) / np.sqrt(size_out)
            bias = 0.0

        layer = init_Linear(
            size_in=size_in,
            size_out=size_out,
            w_init_fn=w_init_fn,
            w_init_fn_args=(mean, std),
            b_init_fn=b_init_fn,
            b_init_fn_args=(bias,),
        )

        if self.is_encoding_active:
            if layer_index == 0:
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(
                    layer.weight[:, :3], mean=0.0, std=np.sqrt(2) / np.sqrt(size_out)
                )
            elif layer_index in self.skip_layers:
                torch.nn.init.constant_(
                    layer.weight[:, -(self.input_channels - 3) :], 0.0
                )

        return layer

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Tensor containing the points that are processed. Its
                shape is ``(batch_size, n_rays, 3)`` or ``(n_rays, 3)``.

        Returns:
            dict: Dictionary containing the computed SDF values (as torch.Tensor of
            shape ``(*points.shape[:-1])``) and feature vectors (as torch.Tensor of
            shape ``(*points.shape[:-1], output_channels - 1)``).
        """
        beta = self.hyperparams.softplus.beta

        valid_shape = (points.ndim == 2 or points.ndim == 3) and (points.shape[-1] == 3)
        check_true(valid_shape, "valid_shape")

        x = points.reshape(-1, 3)

        if self.is_encoding_active:
            x = self.encoding_fn(x)

        h = x

        for i in range(0, self.base_network_depth + 1):
            layer = getattr(self, "linear_" + str(i))

            if i in self.skip_layers:
                h = torch.cat([h, x], dim=-1) / math.sqrt(2)
            h = layer(h)

            if i < self.base_network_depth:
                h = softplus(h, beta=beta)

        sdf_vals = h[:, 0].reshape(*points.shape[:-1])
        features = h[:, 1:].reshape(*points.shape[:-1], -1)

        output_dict = {
            "sdf_vals": sdf_vals,
            "features": features,
        }

        return output_dict
