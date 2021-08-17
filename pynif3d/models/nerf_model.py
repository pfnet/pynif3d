import torch
import torch.nn as nn

from pynif3d import logger
from pynif3d.common.layer_generator import init_Linear
from pynif3d.common.verification import (
    check_callable,
    check_iterable,
    check_not_none,
    check_pos_int,
)
from pynif3d.encoding import PositionalEncoding
from pynif3d.log.log_funcs import func_logger


class NeRFModel(nn.Module):
    """
    The multi-layer MLP model for NeRF rendering. If provided, it applies positional
    encoding to the inputs overrides the input channel information accordingly. It can
    also integrate view direction information into the network.

    Usage:

    .. code-block:: python

        model = NeRF()
        prediction = model(points, view_dirs)
    """

    @func_logger
    def __init__(
        self,
        input_channels=3,
        input_channel_view_dirs=3,
        output_channels=4,
        base_network_depth=8,
        base_network_channels=256,
        skip_layers=None,
        is_use_view_directions=True,
        view_dir_network_depth=1,
        view_dir_network_channels=256,
        encoding_fn=None,
        encoding_viewdir_fn=None,
        is_input_encoding_active=True,
        init_kwargs=None,
        normalize_weights=False,
    ):
        """
        Args:
            input_channels (int): The input channel dimension to the model. If
                positional encoding is used, this value will be overridden. Default is
                3 (XYZ).
            input_channel_view_dirs (int): The input channel dimension for viewing
                directions. If positional encoding is used, this value will be
                overridden. Default is 3 (XYZ).
            output_channels (int): The output channel dimension. Default is 4 (RGBA).
            base_network_depth (int): The depth of the network MLP layers. One linear
                layer will be added to the base network for each increment. Default is
                8.
            base_network_channels (int): The output dimension of each inner linear
                layers of the MLP model. A positive integer value is expected. Default
                is 256.
            skip_layers (Iterable): The layers to add skip connection. It shall be an
                iterable of positive integers. Values larger than `network_depth` will
                be discarded. Default is [4,].
            is_use_view_directions (bool): Boolean flag indicating whether to use view
                direction (True) or not (False). If True, the view direction block will
                be added on top of the base MLP layers. Default is True.
            view_dir_network_depth (int): The depth of the network that processes view
                directions. One linear layer for processing view direction will be added
                to the network for each increment. Default value is 1.
            view_dir_network_channels (int): The output dimension of each inner linear
                layers of the MLP model which processes view directions. A positive
                integer is expected. Default value is 256.
            encoding_fn (torch.nn.Module): The function that is called in order to apply
                encoding to the NIF model input. Default is `PositionalEncoding()`.
            encoding_viewdir_fn (torch.nn.Module): The function that is called in order
                to apply encoding to the view directions input. Default is
                `PositionalEncoding()`.
            is_input_encoding_active (bool): Boolean flag indicating whether encoding
                shall be applied to both the base network input and view directions.
                Default is True.
            init_kwargs (dict): Dictionary containing the initialization parameters for
                the linear layers.
            normalize_weights (bool): Boolean flag indicating whether to normalize the
                linear layer's weights (True) or not (False).
        """
        super().__init__()

        if init_kwargs is None:
            init_kwargs = {}

        check_pos_int(base_network_channels, "base_network_channels")
        self.base_network_depth = base_network_depth
        check_pos_int(base_network_depth, "base_network_depth")

        self.is_use_view_directions = is_use_view_directions
        self.view_dir_network_depth = view_dir_network_depth

        self.skip_layers = skip_layers
        if self.skip_layers is None:
            self.skip_layers = [4]

        check_iterable(self.skip_layers, "skip_layers")

        self.is_input_encoding_active = is_input_encoding_active
        if self.is_input_encoding_active:
            logger.info("Input encoding is active. Overriding input channel size.")
            if self.is_use_view_directions:
                self.encoding_viewdir_fn = encoding_viewdir_fn
                if self.encoding_viewdir_fn is None:
                    self.encoding_viewdir_fn = PositionalEncoding(
                        input_dimensions=input_channel_view_dirs,
                        num_frequency=4,
                        max_frequency=3,
                    )
                check_callable(
                    self.encoding_viewdir_fn, "get_dimensions", "encoding_viewdir_fn"
                )
                input_channel_view_dirs = self.encoding_viewdir_fn.get_dimensions()

            self.encoding_fn = encoding_fn
            if self.encoding_fn is None:
                self.encoding_fn = PositionalEncoding(input_dimensions=input_channels)
            check_callable(self.encoding_fn, "get_dimensions", "encoding_fn")
            input_channels = self.encoding_fn.get_dimensions()

        # Define the base network.
        self.input_channels = input_channels

        for layer_index in range(base_network_depth):
            size_in = base_network_channels
            if layer_index == 0:
                size_in = input_channels

            size_out = base_network_channels
            if layer_index - 1 in self.skip_layers:
                size_in = base_network_channels + input_channels

            layer = init_Linear(size_in, size_out, **init_kwargs)
            if normalize_weights:
                torch.nn.utils.weight_norm(layer)

            setattr(self, "linear_" + str(layer_index), layer)

        # Define the network for view directions.
        self.input_channel_view_dirs = input_channel_view_dirs

        if is_use_view_directions:
            self.alpha_out = init_Linear(base_network_channels, 1, **init_kwargs)
            self.bottleneck = init_Linear(
                base_network_channels, view_dir_network_channels, **init_kwargs
            )

            if view_dir_network_depth > 0:
                for layer_index in range(view_dir_network_depth):
                    size_in = view_dir_network_channels // 2
                    if layer_index == 0:
                        size_in = base_network_channels + input_channel_view_dirs

                    size_out = view_dir_network_channels // 2
                    if layer_index - 1 in self.skip_layers:
                        size_in = base_network_channels + input_channels

                    layer = init_Linear(size_in, size_out, **init_kwargs)
                    setattr(self, "vd_linear_" + str(layer_index), layer)

                self.prediction = init_Linear(
                    view_dir_network_channels // 2, output_channels - 1, **init_kwargs
                )
            else:
                self.prediction = init_Linear(
                    view_dir_network_channels, output_channels - 1, **init_kwargs
                )
        else:
            self.prediction = init_Linear(
                base_network_channels, output_channels, **init_kwargs
            )

        if normalize_weights:
            self.prediction = torch.nn.utils.weight_norm(self.prediction)

    def forward(self, query_points, view_dirs=None):
        """
        Args:
            query_points (torch.Tensor): Tensor containing the points to that are
                queried. Its shape is ``(number_of_rays, number_of_points_per_ray,
                point_dims)``.
            view_dirs (torch.Tensor): (Optional) Tensor containing the view directions.
                Its shape is ``(number_of_rays, number_of_points_per_ray, point_dims)``.

        Returns:
            torch.Tensor: Tensor containing the prediction result of the model. Its
            shape is ``(n_samples, n_rays, output_channel)``.
        """
        batch_size = len(query_points)
        x = query_points.reshape(batch_size, -1, query_points.shape[-1])

        if self.is_input_encoding_active:
            x = self.encoding_fn(x)

            if self.is_use_view_directions:
                check_not_none(view_dirs, "view_dirs")
                enc_view_dirs = view_dirs[:, :, None].expand(query_points.shape)
                enc_view_dirs = enc_view_dirs.reshape(
                    batch_size, -1, enc_view_dirs.shape[-1]
                )
                enc_view_dirs = self.encoding_viewdir_fn(enc_view_dirs)
                view_dirs = enc_view_dirs.reshape(
                    batch_size, -1, enc_view_dirs.shape[-1]
                )

        x_shape_buffer = query_points.shape

        # Buffer input tensor
        h = x

        # Iterate all layers in the model
        for i in range(0, self.base_network_depth):
            layer = getattr(self, "linear_" + str(i))
            h = torch.relu(layer(h))
            if i in self.skip_layers:
                h = torch.cat((x, h), -1)

        # Add view direction information, if given
        if self.is_use_view_directions:
            check_not_none(view_dirs, "view_dirs")

            alpha = self.alpha_out(h)
            bottleneck_h = self.bottleneck(h)

            h = torch.cat((bottleneck_h, view_dirs), -1)

            for i in range(self.view_dir_network_depth):
                layer = getattr(self, "vd_linear_" + str(i))
                h = torch.relu(layer(h))

            h = self.prediction(h)
            pred = torch.cat((h, alpha), -1)

        else:
            pred = self.prediction(h)

        pred = pred.reshape(*x_shape_buffer[:3], -1)

        return pred
