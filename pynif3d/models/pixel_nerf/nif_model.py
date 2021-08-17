import torch

from pynif3d import logger
from pynif3d.common.layer_generator import init_Linear
from pynif3d.common.torch_helper import repeat_interleave
from pynif3d.common.verification import (
    check_callable,
    check_in_options,
    check_not_none,
    check_pos_int,
)
from pynif3d.encoding import PositionalEncoding
from pynif3d.log.log_funcs import func_logger
from pynif3d.models import ResnetBlockFC


class PixelNeRFNIFModel(torch.nn.Module):
    """
    The multi-layer MLP model for PixelNeRF rendering. If provided, it applies
    positional encoding to the inputs overrides the input channel information
    accordingly. It can also integrate view direction information into the network.
    """

    @func_logger
    def __init__(
        self,
        input_channel_points: int = 3,
        input_channel_view_dirs: int = 3,
        output_channels: int = 4,
        hidden_channels: int = 128,
        is_use_view_directions: bool = True,
        is_point_encoding_active: bool = True,
        is_view_encoding_active: bool = False,
        encoding_fn: torch.nn.Module = None,
        encoding_viewdir_fn: torch.nn.Module = None,
        n_resnet_blocks: int = 5,
        reduce_block_index: int = 3,
        activation_fn: torch.nn.Module = None,
        init_kwargs: dict = None,
    ) -> None:
        """
        Args:
            input_channel_points (int): The input channel dimension to the model. If
                positional encoding is used, this value will be overridden. Default is
                3 (XYZ).
            input_channel_view_dirs (int): The input channel dimension for viewing
                directions. If positional encoding is used, this value will be
                overridden. Default is 3 (XYZ).
            output_channels (int): The output channel dimension. Default is 4 (RGBA).
            hidden_channels (int): The number of hidden channels contained within each
                ResNetBlockFC block. Default is 128.
            is_use_view_directions (bool): Boolean flag indicating whether to use view
                direction (True) or not (False). If True, the view direction block will
                be added on top of the base MLP layers. Default is True.
            is_point_encoding_active (bool): Boolean flag indicating whether encoding
                shall be applied to the input points. Default is True.
            is_view_encoding_active (bool): Boolean flag indicating whether encoding
                shall be applied to the viewing directions. Default is False.
            encoding_fn (torch.nn.Module): The function that is called in order to apply
                encoding to the NIF model input. Default is `PositionalEncoding()`.
            encoding_viewdir_fn (torch.nn.Module): The function that is called in order
                to apply encoding to the view directions input. Default is
                `PositionalEncoding()`.
            n_resnet_blocks (int): The number of ResNetBlockFC blocks that are contained
                within the base network. Default value is 5.
            reduce_block_index (int): The index of the ResNetBlockFC block at which
                the reduce operation is going to be applied (along the dimension that
                is related to the number of objects). Default value is 3.
            activation_fn (torch.nn.Module): The activation function. Default is ReLU.
            init_kwargs (dict): Dictionary containing the initialization parameters for
                the linear layers.
        """
        super().__init__()

        check_pos_int(input_channel_points, "input_channel_points")
        self.input_channel_points = input_channel_points
        check_pos_int(input_channel_view_dirs, "input_channel_view_dirs")
        self.input_channels_view_dirs = input_channel_view_dirs
        check_pos_int(output_channels, "output_channels")
        self.output_channels = output_channels
        check_pos_int(hidden_channels, "hidden_channels")
        self.hidden_channels = hidden_channels
        check_pos_int(n_resnet_blocks, "n_resnet_blocks")
        self.n_resnet_blocks = n_resnet_blocks
        check_pos_int(reduce_block_index, "reduce_block_index")
        self.reduce_block_index = reduce_block_index

        self.is_use_view_directions = is_use_view_directions
        self.is_point_encoding_active = is_point_encoding_active
        self.is_view_encoding_active = is_view_encoding_active

        if is_point_encoding_active or is_view_encoding_active:
            logger.info("Input encoding is active. Overriding input channel size.")

        if is_point_encoding_active:
            if encoding_fn is None:
                encoding_fn = PositionalEncoding(
                    input_dimensions=input_channel_points,
                    num_frequency=6,
                    max_frequency=5,
                    is_log_sampling=True,
                    frequency_factor=1.5,
                )
            check_callable(encoding_fn, "get_dimensions", "encoding_fn")
            self.encoding_fn = encoding_fn
            input_channel_points = self.encoding_fn.get_dimensions()

        input_channels = input_channel_points

        if is_use_view_directions:
            if is_view_encoding_active:
                if encoding_viewdir_fn is None:
                    encoding_viewdir_fn = PositionalEncoding(
                        input_dimensions=input_channel_view_dirs,
                        num_frequency=6,
                        max_frequency=5,
                        is_log_sampling=True,
                        frequency_factor=1.5,
                    )
                check_callable(
                    encoding_viewdir_fn, "get_dimensions", "encoding_viewdir_fn"
                )
                self.encoding_viewdir_fn = encoding_viewdir_fn
                input_channel_view_dirs = self.encoding_viewdir_fn.get_dimensions()

            input_channels += input_channel_view_dirs

        if init_kwargs is None:
            init_kwargs = {
                "w_init_fn": torch.nn.init.kaiming_normal_,
                "b_init_fn": torch.nn.init.zeros_,
            }

        self.input_layer = init_Linear(input_channels, hidden_channels, **init_kwargs)
        self.resnet_blocks = torch.nn.ModuleList(
            [
                ResnetBlockFC(hidden_channels, hidden_channels, hidden_channels)
                for _ in range(n_resnet_blocks)
            ]
        )
        self.prediction = init_Linear(hidden_channels, output_channels, **init_kwargs)

        n_linear_layers = min(reduce_block_index, n_resnet_blocks)
        self.linear_layers = torch.nn.ModuleList(
            [
                init_Linear(hidden_channels, hidden_channels, **init_kwargs)
                for _ in range(n_linear_layers)
            ]
        )

        if activation_fn is None:
            activation_fn = torch.nn.ReLU()
        self.activation_fn = activation_fn

    def forward(
        self,
        ray_points: torch.Tensor,
        camera_poses: torch.Tensor,
        features: torch.Tensor,
        view_dirs: torch.Tensor = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Args:
            ray_points (torch.Tensor): Tensor containing the ray points to that are
                going to be processed. Its shape is
                ``(n_ray_samples, input_channel_points)``.
            camera_poses (torch.Tensor): Tensor containing the camera poses. Its shape
                is ``(n_objects, 3, 4)``.
            features (torch.Tensor): (Optional) Tensor containing the feature vectors.
                Its shape is ``(n_views, n_ray_samples, feature_size)``.
            view_dirs (torch.Tensor): (Optional) Tensor containing the view directions.
                Its shape is ``(n_ray_samples, input_channel_points)``.
            kwargs (dict):
                - **reduce** (str): The reduce operation that is applied to the
                  ResNetBlockFC block at `reduce_block_index`. Currently supported
                  options are "average" and "max".
        Returns:
            torch.Tensor: Tensor containing the prediction result of the model. Its
                shape is ``(n_ray_samples, output_channel)``.
        """
        reduce = kwargs.get("reduce_type", "average")

        choices = ["average", "max"]
        check_in_options(reduce, choices, "reduce")

        n_views, n_samples = features.shape[:2]
        x = repeat_interleave(ray_points, n_views)
        x = camera_poses[..., None, :3, :3] @ ray_points[..., None]
        x = x.squeeze().reshape(-1, 3)
        features = features.reshape(-1, self.hidden_channels)

        if self.is_point_encoding_active:
            x = self.encoding_fn(x)

        if self.is_use_view_directions:
            check_not_none(view_dirs, "view_dirs")
            v = repeat_interleave(view_dirs, n_views)
            v = camera_poses[..., None, :3, :3] @ view_dirs[..., None]
            v = v.reshape(-1, 3)
            x = torch.cat((x, v), dim=-1)

            if self.is_view_encoding_active:
                x = self.encoding_viewdir_fn(x)

        h = self.input_layer(x)

        for index in range(self.n_resnet_blocks):
            if index < self.reduce_block_index:
                dh = self.linear_layers[index](features)
                h = h + dh
            elif index == self.reduce_block_index:
                h = h.reshape(-1, n_views, n_samples, self.hidden_channels)
                if reduce == "average":
                    h = torch.mean(h, dim=1)
                else:
                    h = torch.max(h, dim=1)[0]

            h = self.resnet_blocks[index](h)

        prediction = self.prediction(self.activation_fn(h))
        prediction = prediction.reshape(*ray_points.shape[:-1], prediction.shape[-1])
        return prediction
