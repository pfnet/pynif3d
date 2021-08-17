import torch

from pynif3d.common.verification import check_callable, check_not_none, check_true
from pynif3d.encoding import PositionalEncoding
from pynif3d.log.log_funcs import func_logger
from pynif3d.models.nerf_model import NeRFModel


class IDRRenderingModel(NeRFModel):
    """
    The multi-layer MLP model for IDR rendering. If provided, it applies positional
    encoding to the view directions and overrides the input channel information
    accordingly.

    .. note::

        Please check the paper for more information: https://arxiv.org/abs/2003.09852

    Usage:

    .. code-block:: python

        model = IDRRenderingModel()
        rgb_values = model(points, features, normals, view_dirs)
    """

    @func_logger
    def __init__(
        self,
        input_channel_points=3,
        input_channel_view_dirs=3,
        input_channel_normals=3,
        input_channel_features=256,
        output_channels=3,
        base_network_depth=4,
        base_network_channels=512,
        is_use_view_directions=True,
        is_use_normals=True,
        encoding_viewdir_fn=None,
        is_input_encoding_active=True,
    ):
        """
        Args:
            input_channels (int): The input channel dimension to the model. If
                positional encoding is used, this value will be overridden. Default is
                3 (XYZ).
            input_channel_view_dirs (int): The input channel dimension for viewing
                directions. If positional encoding is used, this value will be
                overridden. Default is 3 (XYZ).
            input_channel_normals (int): The input channel dimension for surface
                normals. Default is 3 (XYZ).
            input_channel_features (int): The input channel dimension for the extracted
                features. Default value is 256.
            output_channels (int): The output channel dimension. Default is 4 (RGBA).
            base_network_depth (int): The depth of the network MLP layers. One linear
                layer will be added to the base network for each increment. Default is
                4.
            base_network_channels (int): The output dimension of each inner linear
                layers of the MLP model. A positive integer value is expected. Default
                is 512.
            is_use_view_directions (bool): Boolean flag indicating whether to use view
                direction (True) or not (False). If True, the view direction block will
                be added on top of the base MLP layers. Default is True.
            is_use_normals (bool): Boolean flag indicating whether to use surface
                normals (True) or not (False). Default is True.
            encoding_viewdir_fn: The function that is called in order to apply encoding
                to the view directions input. Default is `PositionalEncoding()`.
            is_input_encoding_active (bool): Boolean flag indicating whether encoding
                shall be applied to both the base network input and view directions.
                Default is True.
        """
        if is_input_encoding_active:
            if encoding_viewdir_fn is None:
                encoding_viewdir_fn = PositionalEncoding(
                    input_dimensions=input_channel_view_dirs,
                    num_frequency=4,
                    max_frequency=3,
                )
            check_callable(encoding_viewdir_fn, "get_dimensions", "encoding_viewdir_fn")
            input_channel_view_dirs = encoding_viewdir_fn.get_dimensions()

        input_channels = input_channel_points + input_channel_features
        if is_use_view_directions:
            input_channels += input_channel_view_dirs
        if is_use_normals:
            input_channels += input_channel_normals

        skip_layers = []

        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            base_network_depth=base_network_depth,
            base_network_channels=base_network_channels,
            is_input_encoding_active=False,
            is_use_view_directions=False,
            normalize_weights=True,
            skip_layers=skip_layers,
        )

        self.is_use_view_directions = is_use_view_directions
        self.is_use_normals = is_use_normals
        self.encoding_viewdir_fn = encoding_viewdir_fn
        self.is_input_encoding_active = is_input_encoding_active

    def forward(self, points, features, normals=None, view_dirs=None):
        """
        Args:
            points (torch.Tensor): Tensor containing the points that are processed. Its
                shape is ``(batch_size, n_rays, input_channel_points)`` or
                ``(n_rays, input_channel_points)``.
            features (torch.Tensor): Tensor containing the features that are processed.
                Its shape is ``(batch_size, n_rays, input_channel_features)`` or
                ``(n_rays, input_channel_features)``.
            normals (torch.Tensor): (Optional) Tensor containing the normals that are
                processed. Its shape is ``(batch_size, n_rays, input_channel_normals)``
                or ``(n_rays, input_channel_normals)``.
            view_dirs (torch.Tensor): (Optional) Tensor containing the view directions
                that are processed. Its shape is
                ``(batch_size, n_rays, input_channel_view_dirs)`` or
                ``(n_rays, input_channel_view_dirs)``.

        Returns:
            torch.Tensor: Tensor containing the rendered RGB values. Its shape is
            ``(*points.shape[:-1], 3)``.
        """
        valid_shape = (points.ndim == 2 or points.ndim == 3) and (points.shape[-1] == 3)
        check_true(valid_shape, "valid_shape")

        x = points.reshape(-1, 3)

        if self.is_use_view_directions:
            check_not_none(view_dirs, "view_dirs")
            v = view_dirs.reshape(-1, 3)
            if self.is_input_encoding_active:
                v = self.encoding_viewdir_fn(v)
            x = torch.cat([x, v], dim=-1)

        if self.is_use_normals:
            check_not_none(normals, "normals")
            n = normals.reshape(-1, 3)
            x = torch.cat([x, n], dim=-1)

        f = features.reshape(*x.shape[:-1], -1)
        x = torch.cat([x, f], dim=-1)
        h = x

        for layer_index in range(0, self.base_network_depth):
            layer = getattr(self, "linear_" + str(layer_index))
            h = torch.relu(layer(h))

        pred = self.prediction(h)
        pred = pred.reshape(*points.shape[:-1], -1)

        return pred
