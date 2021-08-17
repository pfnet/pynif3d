import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean

from pynif3d.common import init_Linear
from pynif3d.common.torch_helper import coordinate2index, normalize_coordinate
from pynif3d.common.verification import (
    check_callable,
    check_in_options,
    check_not_none,
    check_pos_int,
)
from pynif3d.log.log_funcs import func_logger
from pynif3d.models.con.resnet_fc import ResnetBlockFC


class PointNet_LocalPool(nn.Module):
    """
    The point encoder model for Convolutional Occupancy Networks (CON) as described in:
    https://arxiv.org/abs/2003.04618

    .. note::
        This implementation is based on the original one, which can be found at:
        https://github.com/autonomousvision/convolutional_occupancy_networks

    PointNet-based encoder network with ResNet blocks for each point. It takes the input
    points, applies a variation of PointNet and projects each point to defined plane(s).
    It returns the plane features. The number of input points is fixed.

    Usage:

    .. code-block:: python

        plane = "xz"
        plane_resolution = 256
        feature_channels = 32

        model = PointNet_LocalPool(
            feature_grids=[plane],
            feature_grid_resolution=plane_resolution,
            feature_grid_channels=feature_channels,
        )

        features = model(points)
    """

    @func_logger
    def __init__(
        self,
        input_channels=3,
        point_network_depth=5,
        point_feature_channels=128,
        scatter_type="max",
        feature_grids=None,
        feature_processing_fn=None,
        feature_grid_resolution=32,
        feature_grid_channels=128,
        padding=0.1,
        encoding_fn=None,
    ):
        """
        Args:
            input_channels (int): The input layer's channel size. If `encoding_fn` is
                provided, this value will be overridden by
                `encoding_fn.get_dimensions()`. Default is 3.
            point_network_depth (int): The number of resnet blocks that are connected
                sequentially. Default is 5.
            point_feature_channels (int): The channel size of each Fully-Connected
                ResNet blocks. Default is 128.
            scatter_type (str) : The type of the scattering operation. Options are
                ("mean", "max"). Default is "max".
            feature_grids (iterable): The iterable object to define the planes of the
                points to be projected. The options are ("xy", "yz", "xz", "grid").
                "grid" cannot be used in combination with other options. Default is
                ["xz"].
            feature_processing_fn (instance): (Optional) The model that processes point
                features projected to 2D planes. The instance of the pre-initialized
                model has to be provided. If not provided, the 2D plane processing step
                will be skipped.
            feature_grid_resolution (int): The resolution of the 2D planes that the
                points are projected to. It has to be a positive integer. Only square
                planes are supported for now. Default is 32.
            feature_grid_channels (int): The channel size of the 2D plane features. It
                has to be same as the input channel dimensions of the model provided
                through `plane_processing_fn`. Default is 128.
            padding (float): Padding variable used during the normalization operations.
                Assign to 0 to cancel any padding. Default is 0.1.
            encoding_fn (instance): The function instance that applies encoding to input
                point coordinates. It has to contain the callable `get_dimensions`
                property which returns the resulting dimensions. Default is None.
        """
        super().__init__()

        # Define default values for immutable variables
        if feature_grids is None:
            feature_grids = ["xz"]
        scatter_t_list = {"max": scatter_max, "mean": scatter_mean}

        # Check positional encoding
        if encoding_fn is not None:
            check_not_none(encoding_fn, "encoding_fn")
            check_callable(encoding_fn, "get_dimensions", "encoding_fn")
            input_channels = encoding_fn.get_dimensions()

        # Check input parameters
        check_pos_int(input_channels, "input_channels")
        check_pos_int(point_network_depth, "point_network_depth")
        check_pos_int(point_feature_channels, "point_feature_channels")
        check_pos_int(feature_grid_channels, "feature_grid_channels")
        check_pos_int(feature_grid_resolution, "feature_grid_resolution")
        check_in_options(scatter_type, scatter_t_list.keys(), "scatter_type")

        # Save the input parameters
        self.point_network_depth = point_network_depth
        self.feature_grid_channels = feature_grid_channels
        self.feature_grid_resolution = feature_grid_resolution
        self.feature_grids = feature_grids
        self.padding = padding

        # Define point model
        self.fc_stem = init_Linear(input_channels, 2 * point_feature_channels)

        for i in range(point_network_depth):
            init_fc1_kwargs = {"w_init_fn": torch.nn.init.zeros_}
            setattr(
                self,
                "fc_block_" + str(i),
                ResnetBlockFC(
                    2 * point_feature_channels,
                    point_feature_channels,
                    point_feature_channels,
                    init_fc_1_kwargs=init_fc1_kwargs,
                ),
            )

        self.fc_feature_grid = init_Linear(
            point_feature_channels, feature_grid_channels
        )

        # Define coordinate processing function
        self.feature_processing_fn = feature_processing_fn

        # Define scatter type
        self.scatter = scatter_t_list[scatter_type]

        self.encoding_fn = encoding_fn

    def generate_coordinate_features(self, p, c, feature_grid="xz"):
        """
        Scatters the given features (c) based on given coordinates (p) by using the grid
        resolution. This is the orthographic point-to-plane projection function.

        Args:
            p(torch.Tensor): Tensor containing the locations of the points. Its shape is
                ``(batch_size, number_of_points, 3)``.
            c(torch.Tensor): Tensor containing the point features. Its shape is
                ``(batch_size, number_of_points, feature_dimensions)``.

        Returns:
            torch.Tensor: Tensor containing the scattered features of the points. Its
            shape is
            ``(batch_size, feature_dimensions, grid_resolution, grid_resolution)``.
        """
        # acquire indices of features in coordinate
        # normalize to the range of (0, 1)
        xy = normalize_coordinate(p, plane=feature_grid, padding=self.padding)
        index = coordinate2index(xy, self.feature_grid_resolution)

        dim_size = self.feature_grid_resolution ** 2
        if "grid" in self.feature_grids:
            dim_size *= self.feature_grid_resolution

        # scatter features from points
        fea_grid = c.new_zeros(p.shape[0], self.feature_grid_channels, dim_size)

        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(
            p.shape[0],
            self.feature_grid_channels,
            self.feature_grid_resolution,
            self.feature_grid_resolution,
            -1,
        )

        if fea_grid.shape[-1] == 1:
            fea_grid = fea_grid[..., 0]

        # process the coordinate features with UNet
        if self.feature_processing_fn is not None:
            fea_grid = self.feature_processing_fn(fea_grid)

        return fea_grid

    def pool_local(self, keys, indices, features):
        """
        Applies the max pooling operation to the point features, based on the grid
        resolution. After pooling, the points within the same pooling region are to the
        same features.

        Args:
            keys (list): List containing the plane IDs. It is expected that such indices
                exist in ``indices``.
            indices (dict): Dictionary containing `(plane_id, point_indices)` pairs
                mapping each point's 1D index to a 2D plane. `point_indices` is a
                `torch.Tensor` with shape ``(batch_size, 1, point_count)``.
            features (torch.Tensor): Tensor containing the point features. Its shape is
                ``(batch_size, point_count, feature_size)``.

        Returns:
            torch.Tensor: Tensor with shape ``(batch_size, point_count, feature_size)``.
        """
        fea_dim = features.shape[2]

        dim_size = self.feature_grid_resolution ** 2
        if "grid" in self.feature_grids:
            dim_size = dim_size * self.feature_grid_resolution

        res = []
        for key in keys:
            # scatter coordinate features from points
            fea = self.scatter(
                features.permute(0, 2, 1), indices[key], dim_size=dim_size
            )

            if self.scatter == scatter_max:
                fea = fea[0]

            # gather feature back to points
            res.append(fea.gather(dim=2, index=indices[key].expand(-1, fea_dim, -1)))

        res = torch.sum(torch.stack(res), 0).permute((0, 2, 1))
        return res

    def forward(self, input_points):
        """
        Args:
            input_points (torch.Tensor): Tensor containing the points that are provided
                as input to the network. Its shape is ``(batch_size, n_points,
                input_channels)``.

        Returns:
            dict: Dictionary containing the tensors for all the features of the planes.
        """
        point_coordinates = {}
        point_indices = {}

        for grid_id in self.feature_grids:
            point_coordinates[grid_id] = normalize_coordinate(
                input_points.clone(), plane=grid_id, padding=self.padding
            )
            point_indices[grid_id] = coordinate2index(
                point_coordinates[grid_id], self.feature_grid_resolution
            )

        h = input_points
        if self.encoding_fn is not None:
            h = self.encoding_fn(input_points)

        h = self.fc_stem(h)

        # Run ResNetFC blocks
        h = self.fc_block_0(h)
        for block in range(1, self.point_network_depth):
            layer = getattr(self, "fc_block_" + str(block))
            pooled = self.pool_local(point_coordinates.keys(), point_indices, h)
            h = torch.cat([h, pooled], dim=2)
            h = layer(h)

        c = self.fc_feature_grid(h)

        features = {}
        for grid_id in self.feature_grids:
            features[grid_id] = self.generate_coordinate_features(
                input_points, c, feature_grid=grid_id
            )

        return features
