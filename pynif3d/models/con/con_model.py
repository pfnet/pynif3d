import torch
import torch.nn as nn

from pynif3d.common.verification import check_callable, check_not_none
from pynif3d.log.log_funcs import func_logger
from pynif3d.models.con.resnet_fc import ResnetBlockFC


class ConvolutionalOccupancyNetworksModel(nn.Module):
    """
    The model for Convolutional Occupancy Networks (CON) as described in:
    https://arxiv.org/abs/2003.04618

    .. note::
        This implementation is based on the original one, which can be found at:
        https://github.com/autonomousvision/convolutional_occupancy_networks

    This class is the neural implicit function (NIF) model of CON. It takes the query
    points as input, along with optional plane or grid features at query locations
    and outputs the occupancy probability of the input point. If `encoding_fn` is
    provided, the input points will be processed with `encoding_fn` before being
    supplied to model.

    Usage:

    .. code-block:: python

        model = ConvolutionalOccupancyNetworksModel()
        occupancies = model(query_points, query_features)
    """

    @func_logger
    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        block_depth=5,
        block_channels=256,
        linear_channels=128,
        is_linear_active=True,
        encoding_fn=None,
    ):
        """
        Args:
            input_channels (int): The input layer's channel size. If `encoding_fn` is
                provided, this value will be overridden by the
                `encoding_fn.get_dimensions()` function. Default is 3.
            output_channels (int): The output channel size. Default is 1.
            block_depth (int): The number of resnet blocks connected sequentially.
                Default is 5.
            block_channels (int): The channel size of each Fully-Connected ResNet
                block. Default is 256.
            linear_channels (int): The channel size for the linear layers that bind
                plane features to Fully-Connected ResNet blocks. This value shall be
                equal to the input feature's channel dimensions. Default is 128.
            is_linear_active (bool): Boolean flag indicating whether linear layers are
                enabled for the plane features. If True, `query_features` shall be
                provided during inference. Default is True.
            encoding_fn: The function instance that is called in order to apply encoding
                to input point coordinates. It has to contain callable `get_dimensions`
                property which returns the resulting dimensions. Default is None.
        """
        super().__init__()

        # Check positional encoding
        if encoding_fn is not None:
            check_not_none(encoding_fn, "encoding_fn")
            check_callable(encoding_fn, "get_dimensions", "encoding_fn")
            input_channels = encoding_fn.get_dimensions()

        self.linear_stem = nn.Linear(input_channels, block_channels)

        for i in range(block_depth):
            init_fc_s_kwargs = {"bias": True}
            if is_linear_active:
                setattr(
                    self, "linear_" + str(i), nn.Linear(linear_channels, block_channels)
                )
            setattr(
                self,
                "block_" + str(i),
                ResnetBlockFC(
                    block_channels,
                    block_channels,
                    block_channels,
                    init_fc_s_kwargs=init_fc_s_kwargs,
                ),
            )

        self.prediction = nn.Linear(block_channels, output_channels)

        # Save parameters
        self.block_depth = block_depth
        self.is_linear_active = is_linear_active
        self.encoding_fn = encoding_fn

    def forward(self, query_points, query_features=None):
        """
        Args:
            query_points (torch.Tensor): The points to provide as input to the network.
                Its shape is ``(batch_size, n_points, input_channels)``.
            query_features (torch.Tensor) : The plane or grid features related to
                `query_points` locations. Its shape is ``(batch_size, n_points,
                linear_channels)``. Optional.

        Returns:
            torch.Tensor: Tensor which holds the occupancy probabilities of query
            locations. Its shape is ``(batch_size, n_points)``.
        """
        if self.encoding_fn is not None:
            query_points = self.encoding_fn(query_points)

        h = torch.relu(self.linear_stem(query_points))

        for i in range(self.block_depth):
            if self.is_linear_active:
                check_not_none(query_features, "query_features")
                layer = getattr(self, "linear_" + str(i))
                h = torch.relu(h + layer(query_features))
            else:
                h = torch.relu(h)

            layer = getattr(self, "block_" + str(i))
            h = layer(h)

        out = self.prediction(torch.relu(h))
        out = out.squeeze(-1)

        return out
