import os

from pynif3d.log.log_funcs import func_logger
from pynif3d.models import ConvolutionalOccupancyNetworksModel, PointNet_LocalPool
from pynif3d.pipeline import BasePipeline
from pynif3d.renderer import PointRenderer
from pynif3d.sampling import FeatureSampler2D


class ConvolutionalOccupancyNetworks(BasePipeline):
    """
    This is the main pipeline function for the Convolutional Occupancy Networks:
    https://arxiv.org/abs/2003.04618

    This class takes the noisy point cloud, applies an encoding function (i.e. PointNet)
    to extract features from inputs, projects the points to 2D plane(s) or 3D grid,
    optionally applies an auto-encoder network in 2D or 3D planes to generate features.
    For each input query point, bilinear/trilinear sampling on feature plane(s) or grid
    is applied in order to extract the features query point features. By applying a
    shallow neural implicit function model, the occupancy probability of each input
    query point is predicted.

    This class takes an encoder, feature sampler, NIF model and rendering functions
    during initialization as input, in order to define the pipeline.

    Usage:

    .. code-block:: python

        model = ConvolutionalOccupancyNetworks()
        occupancies = model(input_points, query_points)
    """

    # flake8: noqa: C901
    @func_logger
    def __init__(
        self,
        encoder_fn=None,
        feature_sampler_fn=None,
        nif_model=None,
        rendering_fn=None,
        pretrained=None,
    ):
        """
        Args:
            encoder_fn (instance): The function instance that is called in order to
                encode the input points. Default is `PointNet_LocalPool`.
            feature_sampler_fn (instance): The function instance that is called in
                order to sample the features on a plane or on a grid. The sampler has to
                match the 2D/3D operation mode. Default is `PlaneFeatureSampler`.
            nif_model (instance): The model instance that outputs occupancy information
                given some query points and sampled features. Default is
                `ConvolutionalOccupancyNetworksModel`.
            rendering_fn (instance): The function instance that is called in order to
                render the query points obtained using the `nif_model`. Default is
                `PointRenderer`.
            pretrained (str): (Optional) The pretrained configuration to load model
                weights from. Default is None.
        """
        super().__init__()

        if encoder_fn is None:
            encoder_fn = PointNet_LocalPool()

        if feature_sampler_fn is None:
            feature_sampler_fn = FeatureSampler2D()

        if nif_model is None:
            nif_model = ConvolutionalOccupancyNetworksModel()

        if rendering_fn is None:
            rendering_fn = PointRenderer()

        self.feature_encoder = encoder_fn
        self.feature_sampler = feature_sampler_fn
        self.nif_model = nif_model
        self.rendering_fn = rendering_fn

        if pretrained is not None:
            yaml_file = os.path.join(os.path.dirname(__file__), "yaml", "con.yaml")
            self.load_pretrained_model(
                yaml_file=yaml_file,
                model_name=pretrained,
            )

    def forward(self, input_points, query_points):
        """
        Args:
            input_points (torch.Tensor): Tensor that holds noisy input points. Its shape
                is ``(batch_size, n_points, point_dimension)``.
            query_points (torch.Tensor): Tensor containing the queried occupancy
                locations. Its shape is ``(batch_size, n_points, point_dimension)``.

        Returns:
            torch.Tensor: Tensor containing occupancy probabilities. Its shape is
            ``(batch_size, n_points)``.
        """
        # Generate features
        features = self.feature_encoder(input_points)

        # Sample query location
        query_features = self.feature_sampler(query_points, features)

        # Apply NIF Model
        occupancies = self.rendering_fn(self.nif_model, query_points, query_features)

        # Aggregate
        # Skip

        return occupancies
