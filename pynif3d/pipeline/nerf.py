import os

import torch

from pynif3d import logger
from pynif3d.aggregation import NeRFAggregator
from pynif3d.camera import CameraRayGenerator
from pynif3d.common.verification import (
    check_equal,
    check_iterable,
    check_pos_int,
    check_shapes_match,
)
from pynif3d.log.log_funcs import func_logger
from pynif3d.models import NeRFModel
from pynif3d.pipeline import BasePipeline
from pynif3d.renderer import PointRenderer
from pynif3d.sampling import (
    AllPixelSampler,
    RandomPixelSampler,
    UniformRaySampler,
    WeightedRaySampler,
)


class NeRF(BasePipeline):
    """
    This is the main pipeline function for the Neural Radiance Fields (NeRF) algorithm:
    https://arxiv.org/abs/2003.08934

    It takes a camera pose as input and returns the rendered pixel values given the
    input pose.

    Usage:

    .. code-block:: python

        image_size = (image_height, image_width)
        focal_length = (focal_x, focal_y)

        model = NeRF(image_size, focal_length)
        pred_dict = model(camera_pose)
    """

    # flake8: noqa: C901
    @func_logger
    def __init__(
        self,
        # Global parameters
        image_size,
        focal_length,
        n_rays_per_image=1024 * 1,
        n_points_per_chunk=1024 * 1,
        # Image sampling parameters
        input_sampler_training=None,
        input_sampler_inference=None,
        background_color=None,
        # Ray sampling parameters
        ray_generator=None,
        ray_samplers=None,
        n_points_per_ray=None,
        level_of_sampling=2,
        near=2,
        far=6,
        # NIF model parameters
        nif_models=None,
        # Renderer parameters
        rendering_fn=None,
        # Aggregation parameters
        aggregation_fn=None,
        pretrained=None,
    ):
        """
        Args:
            image_size (list, tuple): List or tuple containing the spatial image size
                ``(height, width)``. Its shape is ``(2,)``.
            focal_length (list, tuple): List or tuple containing the camera's focal
                length ``(focal_x, focal_y))``. Its shape is ``(2,)``.
            n_rays_per_image (int): The number of ray samples that are extracted from an
                image and processed. Default is `1024`. Optional.
            n_points_per_chunk: The number of sampled points passed to the NIF model at
                once.
            input_sampler_training (instance): (Optional) The pixel sampling function
                used during training. Default is `RandomPixelSampler`.
            input_sampler_inference (instance): (Optional) The pixel sampling function
                used during inference. Default is `AllPixelSampler`.
            ray_generator (instance): (Optional) The function that is called in order to
                generate rays with respect to a given camera pose. Default is
                `CameraRayGenerator`.
            ray_samplers (list, tuple): (Optional) List or tuple of the same length as
                `level_of_sampling` containing the function(s) that define the sampling
                logic for each ray. Default is `UniformRaySampler` for the first level
                and `WeightedRaySampler` for the second level.
            n_points_per_ray (list, tuple): (Optional) List or tuple with a length equal
                to `level_of_sampling` containing the number of points that are sampled
                across each ray. Default is 64 for each level.
            level_of_sampling (list, tuple): (Optional) List or tuple containing the
                levels of fine samples. Default value is 2 to follow coarse/fine pattern
                in the original NeRF paper.
            near (float): (Optional) The boundary value for each sampled ray. Each ray
                will be sampled between [`near`, `far`]. Default is 2.
            far (float): (Optional) The boundary value for each sampled ray. Each ray
                will be sampled between [`near`, `far`]. Default is 6.
            nif_models (list, tuple): (Optional) List or tuple with the length equal to
                `level_of_sampling` containing the models that define the neural
                implicit representation of the 3D scene. Default is `NeRFModel` for each
                level.
            rendering_fn (instance): (Optional) The function that defines the NIF model
                execution logic, in order to obtain the resulting pixel values. Default
                is `PointRenderer`.
            aggregation_fn (instance): (Optional) The function that defines the
                aggregation logic for the predicted 3D point values, in order to obtain
                the final pixel values. Default is `NeRFAggregator`.
            pretrained (str): (Optional) The pretrained configuration to load model
                weights from. Default is None.
        """

        # TODO: implement ndc
        # TODO: implement c2w
        # TODO: implement batch size > 1
        # TODO: implement chunking code into multiple sub-parts
        # TODO: implement non-lindisp case
        # TODO: implement perturbation
        # TODO: Check for GPU mode
        super().__init__()

        # Assign image width and height
        check_iterable(image_size, "image_size")
        check_equal(len(image_size), 2, "image_size", "2")

        image_height, image_width = image_size

        check_pos_int(image_height, "image_height")
        check_pos_int(image_width, "image_width")

        # Input data sampling parameters
        self.input_sampler_training = input_sampler_training
        self.input_sampler_inference = input_sampler_inference
        if input_sampler_training is None:
            self.input_sampler_training = RandomPixelSampler(image_height, image_width)
        if input_sampler_inference is None:
            self.input_sampler_inference = AllPixelSampler(image_height, image_width)

        # Ray sampling parameters
        check_equal(len(focal_length), 2, "len(focal_length)", "2")

        focal_x, focal_y = focal_length

        if type(focal_x) is not float or type(focal_y) is not float:
            msg = "`focal_x` and `focal_y` shall have float type."
            logger.error(msg)
            raise TypeError(msg)

        self.ray_generator = ray_generator
        if ray_generator is None:
            self.ray_generator = CameraRayGenerator(
                image_height, image_width, focal_x, focal_y
            )

        check_pos_int(level_of_sampling, "level_of_sampling")

        # Define ray sampling points per level of sampling
        if n_points_per_ray is None or len(n_points_per_ray) == 0:
            n_points_per_ray = [64] * level_of_sampling
        else:
            # If provided, ensure that sample count is equal to level of sampling
            check_iterable(n_points_per_ray, "n_points_per_ray")
            check_equal(
                len(n_points_per_ray),
                level_of_sampling,
                "n_points_per_ray",
                "level_of_sampling",
            )

        # Define ray sampling functions
        self.ray_samplers = ray_samplers
        if self.ray_samplers is None or len(self.ray_samplers) == 0:
            self.ray_samplers = [UniformRaySampler(near, far, n_points_per_ray[0])]
            for i in range(1, level_of_sampling):
                self.ray_samplers.append(
                    WeightedRaySampler(near, far, n_points_per_ray[i])
                )
        else:
            # If provided, ensure that the count is equal to level of sampling
            check_iterable(ray_samplers, "ray_samplers")
            check_equal(
                ray_samplers, level_of_sampling, "n_ray_samples", "level_of_sampling"
            )

        # Define NIF Model
        if nif_models is None or len(nif_models) == 0:
            init_kwargs = {
                "w_init_fn": torch.nn.init.xavier_uniform_,
                "b_init_fn": torch.nn.init.zeros_,
            }
            nif_models = [
                NeRFModel(init_kwargs=init_kwargs) for _i in range(level_of_sampling)
            ]
        else:
            # If provided, ensure that the count is equal to level of sampling
            check_iterable(self.nif_models, "nif_models")
            check_equal(
                self.nif_models, level_of_sampling, "nif_models", "level_of_sampling"
            )

        for i in range(level_of_sampling):
            setattr(self, "nif_model_" + str(i), nif_models[i])

        # Rendering function
        self.rendering_fn = rendering_fn
        if rendering_fn is None:
            self.rendering_fn = PointRenderer(n_points_per_chunk)

        # Aggregation function
        self.aggregation_fn = aggregation_fn
        if aggregation_fn is None:
            self.aggregation_fn = NeRFAggregator(background_color)

        # Save other parameters
        self.n_rays_per_image = n_rays_per_image
        self.level_of_sampling = level_of_sampling

        if pretrained is not None:
            yaml_file = os.path.join(os.path.dirname(__file__), "yaml", "nerf.yaml")
            self.load_pretrained_model(
                yaml_file=yaml_file,
                model_name=pretrained,
            )

    def forward(self, pose):
        """
        Args:
            pose (torch.Tensor): Tensor containing the camera pose information, used for
                querying. Its shape is ``(3, 4)``.

        Returns:
            dict: Dictionary containing the rendering result (RGB, depth, disparity and
            transparency values for each pixel that is sampled by
            `input_sampler_inference` or `input_sampler_training`).

        """
        # Generate rays
        rays_o, rays_d, view_dirs = self.ray_generator(pose)
        check_shapes_match(rays_o, rays_d, "rays_o", "rays_d")

        # Sample inputs
        if self.training:
            sampled_data = self.input_sampler_training(
                self.n_rays_per_image,
                rays_o=rays_o,
                rays_d=rays_d,
                view_dirs=view_dirs,
            )
        else:
            sampled_data = self.input_sampler_inference(
                rays_o=rays_o, rays_d=rays_d, view_dirs=view_dirs
            )

        sample_coordinates = sampled_data["sample_coordinates"]
        rays_o = sampled_data["rays_o"]
        rays_d = sampled_data["rays_d"]
        view_dirs = sampled_data["view_dirs"]

        input_dict = {
            "rays_d": rays_d,
            "rays_o": rays_o,
        }
        output_dict = {
            "sample_coordinates": sample_coordinates,
            "rgb_map": [],
            "depth_map": [],
            "disparity_map": [],
            "alpha_map": [],
        }

        for sampling_level in range(self.level_of_sampling):
            # Sample points across the rays
            query_points, z_vals = self.ray_samplers[sampling_level](**input_dict)

            # Get the render model
            nif_model = getattr(self, "nif_model_" + str(sampling_level))

            # Render the sampled points
            nif_prediction = self.rendering_fn(nif_model, query_points, view_dirs)

            # Aggregate the rendered values and create the output
            rendered_data = self.aggregation_fn(nif_prediction, z_vals, rays_d)
            rgb_map, depth_map, disparity_map, alpha_map, weights = rendered_data

            input_dict["weights"] = weights
            input_dict["z_vals"] = z_vals

            output_dict["rgb_map"].append(rgb_map)
            output_dict["depth_map"].append(depth_map)
            output_dict["disparity_map"].append(disparity_map)
            output_dict["alpha_map"].append(alpha_map)

        return output_dict
