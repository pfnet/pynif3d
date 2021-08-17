import torch
import torch.nn as nn

from pynif3d.log.log_funcs import func_logger


class NeRFAggregator(nn.Module):
    """
    Color aggregation function. Takes the raw predictions obtained from a NIF model, the
    depth values and a ray direction to produce the final color of a pixel. Please refer
    to the original NeRF paper for more information.

    Usage:

    .. code-block:: python

        # Assume the models are given.
        nif_model = NeRFModel()
        renderer = PointRenderer()
        aggregator = NeRFAggregator()

        # Get the RGBA values for the corresponding points and view directions.
        rgba_values = renderer(nif_model, query_points, view_directions)

        # Aggregate the computed RGBA values to form the RGB maps, depth maps etc.
        rendered_data = aggregator(rgba_values, ray_z_values, ray_directions)
        rgb_map, depth_map, disparity_map, alpha_map, weight = rendered_data
    """

    @func_logger
    def __init__(self, background_color=None, noise_std=0.0):
        """
        Args:
            background_color (torch.Tensor): The background color to be added for
                rendering. If set to None, no background will be added. Default is None.

            noise_std (float): The standard deviation of the noise to be added to alpha.
                Set 0 to disable the noise addition. Default is 0.
        """

        super().__init__()

        if background_color is None:
            background_color = torch.full((3,), -1, dtype=torch.float32)

        self.register_buffer("bg_color", background_color)
        self.noise_std = noise_std

    def forward(self, raw_model_prediction, z_vals, rays_d):
        """
        Args:
            raw_model_prediction (torch.Tensor): The prediction output of a NIF model.
                Its shape is ``(number_of_rays, number_of_points_per_ray, 4)``.
            z_vals (torch.Tensor): Depth values of the rays. Its shape is
                ``(number_of_rays, number_of_points_per_ray, 4)``.
            rays_d (torch.Tensor): The direction vector of the rays. Its shape is
                ``(number_of_rays, 3)``.

        Returns:
            tuple: Tuple containing:
                - **rgb_map** (torch.Tensor): The RGB pixel values of the rays. Its
                  shape is ``(number_of_rays, 3)``.
                - **depth_map** (torch.Tensor): The depth pixel values of the rays. Its
                  shape is ``(number_of_rays,)``.
                - **disparity_map** (torch.Tensor): The inverse depth pixel values of
                  the rays. Its shape is ``(number_of_rays,)``.
                - **alpha_map** (torch.Tensor): The transparency/alpha value of each
                  pixel. Its shape is ``(number_of_rays,)``.
                - **weights** (torch.Tensor): The weights of each sampled points across
                  a ray which impact the final RGB/depth/disparity/transparency value of
                  a pixel. Its shape is ``(number_of_rays, number_of_points_per_ray)``.


        """
        # Calculate the distance of each sampled point across a ray
        p_distances = z_vals[..., 1:] - z_vals[..., :-1]

        last_elem = torch.as_tensor([1e10])[None, :].expand(p_distances[..., :1].shape)
        last_elem = last_elem.to(p_distances.device)
        p_distances = torch.cat([p_distances, last_elem], dim=-1)

        p_distances = p_distances * torch.norm(rays_d[..., None, :], dim=-1)

        # Generate alpha and color channels
        rgb = torch.sigmoid(raw_model_prediction[..., :3])

        # Add noise to alpha
        noise = torch.rand_like(raw_model_prediction[..., 3]) * self.noise_std
        alpha = 1.0 - torch.exp(
            -torch.relu(raw_model_prediction[..., 3] + noise) * p_distances
        )

        # Compute the exclusive cumulative product based on alpha
        weights = torch.nn.functional.pad(
            1.0 - alpha[:, :, :-1] + 1e-10, pad=[1, 0], value=1
        )
        weights = torch.cumprod(weights, dim=-1)

        # Generate the weight of each point based on alpha
        weights = alpha * weights

        # Generate rgb map
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

        # Generate alpha map
        alpha_map = torch.sum(weights, -1)

        # Generate depth map
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Generate disparity map
        disparity_map = 1.0 / (torch.relu(depth_map / alpha_map - 1e-10) + 1e-10)

        is_bg_color = torch.logical_and(self.bg_color >= 0, self.bg_color <= 1).all()
        if is_bg_color:
            rgb_map = rgb_map + (1.0 - alpha_map[..., None]) * self.bg_color

        return rgb_map, depth_map, disparity_map, alpha_map, weights
