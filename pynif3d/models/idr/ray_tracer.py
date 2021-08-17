import torch

from pynif3d.camera import SphereRayTracer
from pynif3d.common.verification import check_equal, check_not_none, check_shapes_match
from pynif3d.log.log_funcs import func_logger
from pynif3d.sampling.ray.secant import SecantRaySampler
from pynif3d.utils.functional import ray_sphere_intersection


class IDRRayTracer(torch.nn.Module):
    """
    The IDR ray tracer. Takes ray directions, ray origins + 2D object mask as input and
    computes the intersection points with the surface. Also handles the non-convergent
    rays.

    Usage:

    .. code-block:: python

        # Assume that an SDF model (torch.nn.Module) is given.
        model = IDRRayTracer(sdf_model)
        points, z_vals, mask_intersect = model(ray_directions, ray_origins, object_mask)
    """

    @func_logger
    def __init__(self, sdf_model, **kwargs):
        """
        Args:
            sdf_model (instance): Instance of an SDF model.
        """
        super().__init__()
        check_not_none(sdf_model, "sdf_model")
        self.sdf_model = sdf_model

        self.ray_tracer = SphereRayTracer(sdf_model)
        self.ray_sampler = SecantRaySampler(sdf_model)

    def forward(self, ray_directions, ray_origins, object_mask):
        """
        Args:
            ray_directions (torch.Tensor): Tensor containing the ray directions. Its
                shape is ``(n_rays, 3)``.
            ray_origins (torch.Tensor): Tensor containing the ray origins. Its shape is
                ``(n_rays, 3)``.
            object_mask (torch.Tensor): Boolean tensor containing the object mask for
                the given rays. If rays_d[i] intersects the object, object_mask[i] is
                marked as True, otherwise as False. Its shape is ``(n_rays,)``.

        Returns:
            tuple: Tuple containing the sampled points, their corresponding Z values and
                point-to-surface-intersection mask.
        """
        check_shapes_match(ray_directions, ray_origins, "ray_origins", "ray_directions")
        check_equal(ray_origins.shape[-1], 3, "ray_origins.shape[-1]", "3")

        rays_d = ray_directions.reshape(-1, 3)
        rays_o = ray_origins.reshape(-1, 3)
        rays_m = object_mask.reshape(-1)

        check_equal(len(rays_m), len(rays_d), "len(rays_m)", "len(rays_d)")

        # Prepare the output data.
        points_all = torch.zeros_like(rays_d, dtype=torch.float32)
        z_vals_all = torch.zeros_like(rays_m, dtype=torch.float32)
        network_mask_all = torch.zeros_like(rays_m)
        mask_unfinished_all = torch.zeros_like(rays_m)

        # Determine which rays will intersect the surface.
        z_vals_intersect, mask_intersect = ray_sphere_intersection(rays_d, rays_o)
        z_vals_intersect[z_vals_intersect < 0] = 0

        if not torch.any(mask_intersect):
            return points_all, z_vals_all, network_mask_all

        # Compute the intersection between the current ray directions and the spheres
        # centered at the origins of the rays. The intersection between a ray and a
        # sphere consists of two intersection points (stored in the first channel).
        points_tracer, z_vals_tracer, mask_unfinished = self.ray_tracer(
            rays_d=rays_d[mask_intersect],
            rays_o=rays_o[mask_intersect],
            z_vals=z_vals_intersect[:, mask_intersect],
        )
        zs_min = z_vals_tracer[0]
        zs_max = z_vals_tracer[1]
        network_mask = zs_min < zs_max

        # Discard the second intersection points.
        points_tracer = points_tracer[0]
        z_vals_tracer = z_vals_tracer[0]
        mask_unfinished = mask_unfinished[0]

        # Handle the non-convergent rays.
        if torch.any(mask_unfinished):
            secant_points, secant_z_vals, secant_mask = self.ray_sampler(
                rays_d=rays_d[mask_intersect][mask_unfinished],
                rays_o=rays_o[mask_intersect][mask_unfinished],
                rays_m=rays_m[mask_intersect][mask_unfinished],
                zs_min=zs_min[mask_unfinished],
                zs_max=zs_max[mask_unfinished],
            )
            points_tracer[mask_unfinished] = secant_points
            z_vals_tracer[mask_unfinished] = secant_z_vals
            network_mask[mask_unfinished] = secant_mask

        # Update the output data. So far all the computations have been done using the
        # points that intersect the spheres.
        points_all[mask_intersect] = points_tracer
        z_vals_all[mask_intersect] = z_vals_tracer
        mask_unfinished_all[mask_intersect] = mask_unfinished
        network_mask_all[mask_intersect] = network_mask

        if not self.training:
            return points_all, z_vals_all, network_mask_all

        # Handle the outliers.
        mask_i = ~network_mask_all & rays_m & ~mask_unfinished_all
        mask_o = ~rays_m & ~mask_unfinished_all
        mask = (mask_i | mask_o) & ~mask_intersect

        if torch.any(mask):
            rays_o_left_out = rays_o[mask]
            rays_d_left_out = rays_d[mask]
            z_vals_left_out = -torch.sum(rays_d_left_out * rays_o_left_out, dim=-1)

            z_vals_all[mask] = z_vals_left_out
            points_all[mask] = (
                z_vals_left_out[..., None] * rays_d_left_out + rays_o_left_out
            )

        mask = (mask_i | mask_o) & mask_intersect

        if torch.any(mask):
            zs_min, zs_max = z_vals_intersect
            zs_min[network_mask_all & mask_o] = z_vals_all[network_mask_all & mask_o]
            points_min_sdf, z_vals_min_sdf = self.sample_min_sdf_uniform(
                rays_d=rays_d[mask],
                rays_o=rays_o[mask],
                zs_min=zs_min[mask],
                zs_max=zs_max[mask],
            )
            points_all[mask] = points_min_sdf
            z_vals_all[mask] = z_vals_min_sdf

        return points_all, z_vals_all, network_mask_all

    def sample_min_sdf_uniform(self, rays_d, rays_o, zs_min, zs_max, **kwargs):
        """
        Uniformly samples points along the ray, in the [zs_min, zs_max] interval, and
        selects the ones with the minimum SDF values.
        Args:
            rays_d (torch.Tensor): Tensor containing the ray directions. Its shape is
                ``(n_rays, 3)``.
            rays_o (torch.Tensor): Tensor containing the ray origins. Its shape is
                ``(n_rays, 3)``.
            zs_min (torch.Tensor): Tensor containing the minimum Z values of the points
                that are sampled along the ray. Its shape is ``(n_rays,)``.
            zs_max (torch.Tensor): Tensor containing the maximum Z values of the points
                that are sampled along the ray. Its shape is ``(n_rays,)``.
            kwargs (dict):
                - **n_samples** (int): The number of points that are sampled along the
                  rays. Default value is 100.
                - **chunk_size** (int): The size of the chunk of points that is passed
                  to the SDF model. Default value is 10000.
        Returns:
            tuple: Tuple containing the samples points (as a torch.Tensor with shape
            ``(n_rays, 3)``) and corresponding Z values (as a torch.Tensor with
            shape ``(n_rays,)``).
        """
        n_samples = kwargs.get("n_samples", 100)
        chunk_size = kwargs.get("chunk_size", 10000)

        z_vals = torch.rand(n_samples, device=rays_d.device)

        z_vals = z_vals[None, ...] * (zs_max - zs_min)[..., None] + zs_min[..., None]
        points = z_vals[..., None] * rays_d[..., None, :] + rays_o[..., None, :]

        chunks = torch.split(points.reshape(-1, 3), chunk_size, dim=0)
        sdf_vals = torch.cat([self.sdf_model(p) for p in chunks]).reshape(-1, n_samples)
        indices = torch.argmin(sdf_vals, dim=-1)

        points = points[torch.arange(len(rays_d)), indices]
        z_vals = z_vals[torch.arange(len(rays_d)), indices]

        return points, z_vals
