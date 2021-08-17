import torch

from pynif3d.common.verification import check_not_none
from pynif3d.log.log_funcs import func_logger
from pynif3d.sampling.ray.uniform import UniformRaySampler
from pynif3d.utils.functional import secant


class SecantRaySampler(torch.nn.Module):
    """
    Samples the ray in a given range, computes the SDF values for the sampled points
    and runs the secant method for the rays which have sign transition. Returns the
    resulting points and their corresponding SDF values.

    :: note:

        For more information about the secant method, please check the following page:
        https://en.wikipedia.org/wiki/Secant_method

    Usage:

    .. code-block:: python

        # Assume an SDF model (torch.nn.Module) is given.
        sampler = SecantRaySampler(sdf_model)
        points, z_vals, mask = sampler(
            ray_directions, ray_origins, ray_mask, zs_min, zs_max
        )
    """

    @func_logger
    def __init__(self, sdf_model):
        """
        Args:
            sdf_model (instance): Instance of an SDF model. When calling the `forward`
                method with some input points, it needs to return a dictionary
                containing the SDF values corresponding to those points, as an
                "sdf_vals" key/value pair.
        """
        super().__init__()
        check_not_none(sdf_model, "sdf_model")
        self.sdf_model = sdf_model

    def forward(self, rays_d, rays_o, rays_m, zs_min, zs_max, **kwargs):
        """
        Args:
            rays_d (torch.Tensor): Tensor containing the ray directions. Its shape is
                ``(n_rays, 3)``.
            rays_o (torch.Tensor): Tensor containing the ray origins. Its shape is
                ``(n_rays, 3)``.
            rays_m (torch.Tensor): Boolean tensor containing the object mask for
                the given rays. If rays_d[i] intersects the object, rays_m[i] is
                marked as True, otherwise as False. Its shape is ``(n_rays,)``.
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
            tuple: Tuple containing the secant points (as a torch.Tensor with shape
            ``(n_rays, 3)``), corresponding Z values (as a torch.Tensor with shape
            ``(n_rays,)``) and a mask (as a torch.Tensor with shape ``(n_rays,)``)
            specifying which points were successfully found by the secant method
            to be roots of the optimization function.
        """
        n_samples = kwargs.get("n_samples", 100)
        chunk_size = kwargs.get("chunk_size", 10000)

        # Sample the rays uniformly between the `[zs_min, zs_max]` interval.
        ray_sampler = UniformRaySampler(zs_min, zs_max, n_samples, is_perturb=False)
        points, z_vals = ray_sampler(rays_d, rays_o)
        points = points.reshape(-1, n_samples, 3)
        z_vals = z_vals.reshape(-1, n_samples)

        # Compute the SDF values corresponding to the sampled points.
        chunks = torch.split(points.reshape(-1, 3), chunk_size, dim=0)
        sdf_vals = torch.cat([self.sdf_model(p) for p in chunks]).reshape(-1, n_samples)

        # Get the index of the first negative SDF value (if any) encountered along the
        # ray or of the last positive one if there are no negative SDF values.
        ind_min_sdf = torch.arange(n_samples, 0, -1, device=rays_d.device)
        ind_min_sdf = torch.argmin(torch.sign(sdf_vals) * ind_min_sdf, dim=-1)
        secant_points = points[torch.arange(len(points)), ind_min_sdf, :]
        secant_z_vals = z_vals[torch.arange(len(points)), ind_min_sdf]

        # Handle the outliers. The points that are located inside the surface should
        # have a negative SDF value.
        network_mask = sdf_vals[torch.arange(len(sdf_vals)), ind_min_sdf] < 0
        inlier_mask = torch.logical_and(rays_m.reshape(-1), network_mask)
        outlier_mask = torch.logical_not(inlier_mask)

        if torch.any(outlier_mask):
            indices_outliers = torch.argmin(sdf_vals[outlier_mask, :], dim=-1)
            secant_points[outlier_mask] = points[outlier_mask, indices_outliers, :]
            secant_z_vals[outlier_mask] = z_vals[outlier_mask, indices_outliers]

        # Determine the target points and run the secant method.
        secant_mask = torch.ones(len(rays_d), dtype=bool, device=rays_d.device)
        secant_mask[torch.logical_not(network_mask)] = False

        if not self.training:
            inlier_mask = network_mask

        # Run the secant method on the chosen interval range.
        if torch.any(inlier_mask):
            zs_max = z_vals[torch.arange(len(z_vals)), ind_min_sdf][inlier_mask]
            zs_min = z_vals[torch.arange(len(z_vals)), ind_min_sdf - 1][inlier_mask]

            rays_o_secant = rays_o.reshape(-1, 3)[inlier_mask]
            rays_d_secant = rays_d.reshape(-1, 3)[inlier_mask]

            # Define the function that optimization is run for.
            def secant_fn(zs):
                _points = rays_o_secant + zs[..., None] * rays_d_secant
                _result = self.sdf_model(_points)
                return _result

            z_pred_secant = secant(secant_fn, zs_min, zs_max)

            secant_points[inlier_mask] = (
                z_pred_secant[..., None] * rays_d_secant + rays_o_secant
            )
            secant_z_vals[inlier_mask] = z_pred_secant

        return secant_points, secant_z_vals, secant_mask
