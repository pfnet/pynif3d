import torch

from pynif3d.common.verification import check_equal, check_shapes_match


class SphereRayTracer(torch.nn.Module):
    """
    Determines the intersection between a set of rays and a surface defined by an
    implicit representation using the sphere tracing algorithm.

    Usage:

    .. code-block:: python

        # Assume an SDF model (torch.nn.Module) is given.
        ray_tracer = SphereRayTracer(sdf_model)
        points, z_vals, mask_not_converged = ray_tracer(ray_directions, ray_origins)
    """

    def __init__(self, sdf_model, **kwargs):
        """
        Args:
            sdf_model (instance): Instance of an SDF model.
            kwargs (dict):
                - **sdf_threshold** (float): The SDF threshold that is used to determine
                  whether points are close enough to the surface or not. The closer they
                  are, the lower the SDF value becomes.
                - **n_iterations** (int): The number of iterations that the sphere
                  tracing algorithm is run for.
                - **n_fix_iterations** (int): The number of iterations that the
                  algorithm for correcting overshooting SDF values is run for.
        """
        super().__init__()

        self.sdf_model = sdf_model
        self.sdf_threshold = kwargs.get("sdf_threshold", 5e-5)
        self.n_iterations = kwargs.get("n_iterations", 10)
        self.n_fix_iterations = kwargs.get("n_fix_iterations", 10)

    def fix_overshoot(self, points, z_vals, rays_d, rays_o, sdf_vals, next_sdf_vals):
        mask = next_sdf_vals < 0
        sign = torch.ones_like(next_sdf_vals)
        sign[1, ...] = -1

        for i in range(self.n_fix_iterations):
            if not mask.any():
                break

            z_vals[mask] = z_vals[mask] - 0.5 / (2 ** i) * sign[mask] * sdf_vals[mask]
            points[mask] = z_vals[mask][..., None] * rays_d[mask] + rays_o[mask]
            next_sdf_vals[mask] = self.sdf_model(points[mask])
            mask = next_sdf_vals < 0

        return points, z_vals, next_sdf_vals

    def forward(self, rays_d, rays_o, z_vals=None):
        """
        Args:
            rays_d (torch.Tensor): Tensor containing the ray directions. Its shape is
                ``(batch_size, n_rays, 3)`` or ``(n_rays, 3)``.
            rays_o (torch.Tensor): Tensor containing the ray origins. Its shape is
                ``(batch_size, n_rays, 3)`` or ``(n_rays, 3)``.
            z_vals (torch.Tensor): Tensor containing the initial Z values. Its shape is
                ``(batch_size, n_rays)`` or ``(n_rays,)``.

        Returns:
            tuple: Tuple containing the intersection points (as a torch.Tensor with
                shape ``(2, n_rays, 3)``), the Z values along the ray (as a torch.Tensor
                with shape ``(2, n_rays,)``) and a mask specifying which points the
                algorithm has not converged for (as a torch.Tensor with shape
                ``(2, n_rays,)``). Note that the first channel encodes the intersection
                information between the ray and the sphere that is processed at each
                iteration of the algorithm.
        """
        check_shapes_match(rays_d, rays_o, "rays_d", "rays_o")
        check_equal(rays_o.shape[-1], 3, "rays_o.shape[-1]", "3")

        directions = rays_d.reshape(-1, 3)[None, ...].repeat(2, 1, 1)
        origins = rays_o.reshape(-1, 3)[None, ...].repeat(2, 1, 1)

        if z_vals is None:
            z_vals = torch.zeros(directions.shape[:-1], device=rays_d.device)
        points = z_vals[..., None] * directions + origins

        # Compute the SDF values for the points that need to be processed. These
        # values will define the distance that we need to "travel" along the given
        # directions in order to get close to the surface.
        next_sdf_vals = self.sdf_model(points)
        mask_unfinished = torch.ones_like(next_sdf_vals, dtype=bool)

        # For determining the direction along which the tracer should move.
        sign = torch.ones_like(next_sdf_vals)
        sign[1, ...] = -1

        for iteration in range(self.n_iterations + 1):
            # Continue to process only the points for which the SDF values are above
            # a given threshold (far away from the surface).
            sdf_vals = torch.zeros_like(next_sdf_vals)
            sdf_vals[mask_unfinished] = next_sdf_vals[mask_unfinished]
            sdf_vals[sdf_vals <= self.sdf_threshold] = 0

            mask_unfinished = torch.logical_and(
                mask_unfinished, sdf_vals > self.sdf_threshold
            )
            if not torch.any(mask_unfinished) or iteration == self.n_iterations:
                break

            # Move along the ray directions based on the current SDF values.
            z_vals = z_vals + sign * sdf_vals
            points = z_vals[..., None] * directions + origins

            # Compute the next SDF values.
            next_sdf_vals = torch.zeros_like(next_sdf_vals)
            next_sdf_vals[mask_unfinished] = self.sdf_model(points[mask_unfinished])

            # Correct the SDF values for the points that overshoot.
            points, z_vals, next_sdf_vals = self.fix_overshoot(
                points, z_vals, directions, origins, sdf_vals, next_sdf_vals
            )

            # Update the masks.
            mask_unfinished = torch.logical_and(mask_unfinished, z_vals[0] < z_vals[1])

        return points, z_vals, mask_unfinished
