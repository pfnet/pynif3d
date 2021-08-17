import torch

from pynif3d.common.verification import check_pos_int
from pynif3d.log.log_funcs import func_logger


class UniformRaySampler(torch.nn.Module):
    """
    Randomly samples a ray. Takes as input a ray origin and direction, `near`, `far`,
    the number of points to sample and generates point coordinates across each given
    ray.

    Usage:

    .. code-block:: python

        near = 0.1
        far = 5.0
        n_samples = 1000

        sampler = UniformRaySampler(near, far, n_samples)
        points, z_vals = sampler(ray_directions, ray_origins)
    """

    @func_logger
    def __init__(self, near, far, n_samples, is_perturb=True):
        """
        Args:
            near (float): Minimum depth value corresponding to the sampled points.
            far (float): Maximum depth value corresponding to the sampled points.
            n_samples (int): Number of sampled points along the ray.
            is_perturb (bool): Boolean flag indicating whether to perturb the sampled
                points (True) or not (False). Default value is True.
        """
        super().__init__()

        check_pos_int(n_samples, "n_samples")
        self.n_samples = n_samples

        self.near = near
        self.far = far
        self.is_perturb = is_perturb

    def forward(self, rays_d, rays_o):
        """
        Args:
            rays_d (torch.Tensor): Tensor containing the ray directions. Its shape is
                ``(n_ray_samples, 3)`` or ``(batch_size, n_ray_samples, 3)``.
            rays_o (torch.Tensor): Tensor containing the ray origins. Its shape is
                ``(n_ray_samples, 3)`` or ``(batch_size, n_ray_samples, 3)``.

        Returns:
            tuple: Tuple containing the sampled points and the corresponding Z values.
        """
        zs = torch.linspace(0.0, 1.0, self.n_samples, device=rays_d.device)
        near = torch.as_tensor(self.near, device=rays_d.device)
        far = torch.as_tensor(self.far, device=rays_d.device)

        z_vals = near[..., None] * (1.0 - zs) + far[..., None] * zs
        rays_d_all = rays_d.reshape(-1, 3)
        rays_o_all = rays_o.reshape(-1, 3)
        points = rays_o_all[..., None, :] + z_vals[..., None] * rays_d_all[..., None, :]

        # Perturb points by adding some noise
        if self.is_perturb:
            # get intervals between samples
            mids = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)

            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        points = points.reshape(rays_d.shape[:-1] + (self.n_samples, 3))
        z_vals = z_vals.expand(rays_d.shape[:-1] + (self.n_samples,))
        return points, z_vals
