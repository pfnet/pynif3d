import torch
from torchsearchsorted import searchsorted

from pynif3d.log.log_funcs import func_logger


class WeightedRaySampler(torch.nn.Module):
    """
    Neural implicit model-based importance sampling function for rays which  is used in
    the NeRF paper. For details, please check https://arxiv.org/abs/2003.08934.

    Usage:

    .. code-block:: python

        near = 0.1
        far = 5.0
        n_samples = 1000

        sampler = WeightedRaySampler(near, far, n_samples)
        points, z_vals = sampler(ray_directions, ray_origins, z_vals, weights)
    """

    @func_logger
    def __init__(self, near, far, n_sample, eps=1e-5):
        """
        Args:
            near (float): Minimum depth value corresponding to the sampled points.
            far (float): Maximum depth value corresponding to the sampled points.
            n_sample (int): Number of sampled points along the ray.
            eps (float): Epsilon that is added to the weights, in order to avoid zero
                values. Default value is 1e-5.
        """
        super().__init__()

        self.near = near
        self.far = far
        self.n_sample = n_sample
        self.eps = eps

    def forward(self, rays_d, rays_o, z_vals, weights, **kwargs):
        """
        Args:
            rays_d (torch.Tensor): Tensor containing ray directions. Its shape is
                ``(batch_size, n_rays, 3)``.
            rays_o (torch.Tensor): Tensor containing ray origins. Its shape is
                ``(batch_size, n_rays, 3)``.
            z_vals (torch.Tensor): Tensor containing Z values. Its shape is
                ``(n_rays, n_samples_per_ray,)``.
            weights (torch.Tensor): Tensor containing the sampling weights. Its shape is
                ``(batch_size, n_rays, n_samples_per_ray)``.
            kwargs (dict):
                - **is_deterministic** (bool): (Optional) Boolean flag indicating
                  whether to sample the rays in a deterministic manner (True) or not
                  (False). Default is False.

        Returns:
            tuple: Tuple containing the sampled points and Z values.
        """
        is_deterministic = kwargs.get("is_deterministic", False)

        z_vals = z_vals.reshape(-1, z_vals.shape[-1])
        z_vals_half = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
        z_samples = self.sample_pdf(
            z_vals_half, weights[..., 1:-1], is_deterministic=is_deterministic
        )
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(
            torch.cat([z_vals[None, :].expand(z_samples.shape), z_samples], -1), -1
        )

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return pts, z_vals

    def sample_pdf(self, bins, weights, is_deterministic=False):
        # Get pdf
        weights = weights + self.eps
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        # Take uniform samples
        if is_deterministic:
            u = torch.linspace(0.0, 1.0, self.n_sample, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.n_sample])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [self.n_sample], device=bins.device)

        # Invert CDF. Currently, searchsorted only supports 2D arrays.
        samples = []
        for batch_index in range(len(cdf)):
            inds = searchsorted(
                cdf[batch_index], u[batch_index].contiguous(), side="right"
            )
            below = torch.relu(inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)

            matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
            cdf_g = torch.gather(
                cdf[batch_index].unsqueeze(1).expand(matched_shape), 2, inds_g
            )
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u[batch_index] - cdf_g[..., 0]) / denom
            samples.append(bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]))

        samples = torch.stack(samples, dim=0)
        return samples
