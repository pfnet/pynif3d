import torch

from pynif3d.common.verification import check_true
from pynif3d.log.log_funcs import func_logger


class IDRSampleNetwork(torch.nn.Module):
    """
    Computes the differentiable intersection between the viewing ray and the surface.
    Please check equation 3 from the paper and section 3 from the supplementary material
    for more details.
    """

    @func_logger
    def forward(
        self,
        rays_d,
        rays_o,
        z_vals,
        z_pred,
        mask_surface,
        z_at_surface,
        grad_at_surface,
    ):
        """
        Args:
            rays_d (torch.Tensor): Tensor containing the directions of the rays. Its
                shape is ``(n_rays, 3)``.
            rays_o (torch.Tensor): Tensor containing the origins of the rays. Its shape
                is ``(n_rays, 3)``.
            z_vals (torch.Tensor): Tensor containing the distances to the surface. Its
                shape is ``(n_rays,)``.
            z_pred (torch.Tensor): Tensor containing the distances to the surface. Its
                shape is ``(n_rays,)``.
            mask_surface (torch.Tensor): Tensor containing the surface mask. Its shape
                is ``(n_rays, 3)``.
            z_at_surface (torch.Tensor): Tensor containing the SDF values computed at
                the surface points. Its shape is ``(n_points_intersect,)``.
            grad_at_surface (torch.Tensor): Tensor containing the gradient values at the
                intersection points. Its shape is ``(n_points_intersect, 3)``.

        Returns:
            torch.Tensor: Tensor containing the differentiable intersection points. Its
            shape is ``(n_points_intersect, 3)``.
        """

        # The original implementation expects rays_d to be detached. Check whether
        # rays_d.requires_grad is False or not.
        check_true(not rays_d.requires_grad, "not rays_d.requires_grad")

        rays_o_surface = rays_o[mask_surface]
        rays_d_surface = rays_d[mask_surface]
        z_vals_surface = z_vals[mask_surface]
        z_pred_surface = z_pred[mask_surface]

        dot_prod = torch.bmm(grad_at_surface[..., None, :], rays_d_surface[..., None])
        dot_prod = dot_prod.squeeze()
        zs_theta = z_vals_surface - (z_pred_surface - z_at_surface) / dot_prod

        points = rays_o_surface + zs_theta[..., None] * rays_d_surface
        return points
