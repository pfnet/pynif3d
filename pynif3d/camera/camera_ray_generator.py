import torch

from pynif3d.common.verification import check_pos_int, check_true
from pynif3d.log.log_funcs import func_logger


class CameraRayGenerator(torch.nn.Module):
    """
    Generates rays for each pixel of a pinhole camera model. Takes the spatial
    dimensions of the camera's image plane along with focal lengths and camera
    pose to generate a ray for each pixel.

    Usage:

    .. code-block:: python

        ray_generator = CameraRayGenerator(image_height, image_width, focal_x, focal_y)
        ray_directions, ray_origins, view_directions = ray_generator(camera_poses)
    """

    @func_logger
    def __init__(self, height, width, focal_x, focal_y, center_x=None, center_y=None):
        super().__init__()

        check_pos_int(height, "height")
        check_pos_int(width, "width")

        if center_x is None:
            center_x = 0.5 * width
        if center_y is None:
            center_y = 0.5 * height

        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
        )
        ray_directions = torch.stack(
            [
                (i - center_x) / focal_x,
                -(j - center_y) / focal_y,
                -torch.ones_like(i),
            ],
            -1,
        )
        self.register_buffer("ray_directions", ray_directions)

    def forward(self, camera_poses):
        """
        Args:
            camera_poses (torch.Tensor): Tensor containing the Rt pose matrices. Its
                shape is ``(N, 3, 4)`` or ``(3, 4)``.

        Returns:
            tuple: Tuple containing:
                - **rays_o** (torch.Tensor): The origin coordinates of the rays. Its
                  shape is ``(batch_size, 3, height, width)``.
                - **rays_d** (torch.Tensor): The non-normalized direction vector of the
                  rays. Its shape is ``(batch_size, 3, height, width)``.
                - **view_dirs** (torch.Tensor): The unit-normalized direction vector of
                  the rays. Its shape is ``(batch_size, 3, height, width)``.
        """
        valid_shape = camera_poses.ndim in [2, 3] and camera_poses.shape[-2:] == (3, 4)
        check_true(valid_shape, "valid_shape")

        if camera_poses.ndim == 2:
            camera_poses = camera_poses[None, ...]

        ray_directions = self.ray_directions[..., None, None, :]
        ray_directions = ray_directions.to(camera_poses.device)

        rays_d = torch.sum(ray_directions * camera_poses[:, :3, :3], -1)
        rays_o = camera_poses[:, :3, -1].expand(rays_d.shape)
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rays_d = rays_d.permute(2, 3, 1, 0)
        rays_o = rays_o.permute(2, 3, 1, 0)
        view_dirs = view_dirs.permute(2, 3, 1, 0)

        return rays_o, rays_d, view_dirs
