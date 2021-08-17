import torch

from pynif3d.camera import CameraRayGenerator
from pynif3d.common.verification import check_equal, check_iterable, check_pos_int
from pynif3d.log.log_funcs import func_logger
from pynif3d.models.idr.nif_model import IDRNIFModel
from pynif3d.models.idr.ray_tracer import IDRRayTracer
from pynif3d.models.idr.rendering_model import IDRRenderingModel
from pynif3d.models.idr.sample_network import IDRSampleNetwork
from pynif3d.sampling import AllPixelSampler, RandomPixelSampler


class IDR(torch.nn.Module):
    """
    This is the main pipeline function for the Implicit Differentiable Renderer (IDR)
    algorithm:
    https://arxiv.org/abs/2003.09852

    It takes an image, object mask, intrinsic parameters and camera pose as input and
    returns the reconstructed 3D points, the rendered pixel values and the predicted
    mask, given the input pose. During training it also returns the predicted Z values
    of the sampled points, along with the value of `gradient_theta`, used in the
    computation of the eikonal loss.

    Usage:
    .. code-block:: python

        image_size = (image_height, image_width)
        model = IDR(image_size)
        pred_dict = model(image, object_mask, intrinsics, camera_poses)
    """

    @func_logger
    def __init__(
        self,
        image_size,
        n_rays_per_image=2048,
        input_sampler_training=None,
        input_sampler_inference=None,
        nif_model=None,
        rendering_fn=None,
    ):
        """
        Args:
            image_size (tuple): Tuple containing the image size, expressed as
                ``(image_height, image_width)``.
            n_rays_per_image (int): The number of rays to be sampled for each image.
                Default value is 2048.
            input_sampler_training (torch.nn.Module): The ray sampler to be used during
                training. If set to None, it will default to RandomPixelSampler. Default
                value is None.
            input_sampler_inference (torch.nn.Module): The ray sampler to be used during
                inference. If set to None, it will default to AllPixelSampler. Default
                value is None.
            nif_model (torch.nn.Module): NIF model for outputting the prediction. If set
                to None, it will default to IDRNIFModel. Default value is None.
            rendering_fn (torch.nn.Module): The rendering function to be used during
                both training and inference. If set to None, it will default to
                IDRRenderingModel. Default value is None.
        """
        super().__init__()

        check_iterable(image_size, "image_size")
        check_equal(len(image_size), 2, "len(image_size)", "2")
        image_height, image_width = image_size

        check_pos_int(image_height, "image_height")
        check_pos_int(image_width, "image_width")

        check_pos_int(n_rays_per_image, "n_rays_per_image")
        self.n_rays_per_image = n_rays_per_image

        self.input_sampler_training = input_sampler_training
        if input_sampler_training is None:
            self.input_sampler_training = RandomPixelSampler(image_height, image_width)

        self.input_sampler_inference = input_sampler_inference
        if input_sampler_inference is None:
            self.input_sampler_inference = AllPixelSampler(image_height, image_width)

        self.nif_model = nif_model
        if nif_model is None:
            self.nif_model = IDRNIFModel()

        self.rendering_fn = rendering_fn
        if rendering_fn is None:
            self.rendering_fn = IDRRenderingModel()

        self.ray_tracer = IDRRayTracer(lambda x: self.nif_model(x)["sdf_vals"])
        self.sample_network = IDRSampleNetwork()

    def forward(self, image, object_mask, intrinsics, camera_poses, **kwargs):
        """
        Args:
            image (torch.Tensor): Tensor containing the input images. Its shape is
                ``(batch_size, 3, image_height, image_width)``.
            object_mask (torch.Tensor): Tensor containing the object masks. Its shape is
                ``(batch_size, 1, image_height, image_width)``.
            intrinsics (torch.Tensor): Tensor containing the camera intrinsics. Its
                shape is ``(batch_size, 4, 4)``.
            camera_poses (torch.Tensor): Tensor containing the camera poses. Its shape
                is ``(batch_size, 4, 4)``.
            kwargs (dict):
                - **chunk_size** (int): The chunk size of the tensor that is passed for
                  NIF prediction.

        Returns:
            dict: Dictionary containing the prediction outputs: the 3D coordinates of
            the intersection points + corresponding RGB values + the ray-to-surface
            intersection mask (used in training and inference) and Z values + gradient
            theta + sampled 3D coordinates (used in training only).
        """
        chunk_size = kwargs.get("chunk_size", 10000)

        height, width = image.shape[-2:]

        rays_d_all = []
        rays_o_all = []

        for camera_pose, K in zip(camera_poses, intrinsics):
            focal_x = K[0, 0].item()
            focal_y = K[1, 1].item()
            center_x = K[0, 2].item()
            center_y = K[1, 2].item()

            ray_generator = CameraRayGenerator(
                height, width, focal_x, focal_y, center_x, center_y
            )
            rays_o, _, rays_d = ray_generator(camera_pose[..., :3, :])
            rays_d_all.append(rays_d)
            rays_o_all.append(rays_o)

        rays_d = torch.cat(rays_d_all)
        rays_o = torch.cat(rays_o_all)

        if self.training:
            sampled_data = self.input_sampler_training(
                self.n_rays_per_image,
                rays_o=rays_o,
                rays_d=rays_d,
                rays_m=object_mask,
            )
        else:
            sampled_data = self.input_sampler_inference(
                rays_o=rays_o,
                rays_d=rays_d,
                rays_m=object_mask,
            )

        rays_o = sampled_data["rays_o"].reshape(-1, 3)
        rays_d = sampled_data["rays_d"].reshape(-1, 3)
        rays_m = sampled_data["rays_m"].reshape(-1)
        sample_coordinates = sampled_data["sample_coordinates"]

        rays_o_chunks = rays_o.split(chunk_size)
        rays_d_chunks = rays_d.split(chunk_size)
        rays_m_chunks = rays_m.split(chunk_size)
        sample_coordinates_chunks = sample_coordinates.split(chunk_size)

        points_out = []
        rgb_vals_out = []
        mask_out = []
        z_pred_out = []
        gradient_theta_out = []
        sample_coordinates_out = []

        for rays_o, rays_d, rays_m, sample_coordinates in zip(
            rays_o_chunks, rays_d_chunks, rays_m_chunks, sample_coordinates_chunks
        ):
            sample_coordinates_out.append(sample_coordinates)

            # Run the ray tracing algorithm and get the intersection points between the
            # rays and the surface.
            self.nif_model.eval()

            with torch.no_grad():
                points, z_vals, mask = self.ray_tracer(rays_d, rays_o, rays_m)

            self.nif_model.train()

            # Sample differentiable surface points and compute `gradient_theta`.
            if self.training:
                z_pred = self.nif_model(points)["sdf_vals"]
                z_pred_out.append(z_pred)

                mask_surface = torch.logical_and(mask, rays_m)
                points_surface = points[mask_surface]

                n_eikon_points = len(rays_d) // 2
                points_eikonal = torch.rand((n_eikon_points, 3), device=rays_d.device)
                points_eikonal = points_eikonal * 2.0 - 1.0  # Scale to [-1, 1]
                points_eikonal = torch.cat(
                    [points_eikonal, points.clone().detach()], dim=0
                )

                points_all = torch.cat([points_surface, points_eikonal], dim=0)
                grad_z_all = self.compute_gradient(points_all)
                gradient_eikon = grad_z_all[: len(points_surface), :].detach()
                gradient_theta = grad_z_all[len(points_surface) :, :]
                gradient_theta_out.append(gradient_theta)

                if torch.any(mask_surface):
                    z_at_surface = self.nif_model(points_surface)["sdf_vals"].detach()
                    rays_d_detached = rays_d.detach()
                    points_surface = self.sample_network(
                        rays_d=rays_d_detached,
                        rays_o=rays_o,
                        z_vals=z_vals,
                        z_pred=z_pred,
                        mask_surface=mask_surface,
                        z_at_surface=z_at_surface,
                        grad_at_surface=gradient_eikon,
                    )
            else:
                mask_surface = mask
                points_surface = points[mask_surface]
                points = points.detach()
                mask = mask.detach()

            rgb_vals = torch.ones_like(points)
            if len(points_surface) > 0:
                rgb_vals[mask_surface] = self.compute_rgb_values(
                    points=points_surface, view_dirs=-rays_d[mask_surface]
                )

            if not self.training:
                rgb_vals = rgb_vals.detach()

            points_out.append(points)
            rgb_vals_out.append(rgb_vals)
            mask_out.append(mask)

        output_dict = {
            "points": torch.stack(points_out),
            "rgb_vals": torch.stack(rgb_vals_out),
            "mask": torch.stack(mask_out),
        }

        if self.training:
            output_dict["z_pred"] = torch.stack(z_pred_out)
            output_dict["gradient_theta"] = torch.stack(gradient_theta_out)
            output_dict["sample_coordinates"] = torch.stack(sample_coordinates_out)

        return output_dict

    def compute_rgb_values(self, points, view_dirs):
        features = self.nif_model(points)["features"]
        gradient = self.compute_gradient(points)
        rgb_vals = self.rendering_fn(points, features, gradient, view_dirs)
        rgb_vals = torch.tanh(rgb_vals)
        return rgb_vals

    def compute_gradient(self, points):
        points.requires_grad_(True)
        z_vals = self.nif_model(points)["sdf_vals"]
        d_z_vals = torch.ones_like(z_vals, requires_grad=False)
        grad_z = torch.autograd.grad(
            outputs=z_vals,
            inputs=points,
            grad_outputs=d_z_vals,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        grad_z = grad_z[0]
        return grad_z
