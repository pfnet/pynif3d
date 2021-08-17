import os

import numpy as np

from pynif3d import logger
from pynif3d.common.verification import (
    check_in_options,
    check_lengths_match,
    check_path_exists,
    check_pos_int,
)
from pynif3d.datasets.base_dataset import BaseDataset
from pynif3d.datasets.llff_util import (
    average_poses,
    load_data,
    recenter_poses,
    render_path_spiral,
    spherify_poses,
)
from pynif3d.log.log_funcs import func_logger
from pynif3d.utils.transforms import normalize


class LLFF(BaseDataset):
    """
    Loads LLFF data from a given directory into a `Dataset` object.

    Please refer to the following paper for more information:
    https://arxiv.org/abs/1905.00889

    .. note::

        This implementation is based on the code from: https://github.com/bmild/nerf

    Usage:

    .. code-block:: python

        mode = "train"
        scan_id = "bus"
        dataset = LLFF(data_directory, mode, scan_id)
    """

    dataset_url = "https://drive.google.com/u/0/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g"
    dataset_md5 = "74cc8bd336e9a19fce3c03f4a1614c2d"

    @func_logger
    def __init__(
        self,
        data_directory,
        mode,
        scene,
        factor=8,
        recenter=True,
        bd_factor=0.75,
        spherify=False,
        path_z_flat=False,
        download=False,
    ):
        """
        Args:
            data_directory (str): The dataset base directory (see BaseDataset).
            mode (str): The dataset usage mode (see BaseDataset).
            scene (str): The scene name ("armchair", "bus", "cube"...).
            factor (float): The factor to reduce image size by. Default is 8.
            recenter (bool): Boolean flag indicating whether to re-center poses (True)
                or not (False). Default is True.
            bd_factor (float): The factor to rescale poses by. Default is 0.75.
            spherify (bool): Boolean flag indicating whether the poses should be
                converted to spherical coordinates (True) or not (False). Default is
                False.
            path_z_flat (bool): (TODO: Add explanation). Defaults to False.
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
        """
        super(LLFF, self).__init__(data_directory, mode)

        choices = [
            "fern",
            "flower",
            "fortress",
            "horns",
            "leaves",
            "orchids",
            "room",
            "trex",
        ]

        check_in_options(scene, choices, "scene")
        base_directory = os.path.join(data_directory, "nerf_llff_data", scene)

        if download:
            if os.path.exists(base_directory):
                logger.warning(
                    "Dataset already saved to: {}. Aborting download.".format(
                        base_directory
                    )
                )
            else:
                self.download(
                    url=self.dataset_url,
                    save_directory=data_directory,
                    archive_format="zip",
                    md5=self.dataset_md5,
                )

        check_path_exists(base_directory)

        poses, bds, images = load_data(base_directory, factor)

        # Correct rotation matrix ordering
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )

        # Change the first dimension to reflect the batch size
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        scale = 1
        min_bds = np.min(bds)
        max_bds = np.max(bds)

        if bd_factor is not None:
            scale = 1.0 / (min_bds * bd_factor)
        poses[:, :3, 3] *= scale
        bds *= scale

        if recenter:
            poses = recenter_poses(poses)

        if spherify:
            poses, render_poses, bds = spherify_poses(poses, bds)
        else:
            camera_to_world = average_poses(poses)
            up = normalize(poses[:, :3, 1].sum(0))

            # Find a reasonable "focus depth" for this dataset
            close_depth, inf_depth = min_bds * 0.9, max_bds * 5.0
            dt = 0.75
            mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
            focal = mean_dz

            # Get radii for spiral path
            tt = poses[:, :3, 3]
            rads = np.percentile(np.abs(tt), 90, 0)
            camera_to_world_path = camera_to_world
            n_views = 120
            n_rotations = 2

            if path_z_flat:
                z_loc = -close_depth * 0.1
                camera_to_world_path[:3, 3] = (
                    camera_to_world_path[:3, 3] + z_loc * camera_to_world_path[:3, 2]
                )
                rads[2] = 0.0
                n_rotations = 1
                n_views /= 2

            # Generate poses for the spiral path.
            render_poses = render_path_spiral(
                camera_to_world_path,
                up,
                rads,
                focal,
                z_rate=0.5,
                n_rotations=n_rotations,
                n_views=n_views,
            )

        images = images.astype(np.float32)
        poses = poses.astype(np.float32)
        render_poses = np.asarray(render_poses, dtype=np.float32)

        check_pos_int(len(images), "images")
        check_lengths_match(images, poses, "images", "poses")

        self.images = images
        self.poses = poses
        self.render_poses = render_poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        pose = self.poses[item]
        return image, pose
