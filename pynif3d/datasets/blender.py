import json
import os

import cv2
import numpy as np

from pynif3d import logger
from pynif3d.common.verification import (
    check_in_options,
    check_lengths_match,
    check_path_exists,
    check_pos_int,
)
from pynif3d.datasets.base_dataset import BaseDataset
from pynif3d.log.log_funcs import func_logger


class Blender(BaseDataset):
    """
    Implementation of the synthetic dataset (Blender).

    Please refer to the following paper for more information:
    https://arxiv.org/abs/2003.08934

    .. note::

        This implementation is based on the code from: https://github.com/bmild/nerf

    Usage:

    .. code-block:: python

        mode = "train"
        scene = "chair"
        dataset = Blender(data_directory, mode, scene)
    """

    dataset_url = "https://drive.google.com/u/0/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    dataset_md5 = "ac0cfb13b1e4ff748b132abc8e8c26b6"

    @func_logger
    def __init__(
        self,
        data_directory,
        mode,
        scene,
        half_resolution=False,
        white_background=True,
        download=False,
    ):
        """
        Args:
            data_directory (str): The dataset base directory (see BaseDataset).
            mode (str): The dataset usage mode (see BaseDataset).
            scene (str): The scene name ("chair", "drums", "ficus"...).
            half_resolution (bool): Boolean indicating whether to load the dataset in
                half resolution (True) or full resolution (False)
            white_background (bool): Boolean indicating whether to set the dataset's
                background color to white (True) or leave it as it is (False)
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
        """
        super(Blender, self).__init__(data_directory, mode)

        choices = [
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
        ]

        check_in_options(scene, choices, "scene")
        base_directory = os.path.join(data_directory, "nerf_synthetic", scene)

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

        filename = os.path.join(base_directory, "transforms_{}.json".format(mode))
        with open(filename, "r") as f:
            meta = json.load(f)

        images = []
        camera_poses = []

        image_height = None
        image_width = None
        focal_length = None
        camera_angle_x = float(meta["camera_angle_x"])

        for frame in meta["frames"]:
            filename = os.path.join(base_directory, frame["file_path"] + ".png")
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # Ignore the alpha channel during conversion (if it exists)
            image[..., :3] = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)

            if image_height is None:
                image_height, image_width = image.shape[:2]
                focal_length = 0.5 * image_width / float(np.tan(0.5 * camera_angle_x))

            if half_resolution:
                image = cv2.resize(
                    image, (image_height // 2, image_width // 2), cv2.INTER_AREA
                )

            image = image.astype(np.float32) / 255
            images.append(image)
            camera_pose = np.asarray(frame["transform_matrix"], dtype=np.float32)
            camera_poses.append(camera_pose)

        check_pos_int(len(images), "images")
        check_lengths_match(images, camera_poses, "images", "camera_poses")

        images = np.stack(images)
        camera_poses = np.stack(camera_poses)

        if half_resolution:
            image_height = image_height // 2
            image_width = image_width // 2
            focal_length = focal_length / 2

        if white_background:
            images[..., :3] = images[..., :3] * images[..., -1:] + (
                1 - images[..., -1:]
            )

        self.images = images
        self.image_size = (image_height, image_width)
        self.focal_length = (focal_length, focal_length)
        self.camera_poses = camera_poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        camera_pose = self.camera_poses[item]
        return image, camera_pose
