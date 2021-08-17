import os

import cv2
import numpy as np

from pynif3d import logger
from pynif3d.common.verification import check_equal, check_in_options, check_pos_int
from pynif3d.datasets.base_dataset import BaseDataset
from pynif3d.datasets.deep_voxels_util import load_poses_from_dir
from pynif3d.log.log_funcs import func_logger


class DeepVoxels(BaseDataset):
    """
    Loads DeepVoxels data from a given directory into a `Dataset` object.

    Please refer to the following paper for more information:
    https://arxiv.org/abs/1812.01024

    Project page: https://vsitzmann.github.io/deepvoxels

    .. note::

        This implementation is based on the code from: https://github.com/bmild/nerf

    Usage:

    .. code-block:: python

        mode = "train"
        scene = "bus"
        dataset = DeepVoxels(data_directory, mode, scene)
    """

    dataset_url = "https://drive.google.com/u/0/uc?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl"
    dataset_md5 = "d715b810f1a6c2a71187e3235b2c5c56"

    @func_logger
    def __init__(self, data_directory, mode, scene, download=False):
        """
        Args:
            data_directory (str): The dataset base directory (see BaseDataset).
            mode (str): The dataset usage mode (see BaseDataset).
            scene (str): The scene name ("armchair", "bus", "cube"...).
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
        """
        super(DeepVoxels, self).__init__(data_directory, mode)

        choices = [
            "armchair",
            "bus",
            "cube",
            "greek",
            "shoe",
            "vase",
        ]

        check_in_options(scene, choices, "choices")

        if mode == "val":
            mode = "validation"

        base_directory = os.path.join(data_directory, mode, scene)

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

        poses = load_poses_from_dir(os.path.join(base_directory, "pose"))

        image_dir = os.path.join(base_directory, "rgb")
        filenames = [f for f in sorted(os.listdir(image_dir)) if f.endswith("png")]
        images = [cv2.imread(os.path.join(image_dir, f)) / 255.0 for f in filenames]
        images = np.stack(images, 0).astype(np.float32)

        check_pos_int(len(images), "images")
        check_equal(len(images), len(poses), "images", "poses")

        self.images = images
        self.poses = poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        pose = self.poses[item]
        return image, pose
