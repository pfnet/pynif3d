import os

import imageio
import numpy as np

from pynif3d import logger
from pynif3d.common.camera import decompose_projection
from pynif3d.common.verification import check_equal, check_path_exists, is_image_file
from pynif3d.datasets.base_dataset import BaseDataset
from pynif3d.log.log_funcs import func_logger


class DTUMVSIDR(BaseDataset):
    """
    Implementation of the DTU MVS dataset, as used in the IDR paper:

    Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance
    Yariv et al., NeurIPS, 2020

    Please refer to the following paper for more information:
    https://arxiv.org/abs/2003.09852

    Usage:

    .. code-block:: python

        mode = "train"
        scan_id = 110
        dataset = DTUMVSIDR(data_directory, mode, scan_id)
    """

    dataset_url = "https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip?dl=1"
    dataset_md5 = "b1ad1eff5c4a4f99ae4d3503e976dafb"

    @func_logger
    def __init__(self, data_directory, mode, scan_id, download=False, **kwargs):
        """
        Args:
            data_directory (str): The dataset base directory (see BaseDataset).
            mode (str): The dataset usage mode (see BaseDataset).
            scan_id (int): ID of the scan.
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
            kwargs (dict):
                - **calibration_file** (str): The name of the calibration file. Default
                  is "cameras_linear_init.npz".
        """
        super(DTUMVSIDR, self).__init__(data_directory, mode)

        calibration_file = kwargs.get("calibration_file", "cameras.npz")
        base_directory = os.path.join(data_directory, "DTU", "scan" + str(scan_id))

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

        image_directory = os.path.join(base_directory, "image")
        check_path_exists(image_directory)
        images = self._load_images(image_directory)

        mask_directory = os.path.join(base_directory, "mask")
        masks = None
        if os.path.exists(mask_directory):
            masks = self._load_masks(mask_directory)
            check_equal(len(images), len(masks), "len(images)", "len(masks)")

        calibration_path = os.path.join(base_directory, calibration_file)
        check_path_exists(calibration_path)
        intrinsics, camera_poses = self._load_calibration(calibration_path)

        check_equal(len(images), len(intrinsics), "len(images)", "len(intrinsics)")
        check_equal(len(images), len(camera_poses), "len(images)", "len(camera_poses)")

        self.images = images
        self.masks = masks
        self.intrinsics = intrinsics
        self.camera_poses = camera_poses
        self.scan_id = scan_id

    def __getitem__(self, item):
        image = self.images[item]

        mask = None
        if self.masks is not None:
            mask = self.masks[item]

        intrinsics = self.intrinsics[item]
        camera_pose = self.camera_poses[item]

        return image, mask, intrinsics, camera_pose

    def __len__(self):
        return len(self.images)

    def _load_images(self, image_directory, rescale_min=-1, rescale_max=1):
        """
        Loads the RGB images (corresponding to a scene) from the base dataset directory.
        Args:
            image_directory (str): The dataset base directory.
            rescale_min (float): The minimum value the images should be normalized to.
            rescale_max (float): The maximum value the images should be normalized to.

        Returns:
            list: List containing the loaded images as torch.Tensor.
        """
        image_files = sorted(os.listdir(image_directory))
        image_files = [
            os.path.join(image_directory, f)
            for f in image_files
            if is_image_file(f) and not f.startswith("._")
        ]

        images = np.stack([imageio.imread(f) for f in image_files])
        images = images / 255.0 * (rescale_max - rescale_min) + rescale_min
        images = images.transpose(0, 3, 1, 2).astype(np.float32)
        return images

    def _load_masks(self, mask_directory, threshold=127.5):
        """
        Loads the mask images (corresponding to a scene) from the base dataset directory.
        Args:
            mask_directory (str): The mask directory.
            threshold (float): Foreground segmentation threshold.

        Returns:
            list: List containing the loaded images as torch.Tensor.
        """
        mask_files = sorted(os.listdir(mask_directory))
        mask_files = [
            os.path.join(mask_directory, f)
            for f in mask_files
            if is_image_file(f) and not f.startswith("._")
        ]

        masks = np.stack([imageio.imread(f, as_gray=True) for f in mask_files])
        masks = masks.astype(np.float32) > threshold
        masks = masks[..., None].transpose(0, 3, 1, 2)
        return masks

    def _load_calibration(self, calibration_path):
        """
        Loads the calibration data (corresponding to a scene) from the base dataset
        directory.
        Args:
            calibration_path (str): The path to the calibration file.

        Returns:
            tuple: Tuple containing the intrinsic parameters and camera poses. Each
                tuple element is a list containing tensors with shape ``(4, 4)``.
        """
        calibration_dict = np.load(calibration_path)
        calibration_keys = list(calibration_dict.keys())
        scale_matrices = [
            calibration_dict[key]
            for key in calibration_keys
            if key.startswith("scale_mat_") and not key.startswith("scale_mat_inv_")
        ]
        world_matrices = [
            calibration_dict[key]
            for key in calibration_keys
            if key.startswith("world_mat_") and not key.startswith("world_mat_inv_")
        ]

        intrinsics = []
        camera_poses = []

        T = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        for scale_matrix, world_matrix in zip(scale_matrices, world_matrices):
            P = (world_matrix @ scale_matrix)[:3, :4]

            K, Rt = decompose_projection(P[:3])

            # Fix the coordinate system to match the one from the ray generator.
            Rt = Rt @ T

            intrinsics.append(K)
            camera_poses.append(Rt)

        intrinsics = np.stack(intrinsics).astype(np.float32)
        camera_poses = np.stack(camera_poses).astype(np.float32)

        return intrinsics, camera_poses
