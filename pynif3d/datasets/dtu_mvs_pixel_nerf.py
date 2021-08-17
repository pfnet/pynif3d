import os
import shutil

import numpy as np

from pynif3d import logger
from pynif3d.common.verification import check_path_exists
from pynif3d.datasets import DTUMVSIDR
from pynif3d.datasets.base_dataset import BaseDataset
from pynif3d.log.log_funcs import func_logger


class DTUMVSPixelNeRF(BaseDataset):
    """
    Implementation of the DTU MVS dataset, as used in the pixelNeRF paper:

    pixelNeRF: Neural Radiance Fields from One or Few Images
    Yu et al., CVPR, 2021

    Please refer to the following paper for more information:
    https://arxiv.org/abs/2012.02190
    """

    dataset_url = "https://drive.google.com/uc?id=1aTSmJa8Oo2qCc2Ce2kT90MHEA6UTSBKj"
    dataset_md5 = "02af85c542238d9832e348caee2a6bba"

    @func_logger
    def __init__(self, data_directory, mode, scan_ids_file, download=False):
        """
        Args:
            data_directory (str): The dataset base directory (see BaseDataset).
            mode (str): The dataset usage mode (see BaseDataset).
            scan_ids_file (str): The path to the file that contains the IDs of the scans
                that need to be processed.
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
        """
        super(DTUMVSPixelNeRF, self).__init__(data_directory, mode)
        base_directory = os.path.join(data_directory, "DTU")

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
                extract_directory = os.path.join(data_directory, "rs_dtu_4")
                shutil.move(os.path.join(extract_directory, "DTU"), data_directory)
                shutil.rmtree(extract_directory)

        check_path_exists(scan_ids_file)

        with open(scan_ids_file, "r") as stream:
            scan_ids = [int(s.strip().replace("scan", "")) for s in stream.readlines()]

        datasets = [DTUMVSIDR(data_directory, mode, scan_id) for scan_id in scan_ids]

        # In the original code the intrinsics are averaged over images from one scan.
        self.intrinsics = [
            x.intrinsics.mean(axis=0).astype(np.float32) for x in datasets
        ]

        # In the original code the rays are processed in the camera's coordinate system.
        world_to_camera = np.asarray(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.camera_poses = [
            (world_to_camera @ x.camera_poses).astype(np.float32) for x in datasets
        ]
        self.images = [x.images.astype(np.float32) for x in datasets]
        self.scan_id = [[x.scan_id] * len(x) for x in datasets]

    def __getitem__(self, item):
        image = self.images[item]
        intrinsics = self.intrinsics[item]
        camera_pose = self.camera_poses[item]
        return image, intrinsics, camera_pose

    def __len__(self):
        return len(self.images)
