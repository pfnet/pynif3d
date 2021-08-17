import os

import gdown
from torch.utils.data.dataset import Dataset

from pynif3d import logger
from pynif3d.common.verification import check_in_options
from pynif3d.log.log_funcs import func_logger


class BaseDataset(Dataset):
    """
    Base dataset class. All the custom datasets shall inherit this class and implement
    the required functions by overriding them.
    """

    @func_logger
    def __init__(self, data_directory, mode):
        """
        Args:
            data_directory (str): The dataset root directory.
            mode (str): The dataset usage mode ("train", "val" or "test").
        """
        super(Dataset, self).__init__()
        choices = ["train", "val", "test"]

        check_in_options(mode, choices, "mode")

        self.data_directory = data_directory
        self.mode = mode

    def __getitem__(self, item):
        msg = "This function needs to be implemented in the derived class."
        logger.error(msg)
        raise NotImplementedError(msg)

    def __len__(self):
        msg = "This function needs to be implemented in the derived class."
        logger.error(msg)
        raise NotImplementedError()

    def download(self, url, save_directory, archive_format, md5=None):
        choices = ["zip", "tar", "tar.gz", "tar.bz2", "tgz", "tbz"]
        check_in_options(archive_format, choices, "archive_format")

        gdown.cached_download(
            url=url,
            path=os.path.join(save_directory, "dataset." + archive_format),
            md5=md5,
            postprocess=gdown.extractall,
        )
