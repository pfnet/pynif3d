import os
import random

import numpy as np
import yaml

from pynif3d import logger
from pynif3d.common.verification import (
    check_bool,
    check_in_options,
    check_path_exists,
    check_pos_int,
)
from pynif3d.datasets import BaseDataset
from pynif3d.log.log_funcs import func_logger


class Shapes3dDataset(BaseDataset):
    """
    Loads ShapeNet and Synthetic Indoor Scene data from a given directory into a
    `Dataset` object.

    Please refer to the Convolutional Occupancy Networks (CON) paper for more
    information:
    https://arxiv.org/abs/2003.04618

    .. note::

        This implementation is based on the original one, which can be found here:
        https://github.com/autonomousvision/convolutional_occupancy_networks

    Usage:

    .. code-block:: python

        mode = "train"
        dataset = Shapes3dDataset(data_directory, mode)
    """

    # TODO: Add pointcloud-crop
    @func_logger
    def __init__(self, data_directory, mode, download=False, **kwargs):
        """
        Args:
            data_directory (str): The parent dictionary of the dataset.
            mode (str): The subset of the dataset. Has to be one of ("train", "val",
                test").
            download (bool): Flag indicating whether to automatically download the
                dataset (True) or not (False).
            kwargs (dict):
                - **categories** (list): List of strings defining the object categories.
                  Default is None.
                - **points_filename** (str): The name for the points file. Default is
                  "points.npz".
                - **pointcloud_filename** (str): The name for the pointcloud file.
                  Default is "pointcloud.npz".
                - **unpackbits** (bool): Boolean flag which defines if bit unpacking is
                  needed during point cloud and occupancy loading. Default is True.
                - **gt_point_sample_count** (uint): The number of the point samples
                  used as ground truth. Default is 2048.
                - **in_points_sample_count** (uint): The number of the point samples
                  used as input to the network. Default is 3000.
                - **in_points_noise_stddev** (float): The stddev for noise to add to
                  input points. Setting it to 0 will cancel noise addition. Default is
                  0.005.
        """
        super().__init__(data_directory, mode)

        if download:
            if os.path.exists(data_directory):
                logger.warning(
                    "Dataset already saved to: {}. Aborting download.".format(
                        data_directory
                    )
                )
            else:
                raise ValueError(
                    "Unable to download the ShapeNet and Synthetic Indoor Scene "
                    + "datasets, as registration is required for this purpose."
                )

        # Get input parameters
        categories = kwargs.get("categories", None)
        points_filename = kwargs.get("points_filename", "points.npz")
        pointcloud_filename = kwargs.get("pointcloud_filename", "pointcloud.npz")
        unpackbits = kwargs.get("unpackbits", True)
        gt_point_sample_count = kwargs.get("gt_point_sample_count", 2048)
        in_points_sample_count = kwargs.get("in_points_sample_count", 3000)
        in_points_noise_stddev = kwargs.get("in_points_noise_stddev", 0.005)

        # Verify inputs
        check_in_options(mode, ["train", "val", "test"], "mode")
        check_pos_int(gt_point_sample_count, "gt_point_sample_count")
        check_pos_int(in_points_sample_count, "in_points_sample_count")
        check_bool(unpackbits, "unpackbits")

        # Define categories
        # Use all the subfolder if category is not decided
        if categories is None:
            categories = os.listdir(data_directory)
            categories = [
                c for c in categories if os.path.isdir(os.path.join(data_directory, c))
            ]

        # Read or generate metadata
        metadata_file = os.path.join(data_directory, "metadata.yaml")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        # Set index for metadata
        for c_idx, c in enumerate(categories):
            metadata[c]["idx"] = c_idx

        # Get all models
        model_categories = []
        models = []
        for c in categories:
            subpath = os.path.join(data_directory, c)
            check_path_exists(subpath)

            split_file = os.path.join(subpath, mode + ".lst")
            with open(split_file, "r") as f:
                models_c = f.read().split("\n")

            if "" in models_c:
                models_c.remove("")

            model_categories += [c for _m in models_c]
            models += models_c

        # Save the dataset information to class
        self.model_categories = model_categories
        self.models = models
        self.metadata = metadata
        self.categories = categories
        self.points_filename = points_filename
        self.pointcloud_filename = pointcloud_filename
        self.unpackbits = unpackbits
        self.gt_point_sample_count = gt_point_sample_count
        self.in_points_sample_count = in_points_sample_count
        self.in_points_noise_stddev = in_points_noise_stddev

    def __len__(self):
        return len(self.model_categories)

    def __getitem__(self, item):
        model_category = self.model_categories[item]
        model = self.models[item]

        model_path = os.path.join(self.data_directory, model_category, model)

        # Load ground truth points
        # ===========================
        file_path = os.path.join(model_path, self.points_filename)

        # Support for synthetic dataset
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            filename = random.choice(files)
            file_path = os.path.join(file_path, filename)

        if not os.path.exists(file_path):
            # Check if it is multi-file
            logger.warning("file " + file_path + " could not be found, skipping")
            return {}

        query_points_dict = np.load(file_path)
        all_query_points = query_points_dict["points"]

        # Break symmetry if given in float16:
        if all_query_points.dtype == np.float16:
            all_query_points = all_query_points.astype(np.float32)
            all_query_points += 1e-4 * np.random.randn(*all_query_points.shape)

        # Load occupancies
        # ===========================
        all_gt_occupancies = query_points_dict["occupancies"]
        if self.unpackbits:
            all_gt_occupancies = np.unpackbits(all_gt_occupancies)[
                : all_query_points.shape[0]
            ]
        all_gt_occupancies = all_gt_occupancies.astype(np.float32)

        # Load input point clouds
        # ===========================
        file_path = os.path.join(model_path, self.pointcloud_filename)

        # Support for synthetic dataset
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            filename = random.choice(files)
            file_path = os.path.join(file_path, filename)

        check_path_exists(file_path)

        pointcloud_dict = np.load(file_path)
        in_points = pointcloud_dict["points"].astype(np.float32)
        in_normals = pointcloud_dict["normals"].astype(np.float32)

        # Subsample gt points and occupancies
        idx = np.random.randint(
            all_query_points.shape[0], size=self.gt_point_sample_count
        )
        query_points = all_query_points[idx]
        gt_occupancies = all_gt_occupancies[idx]

        # Subsample input points and their normals
        idx = np.random.randint(in_points.shape[0], size=self.in_points_sample_count)
        in_points = in_points[idx]
        in_normals = in_normals[idx]

        # Add noise to inputs
        noise = self.in_points_noise_stddev * np.random.randn(*in_points.shape)
        in_points += noise.astype(np.float32)

        # Prepare the output
        res = {
            "item_index": item,
            "query_points": query_points,
            "gt_occupancies": gt_occupancies,
            "in_points": in_points,
            "in_normals": in_normals,
        }

        if self.mode == "val" or self.mode == "test":
            res["all_query_points"] = all_query_points
            res["all_gt_occupancies"] = all_gt_occupancies

        return res
