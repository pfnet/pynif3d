import os

import numpy as np

from pynif3d.common.verification import check_path_exists, check_pos_int


def load_pose_from_file(filename):
    check_path_exists(filename)

    pose = np.loadtxt(filename)
    pose = pose.reshape([4, 4]).astype(np.float32)
    return pose


def load_poses_from_dir(poses_dir):
    check_path_exists(poses_dir)

    files = sorted(os.listdir(poses_dir))

    check_pos_int(len(files), "files")

    poses = [
        load_pose_from_file(os.path.join(poses_dir, f))
        for f in files
        if f.endswith("txt")
    ]
    poses = np.stack(poses, 0).astype(np.float32)

    transformation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.0],
        ],
        dtype=np.float32,
    )

    poses = poses @ transformation_matrix
    poses = poses[:, :3, :4].astype(np.float32)
    return poses
