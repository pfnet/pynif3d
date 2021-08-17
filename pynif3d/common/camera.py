import cv2
import numpy as np


def decompose_projection(projection):
    decomposition = cv2.decomposeProjectionMatrix(projection)
    K = decomposition[0]
    R = decomposition[1]
    t = decomposition[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4, dtype=projection.dtype)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=projection.dtype)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
