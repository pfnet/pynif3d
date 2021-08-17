import numpy as np

from pynif3d.common.verification import check_in_options


def normalize(x):
    """
    Normalizes an input vector.
    Args:
        x (np.array): Array containing the vector's coordinates.

    Returns:
        np.array: Array containing the normalized coordinates.
    """
    return x / np.linalg.norm(x)


def radians(theta_deg):
    """
    Converts an angle from degrees to radians
    Args:
        theta_deg (float): Angle value in degrees.

    Returns:
        float: Angle value in radians.
    """
    theta_rad = theta_deg / 180.0 * np.pi
    return theta_rad


def translation_mat(t, axis="z"):
    """
    Generates a translation matrix given an input translation vector.
    Args:
        t (float): Array containing the translation vector's coordinates.
        axis (str): Rotation axis ("x", "y" or "z"). Default is "z".

    Returns:
        np.array: Translation matrix of shape ``(4, 4)``.
    """
    choices = ["x", "y", "z"]
    check_in_options(axis, choices, "axis")
    translation = np.eye(4, dtype=np.float32)

    if axis == "x":
        translation[0, -1] = t
    elif axis == "y":
        translation[1, -1] = t
    else:
        translation[2, -1] = t

    return translation


def rotation_mat(angle, axis):
    """
    Creates a rotation matrix given an angle and a coordinate axis.
    Args:
        angle (float): Rotation angle (in radians).
        axis (str): Rotation axis ("x", "y" or "z").

    Returns:
        np.array: Rotation matrix of shape ``(4, 4)``.
    """
    choices = ["x", "y", "z"]
    check_in_options(axis, choices, "axis")

    rotations = {
        "x": np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        "y": np.array(
            [
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        "z": np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    }

    return rotations[axis]
