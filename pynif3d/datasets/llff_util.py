import os
from subprocess import check_output

import cv2
import numpy as np

from pynif3d import logger
from pynif3d.common.verification import check_equal, check_path_exists, check_pos_int
from pynif3d.utils.transforms import normalize


def render_path_spiral(camera_to_world, up, rads, focal, z_rate, n_rotations, n_views):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = camera_to_world[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * n_rotations, n_views + 1)[:-1]:
        c = np.dot(
            camera_to_world[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.0])
            * rads,
        )
        z = normalize(
            c - np.dot(camera_to_world[:3, :4], np.array([0, 0, -focal, 1.0]))
        )
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))
    return render_poses


def min_line_distance(rays_o, rays_d):
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    point_min_distance = np.squeeze(
        -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ b_i.mean(0)
    )
    return point_min_distance


def p34_to_44(p):
    result = np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )
    return result


def spherify_poses(poses, bds):
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]
    min_distance = min_line_distance(rays_o, rays_d)

    center = min_distance
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    camera_to_world = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(camera_to_world[None])) @ p34_to_44(
        poses[:, :3, :4]
    )

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    if rad.min() <= 1e-7:
        logger.warning("Zero-value has been encountered in `rad` variable.")

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    rad_circle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camera_origin = np.array([rad_circle * np.cos(th), rad_circle * np.sin(th), zh])
        up = np.array([0, 0, -1.0], dtype=np.float32)

        vec2 = normalize(camera_origin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camera_origin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def view_matrix(z, up, pos):
    v2 = normalize(z)
    v0 = normalize(np.cross(up, v2))
    v1 = normalize(np.cross(v2, v0))
    matrix = np.stack([v0, v1, v2, pos], 1)
    return matrix


def average_poses(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    v2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    camera_to_world = np.concatenate([view_matrix(v2, up, center), hwf], 1)
    return camera_to_world


def recenter_poses(poses):
    centered_poses = poses + 0
    bottom = np.asarray([[0, 0, 0, 1]], dtype=np.float32)
    camera_to_world = average_poses(poses)
    camera_to_world = np.concatenate([camera_to_world[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(camera_to_world) @ poses
    centered_poses[:, :3, :4] = poses[:, :3, :4]
    return centered_poses


def minify(data_dir, factors=None, resolutions=None):
    if factors is None:
        factors = []

    if resolutions is None:
        resolutions = []

    need_to_load = False

    for r in factors:
        image_dir = os.path.join(data_dir, "images_{}".format(r))
        if not os.path.exists(image_dir):
            need_to_load = True
            break

    for r in resolutions:
        image_dir = os.path.join(data_dir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(image_dir):
            need_to_load = True
            break

    if not need_to_load:
        return

    image_dir = os.path.join(data_dir, "images")
    extensions = [".JPG", ".jpg", ".png", ".jpeg", ".PNG"]
    filenames = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
    filenames = [f for f in filenames if os.path.splitext(f)[1] in extensions]
    image_dir_src = image_dir

    current_dir = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resize_arg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resize_arg = "{}x{}".format(r[1], r[0])

        image_dir_dst = os.path.join(data_dir, name)
        if os.path.exists(image_dir_dst):
            continue

        logger.info("Minifying", r, data_dir)

        os.makedirs(image_dir_dst)
        check_output("cp {}/* {}".format(image_dir_src, image_dir_dst), shell=True)

        ext = filenames[0].split(".")[-1]
        args = " ".join(
            ["mogrify", "-resize", resize_arg, "-format", "png", "*.{}".format(ext)]
        )
        os.chdir(image_dir_dst)
        check_output(args, shell=True)
        os.chdir(current_dir)

        if ext != "png":
            check_output("rm {}/*.{}".format(image_dir_dst, ext), shell=True)
            logger.info("Removed duplicates")


def load_poses_from_dir(data_dir):
    filename = os.path.join(data_dir, "poses_bounds.npy")
    check_path_exists(filename)

    data = np.load(filename)
    check_pos_int(data.size, "data.size")

    poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = data[:, -2:].transpose([1, 0])
    return poses, bds


def get_images_from_dir(image_dir):
    check_path_exists(image_dir)

    extensions = [".JPG", ".jpg", ".png"]
    filenames = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
    filenames = [f for f in filenames if os.path.splitext(f)[1] in extensions]
    return filenames


def preprocess_images(
    data_dir, resize_factor=None, desired_width=None, desired_height=None
):
    image_dir = os.path.join(data_dir, "images")
    filenames = get_images_from_dir(image_dir)
    check_pos_int(len(filenames[0]), "filenames[0]")

    image = cv2.imread(filenames[0])
    image_height, image_width = image.shape[:2]

    suffix = ""

    if resize_factor is not None:
        suffix = "_{}".format(resize_factor)
        minify(data_dir, factors=[resize_factor])
    elif desired_height is not None:
        resize_factor = image_height / float(desired_height)
        scaled_width = int(image_width / resize_factor)
        minify(data_dir, resolutions=[[desired_height, scaled_width]])
        suffix = "_{}x{}".format(scaled_width, desired_height)
    elif desired_width is not None:
        resize_factor = image_width / float(desired_width)
        scaled_height = int(image_height / resize_factor)
        minify(data_dir, resolutions=[[scaled_height, desired_width]])
        suffix = "_{}x{}".format(desired_width, scaled_height)

    image_dir = os.path.join(data_dir, "images" + suffix)
    filenames = get_images_from_dir(image_dir)
    check_pos_int(len(filenames[0]), "filenames[0]")

    images = [cv2.imread(f).astype(np.float32) / 255 for f in filenames]
    images = np.stack(images, 0)
    return images


def load_data(data_dir, resize_factor=None, desired_width=None, desired_height=None):
    poses, bds = load_poses_from_dir(data_dir)
    images = preprocess_images(data_dir, resize_factor, desired_width, desired_height)

    check_equal(len(images), poses.shape[-1], "images", "poses.shape[-1]")

    poses[:2, 4, :] = np.array(images[..., 0].shape[:2]).reshape([2, 1])
    if resize_factor is not None:
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / resize_factor

    return poses, bds, images
