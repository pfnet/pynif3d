import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pynif3d.common.verification import check_path_exists
from pynif3d.pipeline.nerf import NeRF
from pynif3d.utils.transforms import radians, rotation_mat, translation_mat


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", "-m", required=True)
    parser.add_argument("--save-directory", "-sd", required=True)
    args = parser.parse_args()
    return args


def create_render_poses():
    rx = radians(60)
    tz = translation_mat(4)
    rzs = np.linspace(radians(-180), radians(180), 20)

    poses = []
    for rz in rzs:
        pose = rotation_mat(rz, axis="z") @ rotation_mat(rx, axis="x") @ tz
        poses.append(torch.as_tensor(pose))

    poses = torch.stack(poses)
    return poses


def demo(model, image_size, save_directory):
    """
    Renders images from novel viewpoints and creates a movie.

    Args:
        model (NeRF): Instance of the NeRF model.
        image_size (tuple): Dataset image size.
        save_directory (str): Directory where the output video will be saved to.

    Returns:
        None
    """
    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    model.to(device)
    model.eval()

    poses = create_render_poses().to(device)

    frames = []
    for camera_pose in tqdm(poses):
        with torch.no_grad():
            camera_pose = camera_pose[None, :3, ...]
            prediction = model(camera_pose)["rgb_map"][-1]
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.reshape(image_size + (3,))
            prediction = (prediction * 255).astype(np.uint8)
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            frames.append(prediction)

    filename = os.path.join(save_directory, "demo.avi")
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = cv2.VideoWriter(filename, fourcc, 1, image_size)

    for frame in frames:
        writer.write(frame)

    writer.release()
    print("Saved video to: ", filename)


def main():
    args = parse_arguments()

    # Set up the model.
    check_path_exists(args.model_file)

    data = torch.load(args.model_file)
    image_size = data["image_size"]
    focal_length = data["focal_length"]
    state_dict = data["model_state"]

    model = NeRF(image_size, focal_length)
    model.load_state_dict(state_dict)

    # Run the pipeline
    demo(model, image_size, args.save_directory)


if __name__ == "__main__":
    main()
