import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pynif3d.common.verification import check_path_exists
from pynif3d.datasets import DTUMVSIDR
from pynif3d.pipeline import IDR
from pynif3d.utils.transforms import radians, rotation_mat, translation_mat


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--scan-id", "-s", type=int, required=True)
    parser.add_argument("--model-file", "-m", required=True)
    parser.add_argument("--save-directory", "-sd", required=True)
    args = parser.parse_args()
    return args


def create_render_poses(n_poses=20):
    rx = radians(-120)
    tz = translation_mat(4)
    rzs = np.linspace(radians(-180), radians(180), n_poses)

    poses = []
    for rz in rzs:
        pose = rotation_mat(rz, axis="z") @ rotation_mat(rx, axis="x") @ tz
        poses.append(torch.as_tensor(pose))

    poses = torch.stack(poses)
    return poses


def demo(dataset, model, args):
    """
    Computes the average PSNR/MSE on the test dataset.

    Args:
        dataset: Instance of the dataset class
        model: Instance of the model class
        args: Arguments passed to the script
    """
    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    model.to(device)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(args.save_directory, exist_ok=True)

    render_poses = create_render_poses(n_poses=len(test_loader)).to(device)

    for index, batch in tqdm(enumerate(test_loader)):
        images, masks, intrinsics, _ = batch
        image_height, image_width = images.shape[-2:]
        camera_poses = torch.as_tensor(render_poses[index], device=device)
        camera_poses = camera_poses[None, ...]

        images = torch.as_tensor(images, device=device)
        masks = torch.as_tensor(masks, device=device)
        intrinsics = torch.as_tensor(intrinsics, device=device)

        # Run the inference
        prediction = model(images, masks, intrinsics, camera_poses)

        # Create the visualization
        visual = prediction["rgb_vals"].reshape(-1, 3)
        visual = visual.reshape(image_height, image_width, 3)
        visual = (visual + 1.0) * 0.5 * 255
        visual = visual.detach().cpu().numpy().astype(np.uint8)
        visual = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)

        filename = os.path.join(args.save_directory, "image_" + str(index) + ".jpg")
        cv2.imwrite(filename, visual)
        print("Saved image to: ", filename)


def main():
    args = parse_arguments()

    # Set up the model
    check_path_exists(args.model_file)

    data = torch.load(args.model_file)
    state_dict = data["model_state"]

    # Create the dataset object
    dataset = DTUMVSIDR(
        args.data_directory,
        mode="train",
        scan_id=args.scan_id,
    )

    # Set up the model
    image_size = dataset.images.shape[-2:]
    model = IDR(image_size)
    model.load_state_dict(state_dict)

    # Run the training pipeline
    demo(dataset, model, args)


if __name__ == "__main__":
    main()
