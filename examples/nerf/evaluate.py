import argparse
import json

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from pynif3d.common.verification import check_path_exists
from pynif3d.datasets import Blender
from pynif3d.loss.conversions import mse_to_psnr
from pynif3d.pipeline.nerf import NeRF


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--scene", "-s", required=True)
    parser.add_argument("--model-file", "-m", required=True)
    parser.add_argument("--half-resolution", "-hr", default=True)
    parser.add_argument("--white-background", "-b", default=True)
    parser.add_argument("--output-file", "-o", default="./evaluation.json")
    args = parser.parse_args()
    return args


def evaluate(dataset, model, args):
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
    model.train(False)

    test_loader = DataLoader(dataset, num_workers=4)
    average_mse = 0
    average_psnr = 0

    for batch in tqdm(test_loader):
        # Preprocess the data
        image, camera_pose = batch
        batch_size = len(image)

        image = torch.as_tensor(image, device=device)
        image = image[..., :3].permute(0, 3, 1, 2)
        camera_pose = torch.as_tensor(camera_pose, device=device)
        camera_pose = camera_pose[:, :3, :]

        # Run the inference
        with torch.no_grad():
            prediction = model(camera_pose)
            predicted_pixels = prediction["rgb_map"]
            target_pixels = image.reshape(batch_size, 3, -1).transpose(1, 2)

            mse = mse_loss(predicted_pixels[-1], target_pixels)
            average_mse += mse

            psnr = mse_to_psnr(mse)
            average_psnr += psnr

    average_mse /= len(dataset)
    average_psnr /= len(dataset)

    output = {
        "average_mse": float(average_mse),
        "average_psnr": float(average_psnr),
    }
    print(output)

    with open(args.output_file, "w") as json_file:
        json.dump(output, json_file)
        print("Saved results to:", args.output_file)


def main():
    args = parse_arguments()

    # Set up the model
    check_path_exists(args.model_file)

    data = torch.load(args.model_file)
    image_size = data["image_size"]
    focal_length = data["focal_length"]
    state_dict = data["model_state"]

    # Create the dataset object
    dataset = Blender(
        args.data_directory,
        mode="test",
        scene=args.scene,
        half_resolution=args.half_resolution,
        white_background=args.white_background,
    )
    test_skip = 8
    dataset.images = dataset.images[::test_skip]
    dataset.camera_poses = dataset.camera_poses[::test_skip]

    # Set up the model
    model = NeRF(image_size, focal_length)
    model.load_state_dict(state_dict)

    # Run the training pipeline
    evaluate(dataset, model, args)


if __name__ == "__main__":
    main()
