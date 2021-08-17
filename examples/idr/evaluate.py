import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pynif3d.common.verification import check_path_exists
from pynif3d.datasets import DTUMVSIDR
from pynif3d.loss.conversions import mse_to_psnr
from pynif3d.pipeline import IDR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--scan-id", "-s", type=int, default=110)
    parser.add_argument("--model-file", "-m", required=True)
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
    model.eval()

    test_loader = DataLoader(dataset)
    average_mse = 0
    average_psnr = 0

    for batch in tqdm(test_loader):
        # Preprocess the data
        images, masks, intrinsics, camera_poses = batch

        images = torch.as_tensor(images, device=device)
        masks = torch.as_tensor(masks, device=device)
        intrinsics = torch.as_tensor(intrinsics, device=device)
        camera_poses = torch.as_tensor(camera_poses, device=device)

        # Run the inference
        prediction = model(images, masks, intrinsics, camera_poses)
        target_mask = masks.reshape(-1)[..., None]

        predicted_rgb_vals = prediction["rgb_vals"].reshape(-1, 3)
        predicted_rgb_vals = (predicted_rgb_vals + 1.0) * 0.5
        predicted_rgb_vals = predicted_rgb_vals * target_mask

        target_rgb_vals = images.permute(0, 2, 3, 1).reshape(-1, 3)
        target_rgb_vals = (target_rgb_vals + 1.0) * 0.5
        target_rgb_vals = target_rgb_vals * target_mask

        mse = torch.mean(
            ((predicted_rgb_vals - target_rgb_vals) ** 2) * predicted_rgb_vals.size(0)
        ) / torch.sum(target_mask)
        average_mse += mse

        psnr = mse_to_psnr(mse)
        average_psnr += psnr

        print("MSE: ", float(mse), " PSNR: ", float(psnr))

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

    # Run the evaluation pipeline
    evaluate(dataset, model, args)


if __name__ == "__main__":
    main()
