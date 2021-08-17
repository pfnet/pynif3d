import argparse
import os
import random

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pynif3d.common.verification import check_equal
from pynif3d.datasets import Blender
from pynif3d.loss.conversions import mse_to_psnr
from pynif3d.pipeline.nerf import NeRF

model_filename = "model.pt"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--save-directory", "-sd", default="./saved_models")
    parser.add_argument("--batch-size", "-bs", type=int, default=1)
    parser.add_argument("--scene", "-s", default="lego")
    parser.add_argument("--lr", "-l", type=float, default=5e-4)
    parser.add_argument("--decay-rate", "-dr", type=float, default=0.1)
    parser.add_argument("--decay-step", "-ds", type=float, default=250000)
    parser.add_argument("--n-iterations", "-i", type=int, default=1000000)
    parser.add_argument("--random-seed", "-rs", type=int, default=None)
    parser.add_argument("--half-resolution", "-hr", type=bool, default=True)
    parser.add_argument("--white-background", "-wb", type=bool, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return args


def save_checkpoint(
    save_directory, model, optimizer, image_size, focal_length, iteration
):
    checkpoint_path = os.path.join(save_directory, model_filename)
    data = {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "focal_length": focal_length,
        "image_size": image_size,
    }
    torch.save(data, checkpoint_path)
    print("Saved checkpoint to:", checkpoint_path)


def load_checkpoint(save_directory):
    checkpoint_path = os.path.join(save_directory, model_filename)

    if not os.path.exists(checkpoint_path):
        print("No checkpoint paths found in:", save_directory)
        print("Will start training from scratch...")
        return

    checkpoint_data = torch.load(checkpoint_path)
    return checkpoint_data


def train(dataset, model, optimizer, args):
    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    model.to(device)
    model.train()
    start_iteration = 0

    if args.resume:
        checkpoint = load_checkpoint(args.save_directory)
        if checkpoint is not None:
            check_equal(
                dataset.image_size,
                checkpoint["image_size"],
                "dataset.image_size",
                "checkpoint image_size",
            )
            check_equal(
                dataset.focal_length,
                checkpoint["focal_length"],
                "dataset.focal_length",
                "checkpoint focal_length",
            )

            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iteration = checkpoint["iteration"]

    train_loader = DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size, num_workers=4
    )
    train_iter = iter(train_loader)
    checkpoint_interval = 10000
    writer = SummaryWriter(args.save_directory)

    for iteration in range(start_iteration, args.n_iterations):
        # Grab a random batch from the dataset
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Preprocess the data
        images, camera_poses = batch
        images = torch.as_tensor(images, device=device)
        # Discard existing alpha channels and convert to [B, C, H, W]
        images = images[..., :3].permute(0, 3, 1, 2)
        camera_poses = torch.as_tensor(camera_poses, device=device)
        camera_poses = camera_poses[:, :3, :]

        # Run the inference
        prediction = model(camera_poses)
        sampled_coordinates = prediction["sample_coordinates"]
        predicted_pixels = prediction["rgb_map"]

        # Get the ground-truth (target) pixels based on the sampled coordinates
        ys = sampled_coordinates[:, :, 0]
        xs = sampled_coordinates[:, :, 1]
        target_pixels = images[torch.arange(len(images))[:, None], :, ys, xs]

        loss = [mse_loss(pred_pix, target_pixels) for pred_pix in predicted_pixels]
        loss = torch.sum(torch.stack(loss))

        # Update the model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply learning rate scheduling
        initial_lr = optimizer.defaults["lr"]
        decay_rate = args.decay_rate
        lr = initial_lr * decay_rate ** (iteration / args.decay_step)
        optimizer.param_groups[0]["lr"] = lr
        if iteration % 50 == 0:
            mse = loss
            psnr = mse_to_psnr(loss.detach())

            print(
                "[Iteration {}/{}]\tMSE: {:.4f}\tPSNR: {:.4f}\tLR: {:.6f}".format(
                    iteration, args.n_iterations, float(mse), float(psnr), lr
                )
            )
            writer.add_scalar("loss/mse", float(mse), iteration)
            writer.add_scalar("loss/psnr", float(psnr), iteration)

        # Save the model
        if iteration % checkpoint_interval == 0 or iteration == args.n_iterations - 1:
            save_checkpoint(
                args.save_directory,
                model,
                optimizer,
                dataset.image_size,
                dataset.focal_length,
                iteration,
            )


def main():
    args = parse_arguments()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # Create the dataset object
    dataset = Blender(
        args.data_directory,
        mode="train",
        scene=args.scene,
        half_resolution=args.half_resolution,
        white_background=args.white_background,
    )

    # Set up the model
    background_color = None
    if args.white_background:
        background_color = torch.as_tensor([1, 1, 1], dtype=torch.float32)

    model = NeRF(
        dataset.image_size, dataset.focal_length, background_color=background_color
    )

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # Create the directory that stores the saved models
    os.makedirs(args.save_directory, exist_ok=True)

    # Run the training pipeline
    train(dataset, model, optimizer, args)


if __name__ == "__main__":
    main()
