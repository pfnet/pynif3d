import argparse
import os
import random
import time

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pynif3d.datasets import DTUMVSIDR
from pynif3d.loss import eikonal_loss
from pynif3d.pipeline import IDR

model_filename = "model.pt"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--save-directory", "-sd", required=True)
    parser.add_argument("--batch-size", "-bs", type=int, default=1)
    parser.add_argument("--scan-id", "-s", default=115)
    parser.add_argument("--lr", "-l", type=float, default=1e-4)
    parser.add_argument("--n-epochs", "-e", type=int, default=2000)
    parser.add_argument("--checkpoint-interval", "-ci", type=int, default=10)
    parser.add_argument("--rendering-interval", "-ri", type=int, default=100)
    parser.add_argument("--random-seed", "-rs", type=int, default=None)
    parser.add_argument("--mask-loss-weight", "-mw", type=float, default=100)
    parser.add_argument("--mask-alpha", "-ma", type=float, default=50)
    parser.add_argument("--eikonal-loss-weight", "-ew", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return args


def save_checkpoint(save_directory, model, optimizer, epoch):
    checkpoint_path = os.path.join(save_directory, model_filename)
    data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
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


def train(dataset, model, optimizer, scheduler, args):
    start = time.time()

    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    model.to(device)
    model.train()
    start_epoch = 0

    if args.resume:
        checkpoint = load_checkpoint(args.save_directory)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]

    train_loader = DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size, num_workers=4
    )
    writer = SummaryWriter(args.save_directory)

    alpha = args.mask_alpha
    alpha_factor = 2.0
    alpha_milestones = [250, 500, 750, 1000, 1250]
    for milestone in alpha_milestones:
        if start_epoch > milestone:
            alpha = alpha * alpha_factor

    for epoch in range(start_epoch, args.n_epochs):
        loss_rgb = torch.as_tensor(0.0, device=device)
        loss_mask = torch.as_tensor(0.0, device=device)
        loss_eikonal = torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)

        if epoch in alpha_milestones:
            alpha = alpha * alpha_factor

        for batch in train_loader:
            # Preprocess the data
            images, masks, intrinsics, camera_poses = batch

            images = torch.as_tensor(images, device=device)
            masks = torch.as_tensor(masks, device=device)
            intrinsics = torch.as_tensor(intrinsics, device=device)
            camera_poses = torch.as_tensor(camera_poses, device=device)

            # Run the inference
            prediction = model(images, masks, intrinsics, camera_poses)
            sample_coordinates = prediction["sample_coordinates"]
            ys = sample_coordinates[..., 0]
            xs = sample_coordinates[..., 1]

            # Create the ground-truth data
            predicted_mask = prediction["mask"].reshape(-1)
            predicted_rgb_vals = prediction["rgb_vals"].reshape(-1, 3)
            predicted_z_vals = prediction["z_pred"].reshape(-1)
            predicted_grad_theta = prediction["gradient_theta"].reshape(-1, 3)

            target_masks = masks[torch.arange(len(images))[:, None], :, ys, xs]
            target_masks = target_masks.reshape(-1)
            target_rgb_vals = images[torch.arange(len(images))[:, None], :, ys, xs]
            target_rgb_vals = target_rgb_vals.reshape(-1, 3)

            # Compute the RGB loss
            mask_overlap = torch.logical_and(target_masks, predicted_mask)
            if torch.any(mask_overlap):
                loss_rgb = l1_loss(
                    predicted_rgb_vals[mask_overlap],
                    target_rgb_vals[mask_overlap],
                    reduction="sum",
                )
                loss_rgb = loss_rgb / len(target_masks)
            else:
                loss_rgb = torch.as_tensor(0.0, device=device)

            # Compute the mask loss
            mask_overlap_inv = torch.logical_not(mask_overlap)
            if torch.any(mask_overlap_inv):
                z_pred = -alpha * predicted_z_vals[mask_overlap_inv]
                target_z_pred = target_masks[mask_overlap_inv].type(z_pred.dtype)
                cross_entropy = binary_cross_entropy_with_logits(
                    z_pred, target_z_pred, reduction="sum"
                )
                loss_mask = (1 / alpha) * cross_entropy / len(target_masks)
            else:
                loss_mask = torch.as_tensor(0.0, device=device)

            # Compute the eikonal loss
            if predicted_grad_theta is not None:
                loss_eikonal = eikonal_loss(predicted_grad_theta)
            else:
                loss_eikonal = torch.as_tensor(0.0, device=device)

            loss = (
                args.eikonal_loss_weight * loss_eikonal
                + args.mask_loss_weight * loss_mask
                + loss_rgb
            )

            lr = optimizer.param_groups[0]["lr"]
            print(
                "[Iteration {}/{}]\tLoss: {}\tRGB: {}\tMask: {}\tEikonal: {}\tLR: {}".format(
                    epoch,
                    args.n_epochs,
                    loss.item(),
                    loss_rgb.item(),
                    loss_mask.item(),
                    loss_eikonal.item(),
                    lr,
                )
            )

            # Update the model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Apply learning rate scheduling
        scheduler.step()

        writer.add_scalar("train/loss_rgb", loss_rgb.item(), epoch)
        writer.add_scalar("train/loss_mask", loss_mask.item(), epoch)
        writer.add_scalar("train/loss_eikonal", loss_eikonal.item(), epoch)
        writer.add_scalar("train/loss", loss.item(), epoch)

        # Save the model
        if epoch % args.checkpoint_interval == 0 or epoch == args.n_epochs - 1:
            save_checkpoint(
                args.save_directory,
                model,
                optimizer,
                epoch,
            )
        # Save a rendered result
        if epoch % args.rendering_interval == 0 or epoch == args.n_epochs - 1:
            model.eval()
            image_height, image_width = images.shape[-2:]
            prediction = model(images, masks, intrinsics, camera_poses)
            predicted_rgb_vals = prediction["rgb_vals"]  # (-1, chunk_size, 3)
            predicted_image = predicted_rgb_vals.reshape(image_height, image_width, 3)
            predicted_image = (predicted_image + 1.0) * 127.5  # [-1, 1] -> [0, 255]
            predicted_image = predicted_image.detach().cpu().numpy().astype(np.uint8)

            true_image = images[0].permute(1, 2, 0)  # (1, 3, h, w) -> (h, w, 3)
            true_image = (true_image + 1.0) * 127.5  # [-1, 1] -> [0, 255]
            true_image = true_image.detach().cpu().numpy().astype(np.uint8)

            plot_image = np.concatenate([predicted_image, true_image], axis=0)
            plot_image = plot_image.transpose(2, 0, 1)
            writer.add_image("images", plot_image, epoch)  # (h, w, 3) -> (3, h, w)
            model.train()

    print("Elapsed: ", time.time() - start)


def main():
    args = parse_arguments()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # Create the dataset object
    dataset = DTUMVSIDR(
        args.data_directory,
        mode="train",
        scan_id=args.scan_id,
    )

    # Set up the model
    image_size = dataset.images.shape[-2:]
    model = IDR(image_size)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set up the scheduler
    milestones = [1000, 1500]
    scheduler = MultiStepLR(optimizer, milestones, gamma=0.5)

    # Create the directory that stores the saved models
    os.makedirs(args.save_directory, exist_ok=True)

    # Run the training pipeline
    train(dataset, model, optimizer, scheduler, args)


if __name__ == "__main__":
    main()
