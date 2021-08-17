import argparse
import os
import random

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pynif3d.common.verification import check_not_none
from pynif3d.datasets import Shapes3dDataset
from pynif3d.models import ConvolutionalOccupancyNetworksModel, PointNet_LocalPool, UNet
from pynif3d.models.con.unet3d import UNet3D
from pynif3d.pipeline.con import ConvolutionalOccupancyNetworks
from pynif3d.sampling import FeatureSampler2D, FeatureSampler3D

model_filename = "model.pt"


def parse_arguments():
    mode_choices = ["single", "multi", "grid"]
    dataset_choices = ["shapenet", "synthetic_room"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--save-directory", "-sd", default="./saved_models")
    parser.add_argument("--lr", "-l", type=float, default=1e-4)
    parser.add_argument("--mode", "-m", default="grid", choices=mode_choices)
    parser.add_argument("--batch-size", "-bs", type=int, default=32)
    parser.add_argument("--n_iterations", "-i", type=int, default=300000)
    parser.add_argument("--random-seed", "-rs", type=int, default=None)
    parser.add_argument("--dataset", "-ds", default="shapenet", choices=dataset_choices)
    parser.add_argument("--feat-channels", "-fs", type=int, default=32)
    parser.add_argument("--feat-dims", "-fd", type=int, default=64)
    parser.add_argument("--n-levels", "-nl", type=int, default=4)
    parser.add_argument("--resume", "-c", action="store_true")
    args = parser.parse_args()
    return args


def save_checkpoint(save_directory, model, optimizer, iteration):
    checkpoint_path = os.path.join(save_directory, "model.pt")
    data = {
        "iteration": iteration,
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
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iteration = checkpoint["iteration"]

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
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

        has_items = bool(batch)
        if not has_items:
            continue

        input_points = torch.as_tensor(batch["in_points"], device=device)
        query_points = torch.as_tensor(batch["query_points"], device=device)
        target = torch.as_tensor(batch["gt_occupancies"], device=device)

        prediction = model(input_points, query_points)
        loss = binary_cross_entropy_with_logits(prediction, target, reduction="none")
        loss = loss.sum(-1).mean()

        lr = optimizer.param_groups[0]["lr"]
        if iteration % 50 == 0:
            print(
                "[Iteration {}/{}]\tBCE: {:.4f}\tLR: {:.6f}".format(
                    iteration, args.n_iterations, float(loss), lr
                )
            )
            writer.add_scalar("loss/bce", float(loss), iteration)

        # Update the model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the model
        if iteration % checkpoint_interval == 0 or iteration == args.n_iterations - 1:
            save_checkpoint(
                args.save_directory,
                model,
                optimizer,
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
    dataset = None
    if args.dataset == "shapenet":
        dataset = Shapes3dDataset(args.data_directory, mode="train")
    elif args.dataset == "synthetic_room":
        dataset = Shapes3dDataset(
            args.data_directory,
            mode="train",
            points_filename="points_iou",
            pointcloud_filename="pointcloud",
        )

    check_not_none(dataset, "dataset")

    # Set up the model
    if args.mode == "single":
        planes = ["xz"]
    elif args.mode == "multi":
        planes = ["xy", "yz", "xz"]
    else:
        planes = ["grid"]

    if args.mode == "grid":
        # Path for grid based CON
        encoder_fn = PointNet_LocalPool(
            point_feature_channels=args.feat_channels,
            feature_grid_resolution=args.feat_dims,
            feature_grid_channels=args.feat_channels,
            feature_grids=planes,
            feature_processing_fn=UNet3D(
                output_channels=args.feat_channels,
                input_channels=args.feat_channels,
                feature_maps=args.feat_channels,
                num_levels=args.n_levels,
                is_segmentation=True,
            ),
        )
        feature_sampler_fn = FeatureSampler3D()
    else:
        # Path for plane based CON
        encoder_fn = PointNet_LocalPool(
            point_feature_channels=args.feat_channels,
            feature_grid_resolution=args.feat_dims,
            feature_grid_channels=args.feat_channels,
            feature_grids=planes,
            feature_processing_fn=UNet(
                output_channels=args.feat_channels,
                input_channels=args.feat_channels,
                network_depth=args.n_levels,
                first_layer_channels=args.feat_channels,
            ),
        )
        feature_sampler_fn = FeatureSampler2D()

    # Main CON model
    model = ConvolutionalOccupancyNetworks(
        encoder_fn=encoder_fn,
        nif_model=ConvolutionalOccupancyNetworksModel(
            input_channels=3,
            linear_channels=args.feat_channels,
            block_channels=args.feat_channels,
            block_depth=5,
        ),
        feature_sampler_fn=feature_sampler_fn,
    )

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # Create the directory that stores the saved models
    os.makedirs(args.save_directory, exist_ok=True)

    # Run the training pipeline
    train(dataset, model, optimizer, args)


if __name__ == "__main__":
    main()
