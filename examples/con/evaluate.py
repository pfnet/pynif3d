import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pynif3d.common.verification import check_not_none, check_path_exists
from pynif3d.datasets import Shapes3dDataset
from pynif3d.models import ConvolutionalOccupancyNetworksModel, PointNet_LocalPool, UNet
from pynif3d.models.con.unet3d import UNet3D
from pynif3d.pipeline import ConvolutionalOccupancyNetworks
from pynif3d.sampling import FeatureSampler2D, FeatureSampler3D


def parse_arguments():
    mode_choices = ["single", "multi", "grid"]
    dataset_choices = ["shapenet", "synthetic_room"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-dd", required=True)
    parser.add_argument("--model-file", "-m", required=True)
    parser.add_argument("--threshold", "-t", type=float, default=0.2)
    parser.add_argument("--mode", "-mo", default="grid", choices=mode_choices)
    parser.add_argument("--dataset", "-ds", default="shapenet", choices=dataset_choices)
    parser.add_argument("--feat_channels", "-fs", type=int, default=32)
    parser.add_argument("--feat_dims", "-fd", type=int, default=64)
    parser.add_argument("--n_levels", "-nl", type=int, default=4)
    parser.add_argument("--output-file", "-o", default="./evaluation.json")
    args = parser.parse_args()
    return args


def compute_iou(occ1, occ2):
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


def evaluate(dataset, model, args):
    """
    Computes the average IoU on the test dataset.

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
    average_iou = 0

    for batch in tqdm(test_loader):
        # Preprocess the data
        in_points = batch["in_points"].to(device)
        all_query_points = batch["all_query_points"].to(device)
        all_gt_occupancies = batch["all_gt_occupancies"]

        # Run the inference
        with torch.no_grad():
            prediction = model(in_points, all_query_points)

        pred = (prediction > args.threshold).type(all_gt_occupancies.dtype)

        # Calculate IoU
        pred_np = pred.cpu().numpy()
        gt_np = all_gt_occupancies.cpu().numpy()
        iou = compute_iou(pred_np, gt_np)
        average_iou += iou

    output = {
        "average_iou": float(average_iou / len(dataset)),
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
    dataset = None
    if args.dataset == "shapenet":
        dataset = Shapes3dDataset(args.data_directory, mode="val")
    elif args.dataset == "synthetic_room":
        dataset = Shapes3dDataset(
            args.data_directory,
            mode="val",
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

    model.load_state_dict(state_dict)

    # Run the training pipeline
    evaluate(dataset, model, args)


if __name__ == "__main__":
    main()
