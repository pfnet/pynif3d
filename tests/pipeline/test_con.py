from unittest import TestCase

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from pynif3d.datasets import Shapes3dDataset
from pynif3d.models import ConvolutionalOccupancyNetworksModel, PointNet_LocalPool, UNet
from pynif3d.models.con.unet3d import UNet3D
from pynif3d.pipeline import ConvolutionalOccupancyNetworks
from pynif3d.sampling import FeatureSampler3D


class TestCON(TestCase):
    def test_forward(self):
        # Load dataset
        data_dir = "/tmp/datasets/shapenet"

        train_set = Shapes3dDataset(data_dir, "train")

        # Define dataloaders
        train_dl = DataLoader(train_set)
        train_sample = next(iter(train_dl))

        pipeline = ConvolutionalOccupancyNetworks(
            encoder_fn=PointNet_LocalPool(
                point_feature_channels=32,
                feature_grid_channels=32,
                feature_grid_resolution=64,
                feature_grids=["xy", "xz", "yz"],
                feature_processing_fn=UNet(
                    input_channels=32,
                    output_channels=32,
                    network_depth=4,
                    merge_mode="concat",
                    first_layer_channels=32,
                ),
            ),
            nif_model=ConvolutionalOccupancyNetworksModel(
                block_channels=32,
                linear_channels=32,
            ),
        )

        device = None
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

        pipeline.to(device)

        # Intialize training
        optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)

        initial_loss = 0  # Assign a small number
        is_converged = False
        for it in range(1000):
            data = train_sample

            # This part can be replaced with data converters.
            # Check collate_fn of DataLoader for more details
            input_points = data["in_points"].to(
                device,
            )
            query_points = data["query_points"].to(
                device,
            )

            # Run model
            h = pipeline(input_points, query_points)

            # Get the output and apply loss
            loss = F.binary_cross_entropy_with_logits(
                h,
                data["gt_occupancies"].to(
                    device,
                ),
            )
            print(loss)
            if it == 0:
                initial_loss = loss

            if loss <= initial_loss * 0.10:
                is_converged = True
                break

            # Step gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.assertTrue(
            is_converged,
            "Convolutional Occupancy Networks model did " "not converge in plane mode",
        )

    def test_forward_grid(self):
        # Load dataset
        data_dir = "/tmp/datasets/shapenet"

        train_set = Shapes3dDataset(data_dir, "train")

        # Define dataloaders
        train_dl = DataLoader(train_set)
        train_sample = next(iter(train_dl))

        # Path for grid based CON
        encoder_fn = PointNet_LocalPool(
            point_feature_channels=32,
            feature_grid_resolution=32,
            feature_grid_channels=32,
            feature_grids=["grid"],
            feature_processing_fn=UNet3D(
                output_channels=32,
                input_channels=32,
                feature_maps=32,
                num_levels=3,
                is_segmentation=True,
            ),
        )
        feature_sampler_fn = FeatureSampler3D()

        # Main CON model
        pipeline = ConvolutionalOccupancyNetworks(
            encoder_fn=encoder_fn,
            nif_model=ConvolutionalOccupancyNetworksModel(
                input_channels=3,
                linear_channels=32,
                block_channels=32,
                block_depth=5,
            ),
            feature_sampler_fn=feature_sampler_fn,
        )

        device = None
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

        pipeline.to(device)

        # Intialize training
        optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)

        initial_loss = 0  # Assign a small number
        is_converged = False
        for it in range(1000):
            data = train_sample

            # This part can be replaced with data converters.
            # Check collate_fn of DataLoader for more details
            input_points = data["in_points"].to(
                device,
            )
            query_points = data["query_points"].to(
                device,
            )

            # Run model
            h = pipeline(input_points, query_points)

            # Get the output and apply loss
            loss = F.binary_cross_entropy_with_logits(
                h,
                data["gt_occupancies"].to(
                    device,
                ),
            )
            print(loss)
            if it == 0:
                initial_loss = loss

            if loss <= initial_loss * 0.10:
                is_converged = True
                break

            # Step gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.assertTrue(
            is_converged,
            "Convolutional Occupancy Networks model did " "not converge in grid mode",
        )
