import torch
import torch.nn.functional as F

from pynif3d.common.torch_helper import normalize_coordinate
from pynif3d.log.log_funcs import func_logger


class FeatureSampler2D(torch.nn.Module):
    @func_logger
    def __init__(self, sample_mode="bilinear", padding=0.1):
        super().__init__()

        self.padding = padding
        self.sample_mode = sample_mode

    def forward(self, points, plane_features):
        res_features = []

        for plane, features in plane_features.items():
            points_n = normalize_coordinate(
                points.clone(), plane=plane, padding=self.padding
            )

            points_n = points_n[:, :, None]
            vgrid = 2.0 * points_n - 1.0
            res_features.append(
                F.grid_sample(
                    features,
                    vgrid,
                    padding_mode="border",
                    align_corners=True,
                    mode=self.sample_mode,
                ).squeeze(-1)
            )

        res_features = torch.sum(torch.stack(res_features), 0).transpose(1, 2)

        return res_features


class FeatureSampler3D(torch.nn.Module):
    @func_logger
    def __init__(self, sample_mode="bilinear", padding=0.1):
        super().__init__()

        self.padding = padding
        self.sample_mode = sample_mode

    def forward(self, points, plane_features):
        res_features = []

        for plane, features in plane_features.items():
            points_n = normalize_coordinate(
                points.clone(), plane=plane, padding=self.padding
            )

            points_n = points_n[:, :, None, None]
            vgrid = 2.0 * points_n - 1.0
            res_features.append(
                F.grid_sample(
                    features,
                    vgrid,
                    padding_mode="border",
                    align_corners=True,
                    mode=self.sample_mode,
                )
                .squeeze(-1)
                .squeeze(-1)
            )

        res_features = torch.sum(torch.stack(res_features), 0).transpose(1, 2)

        return res_features
