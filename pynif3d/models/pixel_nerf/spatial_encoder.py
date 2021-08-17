import torch
import torchvision
from torch.nn.functional import interpolate

from pynif3d.common.verification import check_pos_int
from pynif3d.log.log_funcs import func_logger


class SpatialEncoder(torch.nn.Module):
    @func_logger
    def __init__(
        self,
        backbone_fn: torch.nn.Module = None,
        backbone_fn_kwargs: dict = None,
        n_layers: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()

        check_pos_int(n_layers, "n_layers")
        self.n_layers = n_layers

        if backbone_fn is None:
            if backbone_fn_kwargs is None:
                backbone_fn_kwargs = {
                    "pretrained": pretrained,
                    "norm_layer": lambda h: torch.nn.BatchNorm2d(
                        h, affine=True, track_running_stats=True
                    ),
                }
            backbone_fn = torchvision.models.resnet34(backbone_fn_kwargs)
            backbone_fn.fc = torch.nn.Sequential()
            backbone_fn.avgpool = torch.nn.Sequential()

        self.backbone_fn = backbone_fn

    def forward(self, images, **kwargs):
        upsample_interpolation = kwargs.get("upsample_interpolation", "bilinear")
        upsample_align_corners = kwargs.get("upsample_align_corners", True)

        h = images

        if issubclass(type(self.backbone_fn), torchvision.models.ResNet):
            resnet_use_first_pool = kwargs.get("resnet_use_first_pool", True)

            h = self.backbone_fn.conv1(h)
            h = self.backbone_fn.bn1(h)
            h = self.backbone_fn.relu(h)

            latent = [h]
            upsample_size = h.shape[-2:]

            for index in range(2, self.n_layers + 1):
                if index == 2 and resnet_use_first_pool:
                    h = self.backbone_fn.maxpool(h)
                layer = getattr(self.backbone_fn, "layer" + str(index - 1))
                h = layer(h)

                upsampled_h = interpolate(
                    h,
                    upsample_size,
                    mode=upsample_interpolation,
                    align_corners=upsample_align_corners,
                )
                latent.append(upsampled_h)

            latent = torch.cat(latent, dim=1)
        else:
            latent = self.backbone_fn(h)

        return latent
