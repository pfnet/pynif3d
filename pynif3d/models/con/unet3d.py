"""
Code from the 3D UNet implementation:
https://github.com/wolny/pytorch-3dunet/
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from pynif3d.common.layer_generator import init_Conv3d, init_GroupNorm
from pynif3d.common.verification import check_in_options
from pynif3d.log.log_funcs import func_logger


class DoubleConv3D_GCR(nn.Module):
    """
    A module consisting of two consecutive convolution block. One block is consisted of
    (GroupNorm3d+Conv3d+ReLU).
    """

    @func_logger
    def __init__(
        self,
        input_channels,
        output_channels,
        is_encoder,
        kernel_size=3,
        num_groups=8,
        padding=1,
    ):
        """

        Args:
            input_channels (int): number of input channels

            output_channels (int): number of output channels

            is_encoder (bool): if True we're in the encoder path,
                               otherwise we're in the decoder

            kernel_size (int): size of the convolving kernel

            num_groups (int): number of groups for the GroupNorm
        """
        super(DoubleConv3D_GCR, self).__init__()

        if is_encoder:
            # Encoder path
            conv1_in_channels = input_channels
            conv1_out_channels = output_channels // 2
            if conv1_out_channels < input_channels:
                conv1_out_channels = input_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, output_channels
        else:
            # Decoder path
            # Decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = input_channels, output_channels
            conv2_in_channels, conv2_out_channels = output_channels, output_channels

        # conv1
        self.group_norm1 = init_GroupNorm(num_groups, conv1_in_channels)
        self.convolution1 = init_Conv3d(
            conv1_in_channels,
            conv1_out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )

        # conv2
        self.group_norm2 = init_GroupNorm(num_groups, conv2_in_channels)
        self.convolution2 = init_Conv3d(
            conv2_in_channels,
            conv2_out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convolution1(self.group_norm1(x)))
        x = self.relu(self.convolution2(self.group_norm2(x)))
        return x


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the max pooling layer.

    """

    @func_logger
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_kernel_size=3,
        pool_kernel_size=(2, 2, 2),
        pool_type="max",
        num_groups=8,
    ):
        """
        Args:
            input_channels (int): number of input channels

            output_channels (int): number of output channels

            conv_kernel_size (int): size of the convolution kernel

            pool_kernel_size (tuple): the size of the window to take a max over

            pool_type (str): pooling layer: 'max' or 'avg'. Set None to disable

            num_groups (int): number of groups for the GroupNorm
        """
        super(Encoder, self).__init__()

        pool_types = {"max": nn.MaxPool3d, "avg": nn.AvgPool3d, None: nn.Identity}
        check_in_options(pool_type, pool_types.keys(), "pool_type")

        self.pooling = pool_types[pool_type](kernel_size=pool_kernel_size)

        self.layer = DoubleConv3D_GCR(
            input_channels,
            output_channels,
            is_encoder=True,
            kernel_size=conv_kernel_size,
            num_groups=num_groups,
        )

    def forward(self, x):
        x = self.pooling(x)
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    followed by a DoubleConv3D_GCR.
    """

    @func_logger
    def __init__(
        self, in_channels, out_channels, kernel_size=3, num_groups=8, mode="nearest"
    ):
        """

        Args:
            in_channels (int): number of input channels

            out_channels (int): number of output channels

            kernel_size (int): size of the convolution kernel

            num_groups (int): number of groups for the GroupNorm
        """
        super(Decoder, self).__init__()
        self.mode = mode
        self.layer = DoubleConv3D_GCR(
            in_channels,
            out_channels,
            is_encoder=False,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )

    def forward(self, encoder_features, x):
        x = F.interpolate(x, size=encoder_features.size()[2:], mode=self.mode)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.layer(x)
        return x


class UNet3D(nn.Module):
    """
    3D Unet class. It applies encoder and decoder as 3D U-Net.

    Usage:

    .. code-block:: python

        input_channels = 16
        output_channels = 32

        model = UNet3D(output_channels, input_channels)
        features = model(x)
    """

    @func_logger
    def __init__(
        self,
        output_channels,
        input_channels,
        final_sigmoid=True,
        feature_maps=64,
        num_groups=8,
        num_levels=4,
        encoder_pool_type="max",
        is_segmentation=False,
        testing=False,
    ):
        """
        Args:
            output_channels (int): number of output segmentation masks; note that the
                value of out_channels might correspond to either different semantic
                classes or to different binary segmentation mask. It's up to the user of
                the class to interpret the out_channels and use the proper loss
                criterion during training (i.e. CrossEntropyLoss (multi-class) or
                BCEWithLogitsLoss (two-class) respectively).
            input_channels (int): The number of input channels
            final_sigmoid (bool): if True apply element-wise nn. Sigmoid after the
                final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if
                nn.BCELoss (two-class) is used to train the model. MUST be False if
                nn.CrossEntropyLoss (multi-class) is used to train the model.
            feature_maps (int, tuple): if int: number of feature maps in the first conv
                layer of the encoder (default: 64); if tuple: number of feature maps at
                each level
            num_groups (int): number of groups for the GroupNorm
            num_levels (int): number of levels in the encoder/decoder path (applied only
                if f_maps is an int)
            is_segmentation (bool): if True (semantic segmentation problem)
                Sigmoid/Softmax normalization is applied after the final convolution;
                if False (regression problem) the normalization layer is skipped at the
                end
            testing (bool): if True (testing mode) the `final_activation` (if present,
                i.e. `is_segmentation=true`) will be applied as the last operation
                during the forward pass; if False the model is in training mode and the
                `final_activation` (even if present) won't be applied; default: False
        """
        super(UNet3D, self).__init__()

        self.testing = testing

        if isinstance(feature_maps, int):
            feature_maps = [feature_maps * 2 ** k for k in range(num_levels)]
        feature_maps.insert(0, input_channels)

        # Encoders
        for idx in range(1, len(feature_maps)):
            pool_type = encoder_pool_type
            if idx <= 1:
                pool_type = None

            encoder = Encoder(
                feature_maps[idx - 1],
                feature_maps[idx],
                pool_type=pool_type,
                num_groups=num_groups,
            )
            setattr(self, "encoder_" + str(idx - 1), encoder)

        # Decoders
        reversed_f_maps = list(reversed(feature_maps))
        for idx in range(0, len(reversed_f_maps) - 2):
            decoder = Decoder(
                reversed_f_maps[idx] + reversed_f_maps[idx + 1],
                reversed_f_maps[idx + 1],
                num_groups=num_groups,
            )
            setattr(self, "decoder_" + str(idx), decoder)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(feature_maps[1], output_channels, 1)

        self.final_activation = None
        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)

        self.num_levels = num_levels

    def forward(self, x):
        # encoder part
        encoders_features = []
        for n_level in range(self.num_levels):
            layer = getattr(self, "encoder_" + str(n_level))
            x = layer(x)
            encoders_features.append(x)
        encoders_features = list(reversed(encoders_features))

        for idx in range(self.num_levels - 1):
            layer = getattr(self, "decoder_" + str(idx))
            x = layer(encoders_features[idx + 1], x)

        x = self.final_conv(x)

        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x
