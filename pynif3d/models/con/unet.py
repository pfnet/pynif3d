import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from pynif3d.common import init_Conv2d, init_ConvTranspose2d
from pynif3d.common.verification import check_in_options
from pynif3d.log.log_funcs import func_logger


class UNet(nn.Module):
    """
    UNet class for Convolutional Occupancy Networks (CON), as described in:
    https://arxiv.org/abs/2003.04618

    .. note::

        This implementation is based on the original one, which can be found at:
        https://github.com/autonomousvision/convolutional_occupancy_networks

    The U-Net is a convolutional encoder-decoder neural network. Contextual spatial
    information (from the decoding, expansive pathway) related to the input tensor is
    merged with information representing the localization of details (from the encoding,
    compressive pathway).

    Usage:

    .. code-block:: python

        input_channels = 3
        output_channels = 1

        model = UNet(output_channels, input_channels)
        h = model(h)
    """

    @func_logger
    def __init__(
        self,
        output_channels,
        input_channels=3,
        network_depth=5,
        first_layer_channels=64,
        upconv_mode="transpose",
        merge_mode="concat",
        **kwargs
    ):
        """
        Args:
            output_channels (int): The number of channels for the output tensor.
            input_channels (int): The number of channels in the input tensor. Default is
                3.
            network_depth (int): The number of convolution blocks. Default is 5.
            first_layer_channels (int): The number of convolutional filters for the
                first convolution. For each depth level, the channel size is multiplied
                by 2.
            up_mode (str): The type of upconvolution ("transpose", "upsample").
            merge_mode (str): The type of the merge operation ("concat", "add").

            kwargs (dict):
                - **w_init_fn**: Callback function for parameter initialization. Default
                  is `xavier_normal`.
                - **w_init_fn_args**: The arguments to pass to the `w_init_fn` function.
                  Optional.
                - **b_init_fn**: Callback function for bias initialization. Default is
                  constant 0.
                - **b_init_fn_args**: The arguments to pass the `b_init_fn` function.
                  Optional.
        """
        super(UNet, self).__init__()

        self.upconv_mode = upconv_mode
        check_in_options(upconv_mode, ("transpose", "upsample"), "upconv_mode")
        check_in_options(merge_mode, ("concat", "add"), "merge_mode")

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if upconv_mode == "upsample" and merge_mode == "add":
            raise ValueError(
                'upconv_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.network_depth = network_depth

        # Define default initializers
        if "w_init_fn" not in kwargs.keys():
            kwargs["w_init_fn"] = init.xavier_normal_
        if "b_init_fn" not in kwargs.keys():
            kwargs["b_init_fn"] = init.constant_
            kwargs["b_init_fn_args"] = [0]

        # create the encoder
        outs = input_channels
        for i in range(network_depth):
            # Define the input, output layers and pooling flag
            ins = outs
            outs = first_layer_channels * (2 ** i)
            pooling = i < network_depth - 1

            setattr(self, "down_conv_" + str(i), DownConv(ins, outs, pooling=pooling))

        # create the decoder
        for i in range(network_depth - 1):
            ins = outs
            outs = ins // 2

            setattr(
                self,
                "up_conv_" + str(i),
                UpConv(ins, outs, up_mode=upconv_mode, merge_mode=merge_mode),
            )

        self.conv_final = init_Conv2d(
            outs,
            output_channels,
            kernel_size=1,
            stride=1,
            w_init_fn=init.xavier_normal_,
            b_init_fn=init.constant_,
            b_init_fn_args=[0],
        )

    def forward(self, h):
        encoder_outs = []

        # Execute encode
        for i in range(self.network_depth):
            layer = getattr(self, "down_conv_" + str(i))
            h, before_pool = layer(h)
            encoder_outs.append(before_pool)

        # Execute decode
        for i in range(self.network_depth - 1):
            layer = getattr(self, "up_conv_" + str(i))
            before_pool = encoder_outs[-(i + 2)]
            h = layer(before_pool, h)

        h = self.conv_final(h)
        return h


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool. A ReLU activation
    follows each convolution.
    """

    def __init__(self, in_channel, out_channel, pooling=True):
        super(DownConv, self).__init__()

        self.conv1 = init_Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = init_Conv2d(out_channel, out_channel, 3, padding=1)

        self.pool = None
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, h):
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        before_pool = h
        if self.pool:
            h = self.pool(h)
        return h, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution. A ReLU activation
    follows each convolution.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        merge_mode="concat",
        up_mode="transpose",
    ):
        super(UpConv, self).__init__()

        self.merge_mode = merge_mode
        self.up_mode = up_mode

        if up_mode == "transpose":
            self.upconv = init_ConvTranspose2d(
                in_channel,
                out_channel,
                2,
                stride=2,
            )
        else:
            self.sampler = nn.Upsample(mode="bilinear", scale_factor=2)
            self.upconv = init_Conv2d(in_channel, out_channel, 1)

        tmp_channel = out_channel
        if self.merge_mode == "concat":
            tmp_channel = 2 * out_channel
        self.conv1 = init_Conv2d(tmp_channel, out_channel, 3, padding=1)
        self.conv2 = init_Conv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, from_down, from_up):
        """
        Args:
            from_down (torch.Tensor): Tensor from the encoder pathway.
            from_up (torch.Tensor): Upconv tensor from the decoder pathway.
        """

        if self.up_mode == "transpose":
            from_up = self.upconv(from_up)
        else:
            from_up = self.sampler(from_up)
            from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            h = torch.cat((from_up, from_down), 1)
        else:
            h = from_up + from_down
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        return h
