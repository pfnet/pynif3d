import inspect

import torch.nn as nn

from pynif3d import logger
from pynif3d.common.verification import check_devices_match, check_shapes_match


def get_function_kwargs(fn, kwargs):
    # Get function signature
    fn_signature = inspect.getfullargspec(fn)

    # Iterate args and bind them to function signature
    fn_args = {}
    for key, value in kwargs.items():
        if key in fn_signature.args[1:]:
            fn_args[key] = value
    return fn_args


def initialize_layer(layer_fn, **kwargs):
    # Get function signature
    fn_args = get_function_kwargs(layer_fn, kwargs)

    # Create layer
    layer = layer_fn(**fn_args)

    # Check if predefined weights and initialization function is defined
    initial_weights = kwargs.get("initial_weights", None)
    w_init_fn = kwargs.get("w_init_fn", None)
    w_init_fn_args = kwargs.get("w_init_fn_args", {})

    bias = kwargs.get("bias", True)
    initial_bias = kwargs.get("initial_bias", None)
    b_init_fn = kwargs.get("b_init_fn", None)
    b_init_fn_args = kwargs.get("b_init_fn_args", {})

    if initial_weights is not None and w_init_fn is not None:
        msg = (
            "`initial_weights` and `w_init_fn` cannot be defined at the"
            + " same time. Use either predefined weights or initialization function. "
        )
        logger.error(msg)
        raise ValueError(msg)

    if initial_bias is not None and b_init_fn is not None:
        msg = (
            "`initial_bias` and `b_init_fn` cannot be defined at the"
            + " same time. Use either predefined weights or initialization function. "
        )
        logger.error(msg)
        raise ValueError(msg)

    if initial_weights is not None:
        check_shapes_match(
            layer.weight, initial_weights, "layer.weights", "initial_weights"
        )
        check_devices_match(
            layer.weight, initial_weights, "layer.weights", "initial_weights"
        )
        layer.weight = initial_weights

    if initial_bias is not None and bias:
        check_shapes_match(layer.bias, initial_bias, "layer.bias", "initial_bias")
        check_devices_match(layer.bias, initial_bias, "layer.bias", "initial_bias")
        layer.bias = initial_bias

    if w_init_fn is not None:
        w_init_fn(layer.weight, *w_init_fn_args)
    if b_init_fn is not None and bias:
        b_init_fn(layer.bias, *b_init_fn_args)

    return layer


def init_Linear(size_in, size_out, **kwargs):
    kwargs["in_features"] = size_in
    kwargs["out_features"] = size_out

    fc_layer = initialize_layer(nn.Linear, **kwargs)

    return fc_layer


def init_Conv2d(size_in, size_out, kernel_size, **kwargs):
    kwargs["in_channels"] = size_in
    kwargs["out_channels"] = size_out
    kwargs["kernel_size"] = kernel_size

    conv_layer = initialize_layer(nn.Conv2d, **kwargs)

    return conv_layer


def init_ConvTranspose2d(size_in, size_out, kernel_size, **kwargs):
    kwargs["in_channels"] = size_in
    kwargs["out_channels"] = size_out
    kwargs["kernel_size"] = kernel_size

    conv_layer = initialize_layer(nn.ConvTranspose2d, **kwargs)

    return conv_layer


def init_Conv3d(size_in, size_out, kernel_size, **kwargs):
    kwargs["in_channels"] = size_in
    kwargs["out_channels"] = size_out
    kwargs["kernel_size"] = kernel_size

    conv_layer = initialize_layer(nn.Conv3d, **kwargs)

    return conv_layer


def init_GroupNorm(num_groups, num_channels, **kwargs):
    kwargs["num_groups"] = num_groups
    kwargs["num_channels"] = num_channels

    norm_layer = initialize_layer(nn.GroupNorm, **kwargs)

    return norm_layer
