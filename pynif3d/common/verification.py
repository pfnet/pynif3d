import os
from collections import Iterable

from pynif3d import logger


def is_image_file(file_path):
    choices = [".jpg", ".png", ".jpeg", ".bmp"]
    filename = os.path.basename(file_path).lower()
    extension = os.path.splitext(filename)[1]
    return extension in choices


def check_callable(fn, property_name, fn_name):
    if not hasattr(fn, property_name):
        raise AttributeNotExistingException(fn_name, property_name)

    if not callable(getattr(fn, property_name)):
        raise NotCallableException(fn_name, property_name)


def check_axis(variable, axis, variable_name):
    if not hasattr(variable, "shape"):
        raise AttributeNotExistingException(variable_name, "shape")

    if len(variable.shape) <= axis:
        raise WrongAxisException(variable_name, axis)


def check_devices_match(variable1, variable2, variable1_name, variable2_name):
    if not hasattr(variable1, "device"):
        raise AttributeNotExistingException(variable1_name, "device")

    if not hasattr(variable2, "device"):
        raise AttributeNotExistingException(variable2_name, "device")

    if variable1.device != variable2.device:
        raise DevicesNotMatchingException(variable1_name, variable2_name)


def check_equal(variable1, variable2, variable1_name, variable2_name):
    if variable1 != variable2:
        raise NotEqualException(variable1_name, variable2_name)


def check_shape(variable, shape, variable_name):
    if not hasattr(variable, "shape"):
        raise AttributeNotExistingException(variable_name, "shape")

    if variable.shape != shape:
        raise WrongShapeException(variable_name, shape)


def check_lengths_match(variable1, variable2, variable1_name, variable2_name):
    if len(variable1) != len(variable2):
        raise ShapesNotMatchingException(variable1_name, variable2_name)


def check_shapes_match(variable1, variable2, variable1_name, variable2_name):
    if not hasattr(variable1, "shape"):
        raise AttributeNotExistingException(variable1_name, "shape")

    if not hasattr(variable2, "shape"):
        raise AttributeNotExistingException(variable2_name, "shape")

    if variable1.shape != variable2.shape:
        raise ShapesNotMatchingException(variable1_name, variable2_name)


def check_true(variable, variable_name):
    check_bool(variable, variable_name)
    if variable is False:
        raise ValueError(variable_name + " should be True, but it evaluated to False.")


def check_not_none(variable, variable_name):
    if variable is None:
        raise NoneVariableException(variable_name)


def check_bool(variable, variable_name):
    if not isinstance(variable, bool):
        raise NotBooleanException(variable_name)


def check_pos_int(variable, variable_name):
    if variable <= 0 or not isinstance(variable, int):
        raise NotPositiveIntegerException(variable_name)


def check_iterable(variable, variable_name):
    if not isinstance(variable, Iterable):
        raise NotIterableException(variable_name)


def check_in_options(variable, options, variable_name):
    check_iterable(options, "options")

    if variable not in options:
        raise NotInOptionsException(variable_name, options)


def check_path_exists(path):
    if not os.path.exists(path):
        raise PathNotFoundException(path)


class ExceptionBase(Exception):
    def __init__(self):
        self.message = ""

    def __str__(self):
        logger.error(self.message)
        return self.message


class NotIterableException(ExceptionBase):
    def __init__(self, variable_name):
        self.message = "The variable `" + variable_name + "` is not iterable."


class NotBooleanException(ExceptionBase):
    def __init__(self, variable_name):
        self.message = "`" + variable_name + "` has to be a boolean value."


class NotPositiveIntegerException(ExceptionBase):
    def __init__(self, variable_name):
        self.message = "`" + variable_name + "` has to be positive integer."


class NotInOptionsException(ExceptionBase):
    def __init__(self, variable_name, options):
        self.message = (
            "`"
            + variable_name
            + "` has to be defined one of the options: "
            + ",".join(options)
        )


class NoneVariableException(ExceptionBase):
    def __init__(self, variable_name):
        self.message = "`" + variable_name + "` cannot be None"


class ShapesNotMatchingException(ExceptionBase):
    def __init__(self, variable1_name, variable2_name):
        self.message = (
            "The shapes for variable`"
            + variable1_name
            + "` and `"
            + variable2_name
            + "` don't match."
        )


class AttributeNotExistingException(ExceptionBase):
    def __init__(self, variable_name, attribute_name):
        self.message = (
            "The variable `"
            + variable_name
            + "` doesn't have attribute `"
            + attribute_name
            + "`. "
        )


class NotEqualException(ExceptionBase):
    def __init__(self, variable1_name, variable2_name):
        self.message = (
            "The variables `"
            + variable1_name
            + "` and `"
            + variable2_name
            + "`"
            + " are not equal."
        )


class WrongShapeException(ExceptionBase):
    def __init__(self, variable_name, shape):
        self.message = (
            "The variable `" + variable_name + "` shall have the shape `" + shape + "`."
        )


class DevicesNotMatchingException(ExceptionBase):
    def __init__(self, variable1_name, variable2_name):
        self.message = (
            "The devices for variable`"
            + variable1_name
            + "` and `"
            + variable2_name
            + "` don't match."
        )


class WrongAxisException(ExceptionBase):
    def __init__(self, variable_name, axis):
        self.message = (
            "The axis`"
            + str(axis)
            + "` is incompatible with the shape of`"
            + variable_name
            + "`."
        )


class PathNotFoundException(ExceptionBase):
    def __init__(self, path):
        self.message = "The path`" + str(path) + "` could not be found."


class NotCallableException(ExceptionBase):
    def __init__(self, fn_name, property_name):
        self.message = (
            "The function `"
            + fn_name
            + "` doesn't have property `"
            + property_name
            + "` or the property is not callable."
        )
