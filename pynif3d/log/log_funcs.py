import functools
import inspect
import logging
import sys
import traceback

from pynif3d import logger
from pynif3d.common.layer_generator import get_function_kwargs


def func_logger(_func=None):
    def log_decorator_info(func):
        @functools.wraps(func)
        def log_decorator_wrapper(self, *args, **kwargs):

            try:
                # Only execute if it is debug level or lower
                # For other log levels, simply execute the function
                if logger.level > logging.DEBUG:
                    return func(self, *args, **kwargs)

                # Get default function arguments
                # Check if there is any parameter passed to function
                # And report all the actively used parameters
                fn_sign = inspect.signature(func).parameters
                fn_override_kwargs = get_function_kwargs(func, kwargs)
                args_idx = 0
                for k, v in fn_sign.items():
                    if k == "self":
                        continue

                    # Assign args first
                    if args_idx < len(args):
                        fn_override_kwargs[k] = args[args_idx]
                        args_idx += 1
                    # If args is over, check for kwargs
                    elif k not in fn_override_kwargs:
                        fn_override_kwargs[k] = v.default

                logger.debug("\n===================================")
                logger.debug(traceback.print_stack())
                logger.debug("Executing function" + str(func))
                args_str = "Function arguments are:"
                for k, v in fn_override_kwargs.items():
                    args_str += "\n\t" + k + ": " + str(v) + ""
                logger.debug(args_str)

                """ log return value from the function """
                value = func(self, *args, **kwargs)
                logger.debug(
                    "Return value from function " + str(func) + ": " + str(value)
                )
                logger.debug("===================================")
            except Exception:
                """log exception if occurs in function"""
                logger.error(f"Exception: {str(sys.exc_info()[1])}")
                raise
            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    else:
        return log_decorator_info(_func)
