# Licensed under an MIT style license -- see LICENSE.md

import functools
import copy
import numpy as np
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def open_config(index=0):
    """Open a configuration file. The function first looks for a config file
    stored as the keyword argument 'config'. If no kwarg found, one must specify
    the argument index which corresponds to the config file. Default is the 0th
    argument.

    Examples
    --------
    @open_config(index=0)
    def open(config):
        print(list(config['condor'].keys()))

    @open_config(index=2)
    def open(parameters, samples, config):
        print(list(config['condor'].keys()))

    @open_config(index=None)
    def open(parameters, samples, config=config):
        print(list(config['condor'].keys()))
    """
    import configparser

    def _safe_read(config, config_file):
        setattr(config, "path_to_file", config_file)
        try:
            setattr(config, "error", False)
            return config.read(config_file)
        except configparser.MissingSectionHeaderError:
            with open(config_file, "r") as f:
                _config = '[config]\n' + f.read()
            return config.read_string(_config)
        except Exception as e:
            setattr(config, "error", e)
            return None

    def decorator(func):
        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            config = configparser.ConfigParser()
            config.optionxform = str
            if kwargs.get("config", None) is not None:
                _safe_read(config, kwargs.get("config"))
                kwargs.update({"config": config})
            else:
                args = list(copy.deepcopy(args))
                _safe_read(config, args[index])
                args[index] = config
            return func(*args, **kwargs)
        return wrapper_function
    return decorator


def bound_samples(minimum=-np.inf, maximum=np.inf, logger_level="debug"):
    """Bound samples to be within a specified range. If any samples lie
    outside of this range, we set these invalid samples to equal the value at
    the boundary.

    Parameters
    ----------
    minimum: float
        lower boundary. Default -np.inf
    maximum: float
        upper boundary. Default np.inf
    logger_level: str
        level to use for any logger messages

    Examples
    --------
    @bound_samples(minimum=-1., maximum=1., logger_level="info")
    def random_samples():
        return np.random.uniform(-2, 2, 10000)

    >>> random_samples()
    PESummary INFO    : 2576/10000 (25.76%) samples lie outside of the specified
    range for the function random_samples (< -1.0). Truncating these samples to
    -1.0.
    PESummary INFO    : 2495/10000 (24.95%) samples lie outside of the specified
    range for the function random_samples (> 1.0). Truncating these samples to
    1.0.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            value = np.atleast_1d(func(*args, **kwargs))
            _minimum_inds = np.argwhere(value < minimum)
            _maximum_inds = np.argwhere(value > maximum)
            zipped = zip([_minimum_inds, _maximum_inds], [minimum, maximum])
            for invalid, bound in zipped:
                if len(invalid):
                    getattr(logger, logger_level)(
                        "{}/{} ({}%) samples lie outside of the specified "
                        "range for the function {} ({} {}). Truncating these "
                        "samples to {}.".format(
                            len(invalid), len(value),
                            np.round(len(invalid) / len(value) * 100, 2),
                            func.__name__, "<" if bound == minimum else ">",
                            bound, bound
                        )
                    )
                    value[invalid] = bound
            return value
        return wrapper_function
    return decorator


def no_latex_plot(func):
    """Turn off latex plotting for a given function
    """
    @functools.wraps(func)
    def wrapper_function(*args, **kwargs):
        from matplotlib import rcParams

        original_tex = rcParams["text.usetex"]
        rcParams["text.usetex"] = False
        value = func(*args, **kwargs)
        rcParams["text.usetex"] = original_tex
        return value
    return wrapper_function


def try_latex_plot(func):
    """Try to make a latex plot, if RuntimeError raised, turn latex off
    and try again
    """
    @functools.wraps(func)
    def wrapper_function(*args, **kwargs):
        from matplotlib import rcParams

        original_tex = rcParams["text.usetex"]
        try:
            value = func(*args, **kwargs)
        except RuntimeError:
            logger.debug("Unable to use latex. Turning off for this plot")
            rcParams["text.usetex"] = False
            value = func(*args, **kwargs)
        rcParams["text.usetex"] = original_tex
        return value
    return wrapper_function


def tmp_directory(func):
    """Make a temporary directory run the function from within that
    directory. Change directory back again after the function has finished
    running
    """
    @functools.wraps(func)
    def wrapper_function(*args, **kwargs):
        import tempfile
        import os

        current_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir="./") as path:
            os.chdir(path)
            value = func(*args, **kwargs)
            os.chdir(current_dir)
        return value
    return wrapper_function


def array_input(ignore_args=None, ignore_kwargs=None, force_return_array=False):
    """Convert the input into an np.ndarray and return either a float or a
    np.ndarray depending on what was input.

    Examples
    --------
    >>> @array_input
    >>> def total_mass(mass_1, mass_2):
    ...    total_mass = mass_1 + mass_2
    ...    return total_mass
    ...
    >>> print(total_mass(30, 10))
    40.0
    >>> print(total_mass([30, 3], [10, 1]))
    [40 4]
    """
    def _array_input(func):
        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            new_args = list(copy.deepcopy(args))
            new_kwargs = kwargs.copy()
            return_float = False
            for num, arg in enumerate(args):
                if ignore_args is not None and num in ignore_args:
                    pass
                elif isinstance(arg, (float, int)):
                    new_args[num] = np.array([arg])
                    return_float = True
                elif isinstance(arg, (list, np.ndarray)):
                    new_args[num] = np.array(arg)
                else:
                    pass
            for key, item in kwargs.items():
                if ignore_kwargs is not None and key in ignore_kwargs:
                    pass
                elif isinstance(item, (float, int)):
                    new_kwargs[key] = np.array([item])
                elif isinstance(item, (list, np.ndarray)):
                    new_kwargs[key] = np.array(item)
            output = func(*new_args, **new_kwargs)
            if isinstance(output, dict):
                return output
            value = np.array(output)
            if return_float and not force_return_array:
                new_value = copy.deepcopy(value)
                if len(new_value) > 1:
                    new_value = np.array([arg[0] for arg in value])
                elif new_value.ndim == 2:
                    new_value = new_value[0]
                else:
                    new_value = float(new_value)
                return new_value
            return value
        return wrapper_function
    return _array_input


def docstring_subfunction(*args):
    """Edit the docstring of a function to show the docstrings of subfunctions
    """
    def wrapper_function(func):
        import importlib

        original_docstring = func.__doc__
        if isinstance(args[0], list):
            original_docstring += "\n\nSubfunctions:\n"
            for subfunction in args[0]:
                _subfunction = subfunction.split(".")
                module = ".".join(_subfunction[:-1])
                function = _subfunction[-1]
                module = importlib.import_module(module)
                original_docstring += "\n{}{}".format(
                    subfunction + "\n" + "-" * len(subfunction) + "\n",
                    getattr(module, function).__doc__
                )
        else:
            _subfunction = args[0].split(".")
            module = ".".join(_subfunction[:-1])
            function = _subfunction[-1]
            module = importlib.import_module(module)
            original_docstring += (
                "\n\nSubfunctions:\n\n{}{}".format(
                    args[0] + "\n" + "-" * len(args[0]) + "\n",
                    getattr(module, function).__doc__
                )
            )
        func.__doc__ = original_docstring
        return func
    return wrapper_function


def set_docstring(docstring):
    def wrapper_function(func):
        func.__doc__ = docstring
        return func
    return wrapper_function


def deprecation(warning):
    def decorator(func):
        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            import warnings

            warnings.warn(warning)
            return func(*args, **kwargs)
        return wrapper_function
    return decorator
