# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import functools
import copy
import numpy as np


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


def array_input(func):
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
    @functools.wraps(func)
    def wrapper_function(*args, **kwargs):
        new_args = list(copy.deepcopy(args))
        new_kwargs = kwargs.copy()
        return_float = False
        for num, arg in enumerate(args):
            if isinstance(arg, (float, int)):
                new_args[num] = np.array([arg])
                return_float = True
            elif isinstance(arg, (list, np.ndarray)):
                new_args[num] = np.array(arg)
            else:
                pass
        for key, item in kwargs.items():
            if isinstance(item, (float, int)):
                new_kwargs[key] = np.array([item])
            elif isinstance(item, (list, np.ndarray)):
                new_kwargs[key] = np.array(item)
        value = np.array(func(*new_args, **new_kwargs))
        if return_float:
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
