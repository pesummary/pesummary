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
    def _safe_read(config, config_file):
        setattr(config, "path_to_file", config_file)
        try:
            setattr(config, "error", False)
            return config.read(config_file)
        except Exception as e:
            setattr(config, "error", e)
            return None

    def decorator(func):
        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            import configparser

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
