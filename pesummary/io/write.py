# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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

import importlib


def write(*args, package="gw", file_format="dat", **kwargs):
    """Read in a results file.

    Parameters
    ----------
    args: tuple
        all args are passed to write function
    package: str
        the package you wish to use
    file_format: str
        the file format you wish to use. Default None. If None, the read
        function loops through all possible options
    kwargs: dict
        all kwargs passed to write function
    """
    def _import(package, file_format):
        """Import format module with importlib
        """
        return importlib.import_module(
            "pesummary.{}.file.formats.{}".format(package, file_format)
        )

    def _write(module, file_format, args, kwargs):
        """Execute the write method
        """
        return getattr(module, "write_{}".format(file_format))(*args, **kwargs)

    if file_format == "h5":
        file_format = "hdf5"

    try:
        module = _import(package, file_format)
        return _write(module, file_format, args, kwargs)
    except (ImportError, AttributeError, ModuleNotFoundError):
        module = _import("core", file_format)
        return _write(module, file_format, args, kwargs)
