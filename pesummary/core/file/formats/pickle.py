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

from pesummary import conf
import pickle
from pesummary.utils.utils import check_filename


def read_pickle(path):
    """Read a pickle file and return the object

    Parameters
    ----------
    path: str
        path to the pickle file you wish to read in
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(
    *args, outdir="./", label=None, filename=None, overwrite=False,
    default_filename="pesummary_{}.pickle", **kwargs
):
    """Write an object to a pickle file

    Parameters
    ----------
    *args: tuple
        all args passed to pickle.dump
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    """
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite
    )
    with open(filename, "wb") as f:
        pickle.dump(*args, f)
