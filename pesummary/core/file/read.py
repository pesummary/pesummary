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

from pesummary.core.file.formats.base_read import Read
from pesummary.core.file.formats.bilby import Bilby
from pesummary.core.file.formats.default import Default
from pesummary.core.file.formats.pesummary import PESummary
from pesummary.utils.utils import logger


def read(path):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to results file
    """
    extension = Read.extension_from_path(path)

    if extension in ["hdf5", "h5", "hdf"]:
        if is_bilby_hdf5_file(path):
            try:
                return Bilby.load_file(path)
            except ImportError:
                logger.warn(
                    "Failed to import `bilby`. Using default load")
                return Default.load_file(path)
        elif is_pesummary_hdf5_file(path):
            try:
                return PESummary.load_file(path)
            except Exception:
                return Default.load_file(path)
        else:
            return Default.load_file(path)
    elif extension == "json":
        if is_bilby_json_file(path):
            try:
                return Bilby.load_file(path)
            except ImportError:
                logger.warn(
                    "Failed to import `bilby`. Using default load")
                return Default.load_file(path)
        elif is_pesummary_json_file(path):
            try:
                return PESummary.load_file(path)
            except Exception:
                return Default.load_file(path)
        else:
            return Default.load_file(path)
    else:
        return Default.load_file(path)


def is_bilby_hdf5_file(path):
    """Determine if the results file is a bilby hdf5 results file

    Parameters
    ----------
    path: str
        path to the results file
    """
    import deepdish
    try:
        f = deepdish.io.load(path)
        if "bilby" in f["version"]:
            return True
    except Exception:
        return False
    return False


def is_bilby_json_file(path):
    """Determine if the results file is a bilby json results file

    Parameters
    ----------
    path: str
        path to the results file
    """
    import json
    with open(path, "r") as f:
        data = json.load(f)
    try:
        if "bilby" in data["version"]:
            return True
        else:
            return False
    except Exception:
        return False


def is_pesummary_hdf5_file(path):
    """Determine if the results file is a pesummary hdf5 file

    Parameters
    ----------
    path: str
        path to the results file
    """
    import h5py
    f = h5py.File(path, 'r')
    outcome = _check_pesummary_file(f)
    f.close()
    return outcome


def is_pesummary_json_file(path):
    """Determine if the results file is a pesummary json file

    Parameters
    ----------
    path: str
        path to results file
    """
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return _check_pesummary_file(data)


def _check_pesummary_file(f):
    """Check the contents of a dictionary to see if it is a pesummary dictionary

    f: dict
        dictionary of the contents of the file
    """
    if "posterior_samples" in f.keys():
        try:
            import collections

            labels = f["posterior_samples"].keys()
            if isinstance(labels, collections.abc.KeysView):
                return True
            else:
                return False
        except Exception:
            return False
    else:
        return False
