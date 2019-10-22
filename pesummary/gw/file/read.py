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
from pesummary.gw.file.formats.lalinference import LALInference
from pesummary.gw.file.formats.bilby import Bilby
from pesummary.gw.file.formats.default import Default
from pesummary.gw.file.formats.pesummary import PESummary
from pesummary.core.file.read import is_bilby_hdf5_file, is_bilby_json_file
from pesummary.core.file.read import is_pesummary_hdf5_file
from pesummary.core.file.read import is_pesummary_json_file
from pesummary.utils.utils import logger


def read(path):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to the results file
    """
    extension = Read.extension_from_path(path)

    if extension in ["hdf5", "h5", "hdf"]:
        if is_lalinference_file(path):
            return LALInference.load_file(path)
        elif is_bilby_hdf5_file(path):
            try:
                return Bilby.load_file(path)
            except ImportError:
                logger.warn(
                    "Failed to import `bilby`. Using default load")
                return Default.laad_file(path)
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


def is_lalinference_file(path):
    """Determine if the results file is a LALInference results file

    Parameters
    ----------
    path: str
        path to the results file
    """
    import h5py
    f = h5py.File(path, 'r')
    keys = list(f.keys())
    f.close()
    if "lalinference" in keys:
        return True
    return False
