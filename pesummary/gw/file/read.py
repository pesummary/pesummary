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
from pesummary.gw.file.formats.pesummary import PESummary, PESummaryDeprecated
from pesummary.gw.file.formats.GWTC1 import GWTC1
from pesummary.core.file.read import (
    is_bilby_hdf5_file, is_bilby_json_file, is_pesummary_hdf5_file,
    is_pesummary_json_file, is_pesummary_hdf5_file_deprecated,
    is_pesummary_json_file_deprecated
)
from pesummary.core.file.read import read as CoreRead
from pesummary.utils.utils import logger


def is_GWTC1_file(path):
    """Determine if the results file is one released as part of the GWTC1
    catalog

    Parameters
    ----------
    path: str
        path to the results file
    """
    import h5py

    f = h5py.File(path, 'r')
    keys = list(f.keys())
    f.close()
    if "Overall_posterior" in keys or "overall_posterior" in keys:
        return True
    return False


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


GW_HDF5_LOAD = {
    is_lalinference_file: LALInference.load_file,
    is_bilby_hdf5_file: Bilby.load_file,
    is_pesummary_hdf5_file: PESummary.load_file,
    is_pesummary_hdf5_file_deprecated: PESummaryDeprecated.load_file,
    is_GWTC1_file: GWTC1.load_file
}

GW_JSON_LOAD = {
    is_bilby_json_file: Bilby.load_file,
    is_pesummary_json_file: PESummary.load_file,
    is_pesummary_json_file_deprecated: PESummaryDeprecated.load_file
}

GW_DEFAULT = {"default": Default.load_file}


def read(path, HDF5_LOAD=GW_HDF5_LOAD, JSON_LOAD=GW_JSON_LOAD, file_format=None):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to results file
    HDF5_LOAD: dict
        dictionary containing possible methods for loading a HDF5 file. Key
        is a function which returns True or False depending on whether the input
        file belongs to that class of objects, value is the load function
    JSON_LOAD: dict
        dictionary containing possible methods for loading a JSON file. Key
        is a function which returns True or False depending on whether the input
        file belongs to that class of objects, value is the load function
    """
    return CoreRead(
        path, HDF5_LOAD=GW_HDF5_LOAD, JSON_LOAD=GW_JSON_LOAD, DEFAULT=GW_DEFAULT,
        file_format=file_format
    )
