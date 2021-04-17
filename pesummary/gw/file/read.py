# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.base_read import Read
from pesummary.gw.file.formats.lalinference import LALInference
from pesummary.gw.file.formats.bilby import Bilby
from pesummary.gw.file.formats.default import Default
from pesummary.gw.file.formats.pesummary import (
    TGRPESummary, PESummary, PESummaryDeprecated
)
from pesummary.gw.file.formats.GWTC1 import GWTC1
from pesummary.core.file.read import (
    is_bilby_hdf5_file, is_bilby_json_file, is_pesummary_hdf5_file,
    is_pesummary_json_file, is_pesummary_hdf5_file_deprecated,
    is_pesummary_json_file_deprecated, _is_pesummary_hdf5_file,
    _is_pesummary_json_file
)
from pesummary.core.file.read import read as CoreRead
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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


def is_tgr_pesummary_hdf5_file(path):
    """Determine if the results file is a pesummary TGR hdf5 file

    Parameters
    ----------
    path: str
        path to results file
    """
    return _is_pesummary_hdf5_file(path, _check_tgr_pesummary_file)


def is_tgr_pesummary_json_file(path):
    """Determine if the results file is a pesummary TGR json file

    Parameters
    ----------
    path: str
        path to results file
    """
    return _is_pesummary_json_file(path, _check_tgr_pesummary_file)


def _check_tgr_pesummary_file(f):
    """Check the contents of a dictionary to see if it is a pesummary TGR
    dictionary

    Parameters
    ----------
    f: dict
        dictionary of the contents of the file
    """
    labels = f.keys()
    if "version" not in labels:
        return False
    try:
        if all(
            "imrct" in f[label].keys() for label in labels if label != "version"
            and label != "history"
        ):
            return True
        else:
            return False
    except Exception:
        return False


GW_HDF5_LOAD = {
    is_lalinference_file: LALInference.load_file,
    is_bilby_hdf5_file: Bilby.load_file,
    is_tgr_pesummary_hdf5_file: TGRPESummary.load_file,
    is_pesummary_hdf5_file: PESummary.load_file,
    is_pesummary_hdf5_file_deprecated: PESummaryDeprecated.load_file,
    is_GWTC1_file: GWTC1.load_file
}

GW_JSON_LOAD = {
    is_bilby_json_file: Bilby.load_file,
    is_tgr_pesummary_json_file: TGRPESummary.load_file,
    is_pesummary_json_file: PESummary.load_file,
    is_pesummary_json_file_deprecated: PESummaryDeprecated.load_file
}

GW_DEFAULT = {"default": Default.load_file}


def read(
    path, HDF5_LOAD=GW_HDF5_LOAD, JSON_LOAD=GW_JSON_LOAD, file_format=None,
    **kwargs
):
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
    **kwargs: dict, optional
        all additional kwargs are passed directly to the load_file class method
    """
    return CoreRead(
        path, HDF5_LOAD=GW_HDF5_LOAD, JSON_LOAD=GW_JSON_LOAD, DEFAULT=GW_DEFAULT,
        file_format=file_format, **kwargs
    )
