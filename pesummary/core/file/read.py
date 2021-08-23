# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.base_read import Read
from pesummary.core.file.formats.bilby import Bilby
from pesummary.core.file.formats.default import Default
from pesummary.core.file.formats.pesummary import PESummary, PESummaryDeprecated
from pesummary.utils.utils import logger
import os

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def is_bilby_hdf5_file(path):
    """Determine if the results file is a bilby hdf5 results file

    Parameters
    ----------
    path: str
        path to the results file
    """
    import h5py
    try:
        f = h5py.File(path, "r")
        if "bilby" in f["version"]:
            return True
        elif "bilby" in str(f["version"][0]):
            return True
        else:
            return False
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
        elif "bilby" in data["version"][0]:
            return True
        else:
            return False
    except Exception:
        return False


def _is_pesummary_hdf5_file(path, check_function):
    """Determine if the results file is a pesummary hdf5 file

    Parameters
    ----------
    path: str
        path to the results file
    check_function: func
        function used to check the result file
    """
    import h5py
    f = h5py.File(path, 'r')
    outcome = check_function(f)
    f.close()
    return outcome


def is_pesummary_hdf5_file_deprecated(path):
    """Determine if the results file is a deprecated pesummary hdf5 file

    Parameters
    ----------
    path: str
        path to the results file
    """
    return _is_pesummary_hdf5_file(path, _check_pesummary_file_deprecated)


def is_pesummary_hdf5_file(path):
    """Determine if the results file is a pesummary hdf5 file

    Parameters
    ----------
    path: str
        path to the results file
    """
    return _is_pesummary_hdf5_file(path, _check_pesummary_file)


def _is_pesummary_json_file(path, check_function):
    """Determine if the result file is a pesummary json file

    Parameters
    ----------
    path: str
        path to the results file
    check_function: func
        function used to check the result file
    """
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return check_function(data)


def is_pesummary_json_file(path):
    """Determine if the results file is a pesummary json file

    Parameters
    ----------
    path: str
        path to results file
    """
    return _is_pesummary_json_file(path, _check_pesummary_file)


def is_pesummary_json_file_deprecated(path):
    """Determine if the results file is a deprecated pesummary json file

    Parameters
    ----------
    path: str
        path to results file
    """
    return _is_pesummary_json_file(path, _check_pesummary_file_deprecated)


def _check_pesummary_file_deprecated(f):
    """Check the contents of a dictionary to see if it is a deprecated pesummary
    dictionary

    Parameters
    ----------
    f: dict
        dictionary of the contents of the file
    """
    if "posterior_samples" in f.keys():
        try:
            import collections

            labels = f["posterior_samples"].keys()
            if isinstance(labels, collections.abc.KeysView):
                _label = list(labels)[0]
                try:
                    _param = list(f["posterior_samples"][_label].keys())[0]
                    samples = f["posterior_samples"][_label][_param]
                    return True
                except Exception:
                    return False
            else:
                return False
        except Exception:
            return False
    else:
        return False


def _check_pesummary_file(f):
    """Check the contents of a dictionary to see if it is a pesummary dictionary

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
                "posterior_samples" in f[label].keys() for label in labels if
                label != "version" and label != "history" and label != "strain"
        ):
            return True
        elif all(
                "mcmc_chains" in f[label].keys() for label in labels if
                label != "version" and label != "history" and label != "strain"
        ):
            return True
        else:
            return False
    except Exception:
        return False


CORE_HDF5_LOAD = {
    is_bilby_hdf5_file: Bilby.load_file,
    is_pesummary_hdf5_file: PESummary.load_file,
    is_pesummary_hdf5_file_deprecated: PESummaryDeprecated.load_file
}

CORE_JSON_LOAD = {
    is_bilby_json_file: Bilby.load_file,
    is_pesummary_json_file: PESummary.load_file,
    is_pesummary_json_file_deprecated: PESummaryDeprecated.load_file
}

CORE_DEFAULT_LOAD = {
    "default": Default.load_file
}

DEFAULT_FORMATS = ["default", "dat", "json", "hdf5", "h5", "txt"]


def _read(path, load_options, default=CORE_DEFAULT_LOAD, **load_kwargs):
    """Try and load a result file according to multiple options

    Parameters
    ----------
    path: str
        path to results file
    load_options: dict
        dictionary of checks and loading functions
    """
    for check, load in load_options.items():
        if check(path):
            try:
                return load(path, **load_kwargs)
            except ImportError as e:
                logger.warning(
                    "Failed due to import error: {}. Using default load".format(
                        e
                    )
                )
                return default["default"](path, **load_kwargs)
            except Exception as e:
                logger.info(
                    "Failed to read in {} with the {} class because {}".format(
                        path, load, e
                    )
                )
                continue
    if len(load_options):
        logger.warning(
            "Using the default load because {} failed the following checks: {}".format(
                path, ", ".join([function.__name__ for function in load_options.keys()])
            )
        )
    return default["default"](path, **load_kwargs)


def _file_format(file_format, options):
    """Return a dictionary of load options. If a file format is specified, then return
    a dictionary containing only that one load option

    Parameters
    ----------
    file_format: str
        the file format you wish to use. If None, all possible load options will be
        returned
    options: dict
        dictionary of possible load options.  Key is a function which returns True
        or False depending on whether the input file belongs to that class of objects,
        value is the load function
    """
    if file_format is None:
        return options
    elif any(file_format.lower() in key for key in DEFAULT_FORMATS):
        reduced = {}
    elif not any(file_format in key.__name__ for key in options.keys()):
        raise ValueError(
            "Unrecognised file format. The format must be a substring of any"
            "of the following: {}".format(
                ", ".join([key.__name__ for key in options.keys()] + DEFAULT_FORMATS)
            )
        )
    else:
        reduced = {
            key: value for key, value in options.items() if file_format in key.__name__
        }
    return reduced


def read(
    path, HDF5_LOAD=CORE_HDF5_LOAD, JSON_LOAD=CORE_JSON_LOAD,
    DEFAULT=CORE_DEFAULT_LOAD, file_format=None, **kwargs
):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to results file
    format: str
        the file format you wish to use. Default None. If None, the read
        function loops through all possible options
    **kwargs: dict, optional
        all additional kwargs are passed directly to the load_file class method
    """
    path = os.path.expanduser(path)
    extension = Read.extension_from_path(path)

    if extension in ["hdf5", "h5", "hdf"]:
        options = _file_format(file_format, HDF5_LOAD)
        return _read(path, options, default=DEFAULT, **kwargs)
    elif extension == "json":
        options = _file_format(file_format, JSON_LOAD)
        return _read(path, options, default=DEFAULT, **kwargs)
    else:
        return DEFAULT["default"](path, file_format=file_format, **kwargs)
