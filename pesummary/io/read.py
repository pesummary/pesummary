# Licensed under an MIT style license -- see LICENSE.md

import os
import importlib
from pathlib import Path
from pesummary.core.file.formats.ini import read_ini
from pesummary.core.file.formats.pickle import read_pickle
from pesummary.gw.file.strain import StrainData
from pesummary.gw.file.skymap import SkyMap

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

OTHER = {
    "fits": SkyMap.from_fits,
    "ini": read_ini,
    "gwf": StrainData.read,
    "lcf": StrainData.read,
    "pickle": read_pickle
}


def _fetch(ff, function):
    """Copy a file from a remote location to a temporary folder ready for
    reading

    Parameters
    ----------
    ff: str
        path to file you wish to read
    function: func
        function you wish to use for fetching the file from a remote location
    """
    filename = function(ff, read_file=False)
    path = str(filename)
    return path


def _fetch_from_url(url):
    """Copy file from url to a temporary folder ready for reading

    Parameters
    ----------
    url: str
        url of a result file you wish to load
    """
    from pesummary.core.fetch import download_and_read_file
    return _fetch(url, download_and_read_file)


def _fetch_from_remote_server(ff):
    """Copy file from remote server to a temporary folder ready for reading

    Parameters
    ----------
    ff: str
        path to a results file on a remote server. Must be in the form
        {username}@{servername}:{path}
    """
    from pesummary.core.fetch import scp_and_read_file
    return _fetch(ff, scp_and_read_file)


def read(
    path, package="gw", file_format=None, skymap=False, strain=False, cls=None,
    checkpoint=False, **kwargs
):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to results file. If path is on a remote server, add username and
        servername in the form {username}@{servername}:{path}
    package: str
        the package you wish to use
    file_format: str
        the file format you wish to use. Default None. If None, the read
        function loops through all possible options
    skymap: Bool, optional
        if True, path is the path to a fits file generated with `ligo.skymap`
    strain: Bool, optional
        if True, path is the path to a frame file containing gravitational
        wave data. All kwargs are passed to
        pesummary.gw.file.strain.StrainData.read
    cls: func, optional
        class to use when reading in a result file
    checkpoint: Bool, optional
        if True, treat path as the path to a checkpoint file
    **kwargs: dict, optional
        all additional kwargs are passed to the `pesummary.{}.file.read.read`
        function
    """
    if not os.path.isfile(path) and "https://" in path:
        path = _fetch_from_url(path)
    elif not os.path.isfile(path) and "@" in path:
        path = _fetch_from_remote_server(path)
    extension = Path(path).suffix[1:]
    if cls is not None:
        return cls.load_file(path, **kwargs)
    if extension in OTHER.keys():
        return OTHER[extension](path, **kwargs)
    elif file_format == "ini":
        return OTHER["ini"](path, **kwargs)
    elif skymap:
        return OTHER["fits"](path, **kwargs)
    elif strain:
        return OTHER["gwf"](path, **kwargs)
    elif checkpoint:
        return OTHER["pickle"](path, **kwargs)

    module = importlib.import_module("pesummary.{}.file.read".format(package))
    return getattr(module, "read")(path, file_format=file_format, **kwargs)
