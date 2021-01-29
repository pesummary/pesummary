# Licensed under an MIT style license -- see LICENSE.md

import importlib
from pathlib import Path
from pesummary.core.file.formats.ini import read_ini
from pesummary.core.file.formats.pickle import read_pickle
from pesummary.gw.file.formats.lcf import read_lcf
from pesummary.gw.file.skymap import SkyMap

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

OTHER = {
    "fits": SkyMap.from_fits,
    "ini": read_ini,
    "lcf": read_lcf,
    "pickle": read_pickle
}


def read(
    path, package="gw", file_format=None, skymap=False, cls=None,
    checkpoint=False, **kwargs
):
    """Read in a results file.

    Parameters
    ----------
    path: str
        path to results file
    package: str
        the package you wish to use
    file_format: str
        the file format you wish to use. Default None. If None, the read
        function loops through all possible options
    skymap: Bool, optional
        if True, path is the path to a fits file generated with `ligo.skymap`
    cls: func, optional
        class to use when reading in a result file
    checkpoint: Bool, optional
        if True, treat path as the path to a checkpoint file
    **kwargs: dict, optional
        all additional kwargs are passed to the `pesummary.{}.file.read.read`
        function
    """
    extension = Path(path).suffix[1:]
    if cls is not None:
        return cls.load_file(path, **kwargs)
    if extension in OTHER.keys():
        return OTHER[extension](path, **kwargs)
    elif file_format == "ini":
        return OTHER["ini"](path, **kwargs)
    elif skymap:
        return OTHER["fits"](path, **kwargs)
    elif checkpoint:
        return OTHER["pickle"](path, **kwargs)

    module = importlib.import_module("pesummary.{}.file.read".format(package))
    return getattr(module, "read")(path, file_format=file_format, **kwargs)
