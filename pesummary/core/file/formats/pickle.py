# Licensed under an MIT style license -- see LICENSE.md

from pesummary import conf
import pickle
from pesummary.utils.utils import check_filename

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
