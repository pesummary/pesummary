# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary import conf
from pesummary.utils.utils import logger, check_filename

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_dat(path, delimiter=None):
    """Grab the parameters and samples in a .dat file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    from .numpy import genfromtxt
    return genfromtxt(path, delimiter=delimiter, names=True)


def _write_dat(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    delimiter=conf.delimiter, default_filename="pesummary_{}.dat", **kwargs
):
    """Write a set of samples to a dat file

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    delimiter: str, optional
        The delimiter you wish to use for the dat file
    """
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite
    )
    np.savetxt(
        filename, samples, delimiter=delimiter, header=delimiter.join(parameters),
        comments=''
    )


def write_dat(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    delimiter=conf.delimiter, **kwargs
):
    """Write a set of samples to a dat file

    Parameters
    ----------
    parameters: nd list
        list of parameters
    samples: nd list
        list of samples. Columns correspond to a given parameter
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    delimiter: str, optional
        The delimiter you wish to use for the dat file
    """
    from pesummary.io.write import _multi_analysis_write

    _multi_analysis_write(
        _write_dat, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, delimiter=delimiter,
        file_format="dat", **kwargs
    )
