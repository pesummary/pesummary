# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.dat import read_dat, _write_dat

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_csv(path):
    """Grab the parameters and samples in a .csv file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    return read_dat(path, delimiter=",")


def _write_csv(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    **kwargs
):
    """Write a set of samples to a csv file

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
    """
    return _write_dat(
        parameters, samples, outdir="./", label=label, filename=filename,
        overwrite=overwrite, delimiter=",", default_filename="pesummary_{}.csv",
        **kwargs
    )


def write_csv(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    **kwargs
):
    """Write a set of samples to a csv file

    Parameters
    ----------
    parameters: nd list
        list of parameters
    samples: nd list
        list of samples. Columns correspond to a given parameter
    outdir: str, optional
        directory to write the csv file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    """
    from pesummary.io.write import _multi_analysis_write

    _multi_analysis_write(
        _write_csv, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, file_format="csv", **kwargs
    )
