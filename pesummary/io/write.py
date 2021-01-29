# Licensed under an MIT style license -- see LICENSE.md

import importlib

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def write(*args, package="gw", file_format="dat", **kwargs):
    """Read in a results file.

    Parameters
    ----------
    args: tuple
        all args are passed to write function
    package: str
        the package you wish to use
    file_format: str
        the file format you wish to use. Default None. If None, the read
        function loops through all possible options
    kwargs: dict
        all kwargs passed to write function
    """
    def _import(package, file_format):
        """Import format module with importlib
        """
        return importlib.import_module(
            "pesummary.{}.file.formats.{}".format(package, file_format)
        )

    def _write(module, file_format, args, kwargs):
        """Execute the write method
        """
        return getattr(module, "write_{}".format(file_format))(*args, **kwargs)

    if file_format == "h5":
        file_format = "hdf5"

    try:
        module = _import(package, file_format)
        return _write(module, file_format, args, kwargs)
    except (ImportError, AttributeError, ModuleNotFoundError):
        module = _import("core", file_format)
        return _write(module, file_format, args, kwargs)


def _multi_analysis_write(
    write_function, parameters, *args, labels=None,
    file_format="dat", outdir="./", filename=None, filenames=[],
    _return=False, **kwargs
):
    """Write a set of samples to file. If a 2d list of parameters are provided
    (i.e. samples from multiple analyses), the samples are saved to separate
    files

    Parameters
    ----------
    write_function: func
        the write function you wish to execute
    parameters: nd list
        list of parameters for a single analysis, or a 2d of parameters from
        multiple analyses
    *args: tuple
        all other arguments are passed to write_function
    labels: list, optional
        labels to use for identifying different analyses. Only used if
        parameters is a 2d list
    file_format: str, optional
        file format you wish to save the file to. Default 'dat'
    outdir: str, optional
        directory to save the files. Default './'
    filename: str, optional
        filename to save a single analyses to. If multiple analyses are to be
        saved, the filenames are '{filename}_{label}.{extension}'. Default
        'pesummary.{file_format}'
    filenames: list, optional
        filenames to save the files to. Only used if parameters is a 2d list
    **kwargs: dict, optional
        all other kwargs are passed to write_function
    """
    import numpy as np
    from pathlib import Path
    from pesummary.utils.utils import logger
    import copy

    _filenames = copy.deepcopy(filenames)
    _file_dict = {}
    if np.array(parameters).ndim > 1:
        log = True
        _label = kwargs.get("label", None)
        if labels is None and _label is not None:
            labels = [
                "{}_{}".format(_label, idx) for idx in range(len(parameters))
            ]
        elif labels is None:
            labels = np.arange(len(parameters))
        if len(_filenames) and len(_filenames) != len(parameters):
            raise ValueError("Please provide a filename for each analysis")
        elif len(_filenames):
            log = False
        elif not len(_filenames):
            if filename is None:
                filename = "pesummary.{}".format(file_format)
            filename = Path(filename)
            for idx in range(len(parameters)):
                _filenames.append(
                    "{}_{}{}".format(filename.stem, idx, filename.suffix)
                )
        if log:
            logger.info(
                "Only a single set of samples can be written to a {} file. "
                "Saving each analysis to a separate file. Filenames will be: "
                "{}".format(file_format, ", ".join(_filenames))
            )
        for idx in range(len(parameters)):
            _args = [a[idx] for a in args]
            _file_dict[labels[idx]] = write_function(
                parameters[idx], *_args, file_format=file_format, outdir=outdir,
                filename=_filenames[idx], **kwargs
            )
        if _return:
            return _file_dict
    else:
        return write_function(
            parameters, *args, file_format=file_format, outdir=outdir,
            filename=filename, **kwargs
        )
