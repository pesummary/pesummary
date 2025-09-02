# Licensed under an MIT style license -- see LICENSE.md

import h5py
import numpy as np
from pesummary.core.file.formats.base_read import Read
from pesummary.utils.dict import load_recursively, paths_to_key

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_hdf5(path, **kwargs):
    """Grab the parameters and samples in a .hdf5 file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    kwargs: dict
        all kwargs passed to _read_hdf5_with_h5py functions
    """
    return _read_hdf5_with_h5py(path, **kwargs)


def _read_hdf5_with_h5py(
    path, remove_params=None, path_to_samples=None,
    return_posterior_dataset=False
):
    """Grab the parameters and samples in a .hdf5 file with h5py

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    remove_params: list, optional
        parameters you wish to remove from the posterior table
    """
    import h5py
    import copy

    if path_to_samples is None:
        path_to_samples = Read.guess_path_to_samples(path)

    f = h5py.File(path, 'r')
    c1 = isinstance(f[path_to_samples], h5py._hl.group.Group)
    if c1 and "parameter_names" not in f[path_to_samples].keys():
        original_parameters = [i for i in f[path_to_samples].keys()]
        if remove_params is not None:
            parameters = [
                i for i in original_parameters if i not in remove_params
            ]
        else:
            parameters = copy.deepcopy(original_parameters)
        n_samples = len(f[path_to_samples][parameters[0]])
        try:
            samples = np.array([
                f[path_to_samples][original_parameters.index(i)] for i in
                parameters
            ]).T
        except (AttributeError, KeyError, TypeError):
            samples = np.array([f[path_to_samples][i] for i in parameters]).T
        cond1 = "loglr" not in parameters or "log_likelihood" not in \
            parameters
        cond2 = "likelihood_stats" in f.keys() and "loglr" in \
            f["likelihood_stats"]
        if cond1 and cond2:
            parameters.append("log_likelihood")
            for num, i in enumerate(samples):
                samples[num].append(float(f["likelihood_stats/loglr"][num]))
    elif c1:
        original_parameters = [
            i.decode("utf-8") if isinstance(i, bytes) else i for i in
            f[path_to_samples]["parameter_names"]
        ]
        if remove_params is not None:
            parameters = [
                i for i in original_parameters if i not in remove_params
            ]
        else:
            parameters = copy.deepcopy(original_parameters)
        samples = np.array(f[path_to_samples]["samples"])
    elif isinstance(f[path_to_samples], h5py._hl.dataset.Dataset):
        parameters = list(f[path_to_samples].dtype.names)
        samples = np.array(f[path_to_samples]).view((float, len(parameters))).tolist()
    for num, par in enumerate(parameters):
        if par == "logL":
            parameters[num] = "log_likelihood"
    if return_posterior_dataset:
        return parameters, samples, f[path_to_samples]
    f.close()
    return parameters, samples


def _write_hdf5(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    dataset_name="posterior_samples", **kwargs
):
    """Write a set of samples to a hdf5 file

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
    dataset_name: str, optional
        name of the dataset to store a set of samples. Default posterior_samples
    """
    from pesummary.utils.samples_dict import SamplesDict
    from pesummary.utils.utils import check_filename

    default_filename = "pesummary_{}.h5"
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=label, filename=filename,
        overwrite=overwrite
    )
    samples = SamplesDict(parameters, np.array(samples).T)
    _samples = samples.to_structured_array()
    with h5py.File(filename, "w") as f:
        f.create_dataset(dataset_name, data=_samples)


def write_hdf5(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    **kwargs
):
    """Write a set of samples to a hdf5 file

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
    from pesummary.io.write import _multi_analysis_write

    _multi_analysis_write(
        _write_hdf5, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, file_format="hdf5", **kwargs
    )
