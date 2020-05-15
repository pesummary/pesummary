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

import h5py
import numpy as np
from pesummary.core.file.formats.base_read import Read


def read_hdf5(path, **kwargs):
    """Grab the parameters and samples in a .hdf5 file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    kwargs: dict
        all kwargs passed to _read_hdf5_with_deepdish or _read_hdf5_with_h5py functions
    """
    try:
        return _read_hdf5_with_deepdish(path, **kwargs)
    except Exception:
        return _read_hdf5_with_h5py(path, **kwargs)


def _read_hdf5_with_deepdish(path, remove_params=None):
    """Grab the parameters and samples in a .hdf5 file with deepdish

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    remove_params: list, optional
        parameters you wish to remove from the posterior table
    """
    import deepdish

    f = deepdish.io.load(path)
    try:
        path_to_results, = Read.paths_to_key("posterior", f)
        path_to_results = path_to_results[0]
    except ValueError:
        try:
            path_to_results, = Read.paths_to_key("posterior_samples", f)
            path_to_results = path_to_results[0]
        except ValueError:
            raise ValueError(
                "Unable to find a 'posterior' or 'posterior_samples' group in the "
                "file '{}'".format(path)
            )
    reduced_f, = Read.load_recursively(path_to_results, f)
    parameters = [i for i in reduced_f.keys()]
    if remove_params is not None:
        for param in remove_params:
            if param in parameters:
                parameters.remove(param)
    data = np.zeros([len(reduced_f[parameters[0]]), len(parameters)])
    for num, par in enumerate(parameters):
        for key, i in enumerate(reduced_f[par]):
            data[key][num] = float(np.real(i))
    data = data.tolist()
    for num, par in enumerate(parameters):
        if par == "logL":
            parameters[num] = "log_likelihood"
    return parameters, data


def _read_hdf5_with_h5py(path, remove_params=None):
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
        samples = [
            [float(f[path_to_samples][original_parameters.index(i)][num])
             for i in parameters] for num in range(n_samples)
        ]
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
        parameters = f[path_to_samples].dtype.names
        samples = [[float(i[parameters.index(j)]) for j in parameters] for
                   i in f[path_to_samples]]
    f.close()
    return parameters, samples


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
        f.create_dataset("posterior_samples", data=_samples)
