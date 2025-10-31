# Licensed under an MIT style license -- see LICENSE.md

import h5py
import numpy as np
import copy
from ..formats.base_read import Read
from .base import _LazyRead

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class LazyHDF5(_LazyRead, h5py.File):
    """Class to a lazily read a hdf5 file

    Parameters
    ----------
    drop_non_numeric: bool, optional
        if True, remove all non_numeric values from the posterior samples
        table
    remove_nan_likelihood_samples: bool, optional
        if True. remove all rows in the posterior samples table that have 'nan'
        likelihood
    path_to_samples: str, optional
        if provided, this path is used to extract the posterior samples. If
        'None', the "guess_path_to_samples" method is used.
    remove_params: list, optional
        list of parameters you wish to remove from the posterior samples
        table. Default None.

    Attributes
    ----------
    parameters: pesummary.utils.parameters.Parameters
        list of parameters in the posterior samples table
    samples: np.ndarray
        2D array of samples in the posterior samples table
    samples_dict: pesummary.utils.samples_dict.SamplesDict
        dictionary displaying the posterior samples table.
    """
    def __init__(self, *args, path_to_samples=None, remove_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._path_to_samples = path_to_samples
        self._remove_params = remove_params

    def grab_samples_from_file(self):
        self.guess_path_to_samples()
        c1 = isinstance(self[self._path_to_samples], h5py._hl.group.Group)
        if c1 and "parameter_names" not in self[self._path_to_samples].keys():
            original_parameters = [i for i in self[self._path_to_samples].keys()]
            original_samples = np.array(
                [
                    self[self._path_to_samples][i][:] for i in
                    original_parameters
                ]
            ).T
        elif c1:
            original_parameters = [
                i.decode("utf-8") if isinstance(i, bytes) else i for i in
                self[self._path_to_samples]["parameter_names"]
            ]
            samples = np.array(f[path_to_samples]["samples"])
        elif isinstance(self[self._path_to_samples], h5py._hl.dataset.Dataset):
            original_parameters = self[self._path_to_samples].dtype.names
            original_samples = np.array(self[self._path_to_samples]).view(
                (float, len(original_parameters))
            )
        if self._remove_params is not None:
            inds_to_keep = np.array(
                [
                    num for num, i in enumerate(original_parameters) if i
                    not in self._remove_params
                ]
            )
        else:
            inds_to_keep = np.arange(len(original_parameters))
        self._parameters = np.array(original_parameters)[inds_to_keep]
        self._samples = (original_samples[:, inds_to_keep])
        for num, par in enumerate(self._parameters):
            if par == "logL":
                self._parameters[num] = "log_likelihood"
        return self._samples

    def guess_path_to_samples(self):
        if self._path_to_samples is None:
            try:
                self._path_to_samples = Read.guess_path_to_samples(self.filename)
            except ValueError:
                # raised if more than one set of samples found in the file.
                raise
        return self._path_to_samples
