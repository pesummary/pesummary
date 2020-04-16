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

import numpy as np
import os
from pesummary.core.file.formats.base_read import Read


class Default(Read):
    """Class to handle the default loading options.

    Attributes
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    injection_parameters: dict
        dictionary of injection parameters extracted from the result file

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, path_to_results_file):
        super(Default, self).__init__(path_to_results_file)

        func_map = {"json": self._grab_data_from_json_file,
                    "dat": self._grab_data_from_dat_file,
                    "txt": self._grab_data_from_dat_file,
                    "hdf5": self._grab_data_from_hdf5_file,
                    "h5": self._grab_data_from_hdf5_file,
                    "hdf": self._grab_data_from_hdf5_file}

        self.load_function = func_map[self.extension]
        try:
            self.load(self.load_function)
        except Exception as e:
            raise Exception(
                "Failed to read data for file %s because: %s" % (
                    self.path_to_results_file, e
                )
            )

    @classmethod
    def load_file(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("%s does not exist" % (path))
        return cls(path)

    @staticmethod
    def _grab_data_from_dat_file(path):
        """Grab the data stored in a .dat file
        """
        parameters, samples = Read._grab_params_and_samples_from_dat_file(
            path)
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_json_file(path):
        """Grab the data stored in a .json file
        """
        parameters, samples = Read._grab_params_and_samples_from_json_file(
            path
        )
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_hdf5_file(path, cls=None):
        """Grab the data stored in an hdf5 file
        """
        if cls is None:
            cls = Default
        try:
            return getattr(cls, "_grab_data_with_deepdish")(path)
        except Exception:
            return getattr(cls, "_grab_data_with_h5py")(path)

    @staticmethod
    def _grab_data_with_deepdish(path, remove_params=None):
        """Grab the data stored in a h5py file with `deepdish`.
        """
        import deepdish

        f = deepdish.io.load(path)
        path_to_results, = Read.paths_to_key("posterior", f)
        if path_to_results[0] == []:
            path_to_results, = Read.paths_to_key("posterior_samples", f)
        path_to_results = path_to_results[0]
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
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": data, "injection": injection
        }

    @staticmethod
    def _grab_data_with_h5py(path, remove_params=None):
        """Grab the data stored in a hdf5 file with `h5py`.
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
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass
