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

import os
import numpy as np
from pesummary.gw.file.formats.base_read import GWRead


class Default(GWRead):
    """Class to handle the default loading options.

    Attributes
    ----------
    path_to_results_file: str
        path to the results file you wish to load
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
        except Exception:
            raise Exception(
                "Failed to read data for file %s" % (self.path_to_results_file))

    @classmethod
    def load_file(cls, path):
        if not os.path.isfile(path):
            raise Exception("%s does not exist" % (path))
        return cls(path)

    @staticmethod
    def _grab_data_from_dat_file(path):
        """Grab the data stored in a .dat file
        """
        parameters, samples = GWRead._grab_params_and_samples_from_dat_file(
            path)
        parameters = GWRead._check_definition_of_inclination(parameters)
        condition1 = "luminosity_distance" not in parameters
        condition2 = "logdistance" in parameters
        if condition1 and condition2:
            parameters.append("luminosity_distance")
            for num, i in enumerate(samples):
                samples[num].append(
                    np.exp(i[parameters.index("logdistance")]))
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_json_file(path):
        """Grab the data stored in a .json file
        """
        parameters, samples = GWRead._grab_params_and_samples_from_json_file(
            path)
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_hdf5_file(path):
        """Grab the data stored in an hdf5 file
        """
        try:
            return Default._grab_data_with_deepdish(path)
        except Exception:
            return Default._grab_data_with_h5py(path)

    @staticmethod
    def _grab_data_with_deepdish(path):
        """Grab the data stored in a h5py file with `deepdish`.
        """
        import deepdish

        f = deepdish.io.load(path)
        path_to_results, = GWRead.paths_to_key("posterior", f)
        if path_to_results[0] == []:
            path_to_results, = GWRead.paths_to_key("posterior_samples", f)
        path_to_results = path_to_results[0]
        reduced_f, = GWRead.load_recusively(path_to_results, f)
        parameters = [i for i in reduced_f.keys()]
        if "waveform_approximant" in parameters:
            approx = reduced_f["waveform_approximant"][0]
            parameters.remove("waveform_approximant")
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
    def _grab_data_with_h5py(path):
        """Grab the data stored in a hdf5 file with `h5py`.
        """
        import h5py

        path_to_samples = GWRead.guess_path_to_samples(path)

        f = h5py.File(path, 'r')
        if isinstance(f[path_to_samples], h5py._hl.group.Group):
            original_parameters = [i for i in f[path_to_samples].keys()]
            parameters = [
                i for i in original_parameters if i != "waveform_approximant"]
            n_samples = len(f[path_to_samples][parameters[0]])
            samples = [[f[path_to_samples][original_parameters.index(i)][num]
                        for i in parameters] for num in range(n_samples)]
            cond1 = "loglr" not in parameters or "log_likelihood" not in \
                parameters
            cond2 = "likelihood_stats" in f.keys() and "loglr" in \
                f["likelihood_stats"]
            if cond1 and cond2:
                parameters.append("log_likelihood")
                for num, i in enumerate(samples):
                    samples[num].append(f["likelihood_stats/loglr"][num])
        elif isinstance(f[path_to_samples], h5py._hl.dataset.Dataset):
            parameters = f[path_to_samples].dtype.names
            samples = [[i[parameters.index(j)] for j in parameters] for i in
                       f[path_to_samples]]
        f.close()
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @property
    def calibration_data_in_results_file(self):
        """
        """
        return None

    def add_injection_parameters_from_file(self, injection_file):
        """
        """
        self.injection_parameters = self._add_injection_parameters_from_file(
            injection_file, self._grab_injection_parameters_from_file)

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass
