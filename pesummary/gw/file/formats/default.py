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
from pesummary.core.file.formats.default import Default as CoreDefault


class Default(GWRead):
    """Class to handle the default loading options.

    Parameters
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

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    to_lalinference:
        convert the posterior samples to a lalinference result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
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
    def grab_extra_kwargs(parameters, samples):
        """Grab any additional information stored in the file
        """
        def find_parameter_given_alternatives(parameters, options):
            if any(i in options for i in parameters):
                parameter = [i for i in parameters if i in options]
                ind = parameters.index(parameter[0])
                return ind
            return None

        kwargs = {"sampler": {}, "meta_data": {}}
        possible_f_ref = ["f_ref", "fRef", "fref", "fref_template"]
        ind = find_parameter_given_alternatives(parameters, possible_f_ref)
        if ind is not None:
            kwargs["meta_data"]["f_ref"] = samples[0][ind]
        possible_f_low = ["flow", "f_low", "fLow", "flow_template"]
        ind = find_parameter_given_alternatives(parameters, possible_f_low)
        if ind is not None:
            kwargs["meta_data"]["f_low"] = samples[0][ind]
        return kwargs

    @staticmethod
    def _grab_data_from_dat_file(path):
        """Grab the data stored in a .dat file
        """
        data = CoreDefault._grab_data_from_dat_file(path)
        parameters, samples = data["parameters"], data["samples"]
        parameters = GWRead._check_definition_of_inclination(parameters)
        condition1 = "luminosity_distance" not in parameters
        condition2 = "logdistance" in parameters
        if condition1 and condition2:
            parameters.append("luminosity_distance")
            for num, i in enumerate(samples):
                samples[num].append(
                    np.exp(i[parameters.index("logdistance")]))
        injection = {i: float("nan") for i in parameters}
        try:
            extra_kwargs = Default.grab_extra_kwargs(parameters, samples)
        except Exception:
            extra_kwargs = {"sampler": {}, "meta_data": {}}
        extra_kwargs["sampler"]["nsamples"] = len(samples)
        return {
            "parameters": parameters, "samples": samples,
            "injection": injection, "kwargs": extra_kwargs
        }

    @staticmethod
    def _grab_data_from_hdf5_file(path):
        """Grab the data stored in an hdf5 file
        """
        return CoreDefault._grab_data_from_hdf5_file(
            path, remove_params=["waveform_approximant"]
        )

    @staticmethod
    def _grab_data_from_json_file(path):
        """Grab the data stored in a .json file
        """
        return CoreDefault._grab_data_from_json_file(path)

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
