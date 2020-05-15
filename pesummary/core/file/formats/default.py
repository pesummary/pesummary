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
        from pesummary.core.file.formats.dat import read_dat

        parameters, samples = read_dat(path)
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_json_file(path):
        """Grab the data stored in a .json file
        """
        from pesummary.core.file.formats.json import read_json

        parameters, samples = read_json(path)
        injection = {i: float("nan") for i in parameters}
        return {
            "parameters": parameters, "samples": samples, "injection": injection
        }

    @staticmethod
    def _grab_data_from_hdf5_file(path, remove_params=[]):
        """Grab the data stored in an hdf5 file
        """
        from pesummary.core.file.formats.hdf5 import read_hdf5

        parameters, samples = read_hdf5(path, remove_params=remove_params)
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
