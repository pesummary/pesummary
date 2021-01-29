# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import os
from pesummary.core.file.formats.base_read import (
    Read, SingleAnalysisRead, MultiAnalysisRead
)

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class SingleAnalysisDefault(SingleAnalysisRead):
    """Class to handle result files which only contain a single analysis

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
    injection_parameters: dict
        dictionary of injection parameters extracted from the result file
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, *args, _data=None, **kwargs):
        super(SingleAnalysisDefault, self).__init__(*args, **kwargs)
        if _data is not None:
            self.load(None, _data=_data, **kwargs)


class MultiAnalysisDefault(MultiAnalysisRead):
    """Class to handle result files which contain multiple analyses

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: 2d list
        list of parameters stored in the result file for each analyses
    samples: 2d list
        list of samples stored in the result file for each analyses
    samples_dict: dict
        dictionary of samples stored in the result file keyed by analysis label
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
    def __init__(self, *args, _data=None, **kwargs):
        super(MultiAnalysisDefault, self).__init__(*args, **kwargs)
        if _data is not None:
            self.load(None, _data=_data, **kwargs)


class Default(object):
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
    def load_map(self):
        return {
            "json": self._grab_data_from_json_file,
            "dat": self._grab_data_from_dat_file,
            "txt": self._grab_data_from_dat_file,
            "csv": self._grab_data_from_csv_file,
            "hdf5": self._grab_data_from_hdf5_file,
            "h5": self._grab_data_from_hdf5_file,
            "hdf": self._grab_data_from_hdf5_file,
            "db": self._grab_data_from_sql_database,
            "sql": self._grab_data_from_sql_database,
            "prior": self._grab_data_from_prior_file,
            "npy": self._grab_data_from_numpy_file
        }

    def __new__(
        self, path_to_results_file, _single_default=SingleAnalysisDefault,
        _multi_default=MultiAnalysisDefault, **kwargs
    ):
        self.module = "core"
        self.extension = Read.extension_from_path(path_to_results_file)
        self.load_function = self.load_map(self)[self.extension]
        try:
            self._load_data = self.load_function(path_to_results_file, **kwargs)
        except Exception as e:
            raise Exception(
                "Failed to read data for file %s because: %s" % (
                    path_to_results_file, e
                )
            )
        if np.array(self._load_data["parameters"]).ndim > 1:
            return _multi_default(
                path_to_results_file, _data=self._load_data, **kwargs
            )
        else:
            return _single_default(
                path_to_results_file, _data=self._load_data, **kwargs
            )

    @classmethod
    def load_file(cls, path, **kwargs):
        if not os.path.isfile(path):
            raise FileNotFoundError("%s does not exist" % (path))
        return cls(path, **kwargs)

    @staticmethod
    def _default_injection(parameters):
        """Return a dictionary of nan's for each parameter

        Parameters
        ----------
        parameters: tuple/list
            list of parameters you wish to have null injections for
        """
        return {i: float("nan") for i in parameters}

    @staticmethod
    def _grab_data_from_dat_file(path, **kwargs):
        """Grab the data stored in a .dat file
        """
        from pesummary.core.file.formats.dat import read_dat

        parameters, samples = read_dat(path)
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters)
        }

    @staticmethod
    def _grab_data_from_numpy_file(path, **kwargs):
        """Grab the data stored in a .npy file
        """
        from pesummary.core.file.formats.numpy import read_numpy

        parameters, samples = read_numpy(path)
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters)
        }

    @staticmethod
    def _grab_data_from_csv_file(path, **kwargs):
        """Grab the data stored in a .csv file
        """
        from pesummary.core.file.formats.csv import read_csv

        parameters, samples = read_csv(path)
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters)
        }

    @staticmethod
    def _grab_data_from_prior_file(path, module="core", **kwargs):
        """Grab the data stored in a .prior file
        """
        import importlib

        module = importlib.import_module(
            "pesummary.{}.file.formats.bilby".format(module)
        )
        func = getattr(module, "prior_samples_from_file")

        samples = func(path, **kwargs)
        parameters = samples.parameters
        analytic = samples.analytic
        return {
            "parameters": parameters, "samples": samples.samples.T.tolist(),
            "injection": Default._default_injection(parameters),
            "analytic": analytic
        }

    @staticmethod
    def _grab_data_from_sql_database(path, **kwargs):
        """Grab the data stored in a sql database
        """
        from pesummary.core.file.formats.sql import read_sql

        parameters, samples, labels = read_sql(path, **kwargs)
        if len(labels) > 1:
            injection = {
                label: Default._default_injection(parameters[num]) for num, label
                in enumerate(labels)
            }
        else:
            injection = Default._default_injection(parameters)
        return {
            "parameters": parameters, "samples": samples, "injection": injection,
            "labels": labels
        }

    @staticmethod
    def _grab_data_from_json_file(path, path_to_samples=None, **kwargs):
        """Grab the data stored in a .json file
        """
        from pesummary.core.file.formats.json import read_json

        parameters, samples = read_json(path, path_to_samples=path_to_samples)
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters)
        }

    @staticmethod
    def _grab_data_from_hdf5_file(
        path, remove_params=[], path_to_samples=None, **kwargs
    ):
        """Grab the data stored in an hdf5 file
        """
        from pesummary.core.file.formats.hdf5 import read_hdf5

        parameters, samples = read_hdf5(
            path, remove_params=remove_params, path_to_samples=path_to_samples
        )
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters)
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
