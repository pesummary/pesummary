# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from pesummary.gw.file.formats.base_read import (
    GWRead, GWSingleAnalysisRead, GWMultiAnalysisRead
)
from pesummary.core.file.formats.default import Default as CoreDefault

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class SingleAnalysisDefault(GWSingleAnalysisRead):
    """Class to handle result files which only contain a single analysis

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    converted_parameters: list
        list of parameters that have been derived from the sampled distributions
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    converted_parameters: list
        list of parameters that have been added
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
    to_lalinference:
        convert the posterior samples to a lalinference result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
    """
    def __init__(self, *args, _data=None, **kwargs):
        super(SingleAnalysisDefault, self).__init__(*args, **kwargs)
        if _data is not None:
            self.load(None, _data=_data, **kwargs)


class MultiAnalysisDefault(GWMultiAnalysisRead):
    """Class to handle result files which contain multiple analyses

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: 2d list
        list of parameters stored in the result file for each analysis
    converted_parameters: 2d list
        list of parameters that have been derived from the sampled distributions
    samples: 3d list
        list of samples stored in the result file for each analysis
    samples_dict: dict
        dictionary of samples stored in the result file keyed by analysis label
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
    def __init__(self, *args, _data=None, **kwargs):
        super(MultiAnalysisDefault, self).__init__(*args, **kwargs)
        if _data is not None:
            self.load(None, _data=_data, **kwargs)


class Default(CoreDefault):
    """Class to handle the default loading options.

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    converted_parameters: list
        list of parameters that have been derived from the sampled distributions
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    converted_parameters: list
        list of parameters that have been added

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
    def load_map(self):
        _load_map = super(Default, self).load_map(self)
        _load_map.update({
            "xml": self._grab_data_from_xml_file
        })
        return _load_map

    def __new__(self, path_to_results_file, **kwargs):
        data = super(Default, self).__new__(
            self, path_to_results_file, _single_default=SingleAnalysisDefault,
            _multi_default=MultiAnalysisDefault, **kwargs
        )
        self.module = "gw"
        return data

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
    def _grab_data_from_dat_file(path, **kwargs):
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
        try:
            extra_kwargs = Default.grab_extra_kwargs(parameters, samples)
        except Exception:
            extra_kwargs = {"sampler": {}, "meta_data": {}}
        extra_kwargs["sampler"]["nsamples"] = len(samples)
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters),
            "kwargs": extra_kwargs
        }

    @staticmethod
    def _grab_data_from_hdf5_file(path, path_to_samples=None, **kwargs):
        """Grab the data stored in an hdf5 file
        """
        return CoreDefault._grab_data_from_hdf5_file(
            path, remove_params=["waveform_approximant"],
            path_to_samples=path_to_samples, **kwargs
        )

    @staticmethod
    def _grab_data_from_numpy_file(path, file_format=None, **kwargs):
        """Grab the data stored in a .npy file
        """
        if file_format is not None:
            try:
                import importlib
                module = importlib.import_module(
                    "pesummary.gw.file.formats.{}".format(file_format)
                )
                data = getattr(module, "read_{}".format(file_format))(path, **kwargs)
                return {
                    "parameters": data[0], "samples": data[1],
                    "injection": Default._default_injection(data[0])
                }
            except ModuleNotFoundError:
                from pesummary.utils.utils import logger
                logger.warn(
                    "Failed to find the module '{}'. Therefore ignoring "
                    "file_format={} and using default load".format(
                        "pesummary.gw.file.formats.{}".format(file_format),
                        file_format
                    )
                )
        return CoreDefault._grab_data_from_numpy_file(path, **kwargs)

    @staticmethod
    def _grab_data_from_prior_file(path, **kwargs):
        """Grab the data stored in a .prior file
        """
        return CoreDefault._grab_data_from_prior_file(
            path, module="gw", **kwargs
        )

    @staticmethod
    def _grab_data_from_xml_file(path, **kwargs):
        """Grab the data stored in an xml file
        """
        from pesummary.gw.file.formats.xml import read_xml

        parameters, samples = read_xml(path, **kwargs)
        extra_kwargs = {"sampler": {"nsamples": len(samples)}, "meta_data": {}}
        return {
            "parameters": parameters, "samples": samples,
            "injection": Default._default_injection(parameters),
            "kwargs": extra_kwargs
        }
