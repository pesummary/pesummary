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

from pesummary.gw.file.formats.base_read import GWRead
from pesummary.core.file.formats.pesummary import (
    PESummary as CorePESummary, PESummaryDeprecated as CorePESummaryDeprecated,
    deprecation_warning
)
from pesummary.utils.utils import logger
import numpy as np
import warnings


def write_pesummary(*args, **kwargs):
    """Write a set of samples to a pesummary file

    Parameters
    ----------
    args: tuple
        either a 2d tuple containing the parameters as first argument and samples
        as the second argument, or a SamplesDict object containing the samples
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    config: dict, optional
        configuration file that you wish to save to file
    injection_data: dict, optional
        dictionary containing the injection values that you wish to save to file keyed
        by parameter
    file_kwargs: dict, optional
        any kwargs that you wish to save to file
    mcmc_samples: Bool, optional
        if True, the set of samples provided are from multiple MCMC chains
    hdf5: Bool, optional
        if True, save the pesummary file in hdf5 format
    kwargs: dict
        all other kwargs are passed to the pesummary.gw.file.meta_file._GWMetaFile class
    """
    from pesummary.core.file.formats.pesummary import write_pesummary as core_write
    from pesummary.gw.file.meta_file import _GWMetaFile

    return core_write(*args, cls=_GWMetaFile, **kwargs)


class PESummary(GWRead, CorePESummary):
    """This class handles the existing posterior_samples.h5 file

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
    approximant: list
        list of approximants stored in the result file
    labels: list
        list of analyses stored in the result file
    config: list
        list of dictonaries containing the configuration files for each
        analysis
    psd: dict
        dictionary containing the psds stored in the result file keyed by
        the analysis label
    calibration: dict
        dictionary containing the calibration posterior samples keyed by
        the analysis label
    skymap: dict
        dictionary containing the skymap probabilities keyed by the analysis
        label
    prior: dict
        dictionary containing the prior samples for each analysis
    weights: dict
        dictionary of weights for each samples for each analysis
    detectors: list
        list of IFOs used in each analysis

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
    to_bilby:
        convert the posterior samples to a bilby result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(PESummary, self).__init__(
            path_to_results_file=path_to_results_file
        )

    @property
    def load_kwargs(self):
        return dict(grab_data_from_dictionary=self._grab_data_from_dictionary)

    def load(self, function, **kwargs):
        """Load a results file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your results file
        """
        data = self.load_from_function(
            function, self.path_to_results_file, **kwargs)
        self.data = data
        self.parameters = self.data["parameters"]
        self.samples = self.data["samples"]
        if "mcmc_samples" in data.keys():
            self.mcmc_samples = data["mcmc_samples"]
        if "version" in data.keys() and data["version"] is not None:
            self.input_version = data["version"]
        else:
            self.input_version = "No version information found"
        if "kwargs" in data.keys():
            self.extra_kwargs = data["kwargs"]
        else:
            self.extra_kwargs = {"sampler": {}, "meta_data": {}}
            self.extra_kwargs["sampler"]["nsamples"] = len(self.data["samples"])
        if data["injection"] is not None:
            self.injection_parameters = data["injection"]
        if "prior" in data.keys() and data["prior"] != {}:
            self.priors = data["prior"]
        if "weights" in self.data.keys():
            self.weights = self.data["weights"]
        if "approximant" in self.data.keys():
            self.approximant = self.data["approximant"]
        if "labels" in self.data.keys():
            self.labels = self.data["labels"]
        if "config" in self.data.keys():
            self.config = self.data["config"]
        if "psd" in self.data.keys():
            from pesummary.gw.file.psd import PSD

            try:
                self.psd = {
                    label: {
                        ifo: PSD(value) for ifo, value in psd_data.items()
                    } for label, psd_data in self.data["psd"].items()
                }
            except (KeyError, AttributeError):
                self.psd = self.data["psd"]
        if "calibration" in self.data.keys():
            from pesummary.gw.file.calibration import Calibration

            try:
                self.calibration = {
                    label: {
                        ifo: Calibration(value) for ifo, value in
                        calibration_data.items()
                    } for label, calibration_data in
                    self.data["calibration"].items()
                }
            except (KeyError, AttributeError):
                self.calibration = self.data["calibration"]
        if "calibration" in self.data["prior"].keys():
            from pesummary.gw.file.calibration import Calibration

            try:
                self.priors["calibration"] = {
                    label: {
                        ifo: Calibration(value) for ifo, value in
                        calibration_data.items()
                    } for label, calibration_data in
                    self.data["prior"]["calibration"].items()
                }
            except (KeyError, AttributeError):
                pass
        if "skymap" in self.data.keys():
            from pesummary.gw.file.skymap import SkyMap

            try:
                self.skymap = {
                    label: SkyMap(skymap["data"], skymap["meta_data"])
                    for label, skymap in self.data["skymap"].items()
                }
            except (KeyError, AttributeError):
                self.skymap = self.data["skymap"]

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        stored_data = CorePESummary._grab_data_from_dictionary(
            dictionary=dictionary
        )

        approx_list = list()
        psd_dict, cal_dict, skymap_dict = {}, {}, {}
        psd, cal = None, None
        for num, label in enumerate(stored_data["labels"]):
            data, = GWRead.load_recursively(label, dictionary)
            if "psds" in data.keys():
                psd_dict[label] = data["psds"]
            if "calibration_envelope" in data.keys():
                cal_dict[label] = data["calibration_envelope"]
            if "skymap" in data.keys():
                skymap_dict[label] = data["skymap"]
            if "approximant" in data.keys():
                approx_list.append(data["approximant"])
            else:
                approx_list.append(None)
        stored_data["approximant"] = approx_list
        stored_data["calibration"] = cal_dict
        stored_data["psd"] = psd_dict
        stored_data["skymap"] = skymap_dict
        return stored_data

    @property
    def calibration_data_in_results_file(self):
        if self.calibration:
            keys = [list(self.calibration[i].keys()) for i in self.labels]
            total = [[self.calibration[key][ifo] for ifo in keys[num]] for
                     num, key in enumerate(self.labels)]
            return total, keys
        return None

    @property
    def detectors(self):
        det_list = list()
        for parameters in self.parameters:
            detectors = list()
            for param in parameters:
                if "_optimal_snr" in param and param != "network_optimal_snr":
                    detectors.append(param.split("_optimal_snr")[0])
            if not detectors:
                detectors.append(None)
            det_list.append(detectors)
        return det_list

    def write(self, labels="all", **kwargs):
        """Save the data to file

        Parameters
        ----------
        package: str, optional
            package you wish to use when writing the data
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        approximant = {
            label: self.approximant[num] if self.approximant[num] != {} else
            None for num, label in enumerate(self.labels)
        }
        properties = dict(
            calibration=self.calibration, psd=self.psd, approximant=approximant,
            skymap=self.skymap
        )
        CorePESummary.write(
            self, package="gw", labels=labels, cls_properties=properties, **kwargs
        )

    def generate_all_posterior_samples(self, **conversion_kwargs):
        if "no_conversion" in conversion_kwargs.keys():
            no_conversion = conversion_kwargs.pop("no_conversion")
        else:
            no_conversion = False
        if no_conversion:
            return
        from pesummary.gw.file.conversions import _Conversion

        converted_params, converted_samples, converted_kwargs = [], [], []
        for param, samples, kwargs in zip(
                self.parameters, self.samples, self.extra_kwargs
        ):
            if conversion_kwargs.get("evolve_spins", False):
                if not conversion_kwargs.get("return_kwargs", False):
                    conversion_kwargs["return_kwargs"] = True
            data = _Conversion(
                param, samples, extra_kwargs=kwargs, return_dict=False,
                **conversion_kwargs
            )
            converted_params.append(data[0])
            converted_samples.append(data[1])
            if kwargs.get("return_kwargs", False):
                converted_kwargs.append(data[2])
        self.parameters = converted_params
        self.samples = converted_samples
        if converted_kwargs != []:
            self.extra_kwargs = {
                label: converted_kwargs[num] for num, label in enumerate(
                    self.labels
                )
            }

    def to_bilby(self):
        """Convert a PESummary metafile to a bilby results object
        """
        from bilby.gw.result import CompactBinaryCoalescenceResult
        from bilby.core.prior import Prior, PriorDict
        from pandas import DataFrame

        objects = dict()
        for num, label in enumerate(self.labels):
            priors = PriorDict()
            logger.warn(
                "No prior information is known so setting it to a default")
            priors.update({parameter: Prior() for parameter in self.parameters[num]})
            posterior_data_frame = DataFrame(
                self.samples[num], columns=self.parameters[num])
            meta_data = {
                "likelihood": {
                    "waveform_arguments": {
                        "waveform_approximant": self.approximant[num]},
                    "interferometers": self.detectors[num]}}
            bilby_object = CompactBinaryCoalescenceResult(
                search_parameter_keys=self.parameters[num],
                posterior=posterior_data_frame, label="pesummary_%s" % label,
                samples=self.samples[num], priors=priors, meta_data=meta_data)
            objects[label] = bilby_object
        return objects

    def to_lalinference(self, labels="all", **kwargs):
        """Convert the samples stored in a PESummary metafile to a .dat file

        Parameters
        ----------
        labels: list, optional
            optional list of analyses to save to file
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        return self.write(
            labels=labels, file_format="lalinference", **kwargs
        )


class PESummaryDeprecated(PESummary):
    """
    """
    def __init__(self, path_to_results_file):
        warnings.warn(deprecation_warning)
        super(PESummaryDeprecated, self).__init__(path_to_results_file)

    @property
    def load_kwargs(self):
        return {
            "grab_data_from_dictionary": PESummaryDeprecated._grab_data_from_dictionary
        }

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        data = CorePESummaryDeprecated._grab_data_from_dictionary(
            dictionary=dictionary
        )

        approx_list = list()
        psd, cal = None, None
        for num, key in enumerate(data["labels"]):
            if "psds" in dictionary.keys():
                psd, = GWRead.load_recursively("psds", dictionary)
            if "calibration_envelope" in dictionary.keys():
                cal, = GWRead.load_recursively("calibration_envelope", dictionary)
            if "approximant" in dictionary.keys():
                if key in dictionary["approximant"].keys():
                    approx_list.append(dictionary["approximant"][key])
                else:
                    approx_list.append(None)
            else:
                approx_list.append(None)
        data["approximant"] = approx_list
        data["calibration"] = cal
        data["psd"] = psd

        return data
