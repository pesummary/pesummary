# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.formats.base_read import GWMultiAnalysisRead
from pesummary.core.file.formats.pesummary import (
    PESummary as CorePESummary, PESummaryDeprecated as CorePESummaryDeprecated
)
from pesummary.utils.utils import logger
from pesummary.utils.dict import load_recursively
from pesummary.utils.decorators import deprecation
import numpy as np
import warnings

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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


class PESummary(GWMultiAnalysisRead, CorePESummary):
    """This class handles the existing posterior_samples.h5 file

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
    gwdata: pesummary.gw.file.strain.StrainDataDict
        dictionary containing the strain data used in the analysis
    prior: dict
        dictionary containing the prior samples for each analysis
    weights: dict
        dictionary of weights for each samples for each analysis
    detectors: list
        list of IFOs used in each analysis
    pe_algorithm: dict
        name of the algorithm used to generate the each analysis

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

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        stored_data = CorePESummary._grab_data_from_dictionary(
            dictionary=dictionary, ignore=["strain"]
        )

        approx_list = list()
        psd_dict, cal_dict, skymap_dict = {}, {}, {}
        psd, cal = None, None
        for num, label in enumerate(stored_data["labels"]):
            data, = load_recursively(label, dictionary)
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
        if "strain" in dictionary.keys():
            stored_data["gwdata"] = dictionary["strain"]
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

    def to_bilby(self, labels="all", **kwargs):
        """Convert a PESummary metafile to a bilby results object
        """
        from bilby.gw.result import CompactBinaryCoalescenceResult

        return CorePESummary.write(
            self, labels=labels, package="core", file_format="bilby",
            _return=True, cls=CompactBinaryCoalescenceResult, **kwargs
        )

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
    @deprecation(
        "This file format is out-of-date and may not be supported in future "
        "releases."
    )
    def __init__(self, path_to_results_file, **kwargs):
        super(PESummaryDeprecated, self).__init__(path_to_results_file, **kwargs)

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
                psd, = load_recursively("psds", dictionary)
            if "calibration_envelope" in dictionary.keys():
                cal, = load_recursively("calibration_envelope", dictionary)
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


class TGRPESummary(PESummary):
    """This class handles TGR PESummary result files

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
    labels: list
        list of analyses stored in the result file
    file_kwargs: dict
        dictionary of kwargs associated with each label
    imrct_deviation: dict
        dictionary of pesummary.utils.probability_dict.ProbabilityDict2D
        objects, one for each analysis
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(PESummary, self).__init__(
            path_to_results_file=path_to_results_file
        )

    def load(self, *args, **kwargs):
        super(TGRPESummary, self).load(*args, **kwargs)
        self.imrct_deviation = {}
        if "imrct_deviation" in self.data.keys():
            if len(self.data["imrct_deviation"]):
                from pesummary.utils.probability_dict import ProbabilityDict2D
                analysis_label = [
                    label.split(":inspiral")[0] for label in self.labels
                    if "inspiral" in label and "postinspiral" not in label
                ]
                self.imrct_deviation = {
                    label: ProbabilityDict2D(
                        {
                            "final_mass_final_spin_deviations": [
                                *self.data["imrct_deviation"][num]
                            ]
                        }
                    ) for num, label in enumerate(analysis_label)
                }
                if len(analysis_label) == 1:
                    self.imrct_deviation = self.imrct_deviation["inspiral"]
            else:
                self.imrct_deviation = {}

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        labels = list(dictionary.keys())
        if "version" in labels:
            labels.remove("version")

        history_dict = None
        if "history" in labels:
            history_dict = dictionary["history"]
            labels.remove("history")
        parameter_list, sample_list, imrct_deviation = [], [], []
        file_kwargs = {}
        _labels = []
        for num, label in enumerate(labels):
            if label == "version" or label == "history":
                continue
            data, = load_recursively(label, dictionary)
            posterior_samples = data["posterior_samples"]
            if "imrct" in data.keys():
                if "meta_data" in data["imrct"].keys():
                    _meta_data = data["imrct"]["meta_data"]
                else:
                    _meta_data = {}
                file_kwargs[label] = _meta_data
                for analysis in ["inspiral", "postinspiral"]:
                    if len(labels) > 1:
                        _labels.append("{}:{}".format(label, analysis))
                    else:
                        _labels.append(analysis)
                    parameters = [
                        j for j in posterior_samples[analysis].dtype.names
                    ]
                    samples = [
                        np.array(j.tolist()) for j in posterior_samples[analysis]
                    ]
                    if isinstance(parameters[0], bytes):
                        parameters = [
                            parameter.decode("utf-8") for parameter in parameters
                        ]
                    parameter_list.append(parameters)
                    sample_list.append(samples)
                imrct_deviation.append(
                    [
                        data["imrct"]["final_mass_deviation"],
                        data["imrct"]["final_spin_deviation"],
                        data["imrct"]["pdf"]
                    ]
                )
        labels = _labels
        return {
            "parameters": parameter_list,
            "samples": sample_list,
            "injection": None,
            "labels": labels,
            "history": history_dict,
            "imrct_deviation": imrct_deviation,
            "kwargs": file_kwargs
        }
