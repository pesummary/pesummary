# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org> This program is free
# software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os

import numpy as np
import pesummary
from pesummary.core.inputs import Input, PostProcessing
from pesummary.gw.file.read import read as GWRead
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import logger, SamplesDict


class GWInput(Input):
    """Super class to handle gw specific command line inputs

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options

    Attributes
    ----------
    result_files: list
        list of result files passed
    compare_results: list
        list of labels stored in the metafile that you wish to compare
    add_to_existing: Bool
        True if we are adding to an existing web directory
    existing_samples: dict
        dictionary of samples stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_injection_data: dict
        dictionary of injection data stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_file_version: dict
        dictionary of file versions stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_config: list
        list of configuration files stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_labels: list
        list of labels stored in an existing metafile. None if
        `self.add_to_existing` is False
    user: str
        the user who submitted the job
    webdir: str
        the directory to store the webpages, plots and metafile produced
    baseurl: str
        the base url of the webpages
    labels: list
        list of labels used to distinguish the result files
    config: list
        list of configuration files for each result file
    injection_file: list
        list of injection files for each result file
    publication: Bool
        if true, publication quality plots are generated. Default False
    kde_plot: Bool
        if true, kde plots are generated instead of histograms. Default False
    samples: dict
        dictionary of posterior samples stored in the result files
    priors: dict
        dictionary of prior samples stored in the result files
    custom_plotting: list
        list containing the directory and name of python file which contains
        custom plotting functions. Default None
    email: str
        the email address of the user
    dump: Bool
        if True, all plots will be dumped onto a single html page. Default False
    hdf5: Bool
        if True, the metafile is stored in hdf5 format. Default False
    approximant: dict
        dictionary of approximants used in the analysis
    gracedb: str
        the gracedb ID for the event
    detectors: list
        the detector network used for each result file
    calibration: dict
        dictionary containing the posterior calibration envelopes for each IFO
        for each result file
    psd: dict
        dictionary containing the psd used for each IFO for each result file
    nsamples_for_skymap: int
        the number of samples to use for the skymap
    sensitivity: Bool
        if True, the sky sensitivity for HL and HLV detector networks are also
        plotted. Default False
    no_ligo_skymap: Bool
        if True, a skymap will not be generated with the ligo.skymap package.
        Default False
    multi_threading_for_skymap: Bool
        if True, multi-threading will be used to speed up skymap generation
    gwdata: dict
        dictionary containing the strain timeseries used for each result file
    notes: str
        notes that you wish to add to the webpages
    """
    def __init__(self, opts):
        logger.info("Command line arguments: %s" % (opts))
        self.opts = opts
        self.result_files = self.opts.samples
        self.meta_file = False
        if self.result_files is not None and len(self.result_files) == 1:
            self.meta_file = self.is_pesummary_metafile(self.result_files[0])
        self.existing = self.opts.existing
        self.compare_results = self.opts.compare_results
        self.add_to_existing = False
        if self.existing is not None:
            self.add_to_existing = True
            self.existing_metafile = None
            self.existing_data = self.grab_data_from_metafile(
                self.existing_metafile, self.existing,
                compare=self.compare_results
            )
            self.existing_samples = self.existing_data[0]
            self.existing_injection_data = self.existing_data[1]
            self.existing_file_version = self.existing_data[2]
            self.existing_file_kwargs = self.existing_data[3]
            self.existing_priors = self.existing_data[4]
            self.existing_config = self.existing_data[5]
            self.existing_labels = self.existing_data[6]
            self.existing_approximant = self.existing_data[7]
            self.existing_psd = self.existing_data[8]
            self.existing_calibration = self.existing_data[9]
        else:
            self.existing_labels = None
            self.existing_samples = None
            self.existing_file_version = None
            self.existing_file_kwargs = None
            self.existing_priors = None
            self.existing_config = None
            self.existing_injection_data = None
            self.existing_approximant = None
            self.existing_psd = None
            self.existing_calibration = None
        self.user = self.opts.user
        self.webdir = self.opts.webdir
        self.baseurl = self.opts.baseurl
        self.labels = self.opts.labels
        self.weights = {i: None for i in self.labels}
        self.config = self.opts.config
        self.injection_file = self.opts.inj_file
        self.publication = self.opts.publication
        self.make_directories()
        self.kde_plot = self.opts.kde_plot
        self.priors = self.opts.prior_file
        self.samples = self.opts.samples
        self.burnin = self.opts.burnin
        self.custom_plotting = self.opts.custom_plotting
        self.email = self.opts.email
        self.dump = self.opts.dump
        self.hdf5 = self.opts.save_to_hdf5
        self.palette = self.opts.palette
        self.include_prior = self.opts.include_prior
        self.colors = None
        self.approximant = self.opts.approximant
        self.gracedb = self.opts.gracedb
        self.detectors = None
        self.calibration = self.opts.calibration
        self.psd = self.opts.psd
        self.nsamples_for_skymap = self.opts.nsamples_for_skymap
        self.sensitivity = self.opts.sensitivity
        self.no_ligo_skymap = self.opts.no_ligo_skymap
        self.multi_threading_for_skymap = self.opts.multi_threading_for_skymap
        self.gwdata = self.opts.gwdata
        self.notes = self.opts.notes
        self.pepredicates_probs = []
        self.pastro_probs = []
        self.copy_files()

    @staticmethod
    def grab_data_from_metafile(existing_file, webdir, compare=None):
        """Grab data from an existing PESummary metafile

        Parameters
        ----------
        existing_file: str
            path to the existing metafile
        webdir: str
            the directory to store the existing configuration file
        compare: list, optional
            list of labels for events stored in an existing metafile that you
            wish to compare
        """
        f = GWRead(existing_file)
        labels = f.labels
        indicies = np.arange(len(labels))

        if compare:
            indicies = []
            for i in compare:
                if i not in labels:
                    raise InputError(
                        "Label '%s' does not exist in the metafile. The list "
                        "of available labels are %s" % (i, labels)
                    )
                indicies.append(labels.index(i))
            labels = compare

        DataFrame = f.samples_dict
        if f.injection_parameters != []:
            inj_values = f.injection_dict
        else:
            inj_values = {
                i: [float("nan")] * len(DataFrame[i]) for i in labels
            }
        for i in inj_values.keys():
            for param in inj_values[i].keys():
                if inj_values[i][param] == "nan":
                    inj_values[i][param] = float("nan")

        if hasattr(f, "priors") and f.priors != {}:
            priors = f.priors["samples"]
            if "calibration" in f.priors.keys():
                priors["calibration"] = f.priors["calibration"]
        else:
            priors = {label: {} for label in labels}

        config = []
        if f.config is not None and not all(i is None for i in f.config):
            config = []
            for i in labels:
                config_dir = os.path.join(webdir, "config")
                f.write_config_to_file(i, outdir=config_dir)
                config_file = os.path.join(
                    config_dir, "{}_config.ini".format(i)
                )
                config.append(config_file)
        else:
            for i in labels:
                config.append(None)

        psd = {}
        if f.psd is not None and f.psd[labels[0]] != {}:
            for i in labels:
                psd[i] = {
                    ifo: f.psd[i][ifo] for ifo in f.psd[i].keys()
                }
        calibration = {}
        if f.calibration is not None and f.calibration[labels[0]] != {}:
            for i in labels:
                calibration[i] = {
                    ifo: f.calibration[i][ifo] for ifo in f.calibration[i].keys()
                }

        if f.weights is not None:
            weights = {i: f.weights[i] for i in labels}
        else:
            weights = {i: None for i in labels}

        return [
            DataFrame, inj_values,
            {
                i: j for i, j in zip(
                    labels, [f.input_version[ind] for ind in indicies]
                )
            },
            {
                i: j for i, j in zip(
                    labels, [f.extra_kwargs[ind] for ind in indicies]
                )
            }, priors, config, labels, weights,
            {
                i: j for i, j in zip(
                    labels, [f.approximant[ind] for ind in indicies]
                )
            }, psd, calibration
        ]

    @staticmethod
    def grab_data_from_file(file, label, config=None, injection=None):
        """Grab data from a result file containing posterior samples

        Parameters
        ----------
        file: str
            path to the result file
        label: str
            label that you wish to use for the result file
        config: str, optional
            path to a configuration file used in the analysis
        injection: str, optional
            path to an injection file used in the analysis
        """
        f = GWRead(file)
        if config is not None:
            f.add_fixed_parameters_from_config_file(config)
        f.generate_all_posterior_samples()
        if injection:
            f.add_injection_parameters_from_file(injection)
        parameters = f.parameters
        samples = np.array(f.samples).T
        DataFrame = {label: SamplesDict(parameters, samples)}
        kwargs = f.extra_kwargs
        if hasattr(f, "injection_parameters"):
            injection = f.injection_parameters
            if injection is not None:
                for i in parameters:
                    if i not in list(injection.keys()):
                        injection[i] = float("nan")
            else:
                injection = {i: j for i, j in zip(
                    parameters, [float("nan")] * len(parameters))}
        else:
            injection = {i: j for i, j in zip(
                parameters, [float("nan")] * len(parameters))}
        version = f.input_version
        if hasattr(f, "priors"):
            priors = f.priors
        else:
            priors = []
        if hasattr(f, "weights"):
            weights = f.weights
        else:
            weights = None
        return [
            DataFrame, {label: injection}, {label: version}, {label: kwargs},
            {label: priors}, {label: weights}
        ]

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        if config and len(config) != len(self.labels):
            raise InputError(
                "Please provide a configuration file for each label"
            )
        if config is None and not self.meta_file:
            self._config = [None] * len(self.labels)
        elif self.meta_file:
            self._config = [None] * len(self.labels)
        else:
            self._config = config

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if not samples:
            raise InputError("Please provide a results file")
        if len(samples) != len(self.labels):
            logger.info(
                "You have passed {} result files and {} labels. Setting "
                "labels = {}".format(
                    len(samples), len(self.labels), self.labels[:len(samples)]
                )
            )
            self.labels = self.labels[:len(samples)]
        samples_dict, injection_data_dict, prior_dict = {}, {}, {}
        file_version_dict, file_kwargs_dict, weights_dict = {}, {}, {}
        approximant_dict, psd_dict, calibration_dict = {}, {}, {}
        config, labels = None, None
        for num, i in enumerate(samples):
            if not os.path.isfile(i):
                raise Exception("File %s does not exist" % (i))
            data = self.grab_data_from_input(
                i, self.labels[num], config=self.config[num],
                injection=self.injection_file[num]
            )
            if len(data) > 6:
                config = data[5]
                labels = data[6]
                weights_dict = data[7]
                prior_dict = data[4]
                approximant_dict = data[8]
                psd_dict = data[9]
                calibration_dict = data[10]
                for j in labels:
                    samples_dict[j] = data[0][j]
                    injection_data_dict[j] = data[1][j]
                    file_version_dict[j] = data[2][j]
                    file_kwargs_dict[j] = data[3][j]
            else:
                samples_dict[self.labels[num]] = data[0][self.labels[num]]
                injection_data_dict[self.labels[num]] = data[1][self.labels[num]]
                file_version_dict[self.labels[num]] = data[2][self.labels[num]]
                file_kwargs_dict[self.labels[num]] = data[3][self.labels[num]]
                prior_dict[self.labels[num]] = data[4][self.labels[num]]
                weights_dict[self.labels[num]] = data[5][self.labels[num]]
        self._samples = samples_dict
        self._injection_data = injection_data_dict
        self._file_version = file_version_dict
        self._file_kwargs = file_kwargs_dict
        if labels is not None:
            labels_iter = labels
        else:
            labels_iter = self.labels
        for i in labels_iter:
            if prior_dict != {} and prior_dict[i] != []:
                if self.priors != {} and i in self.priors["samples"].keys():
                    logger.warn(
                        "Replacing the prior file for {} with the prior "
                        "samples stored in the result file".format(i)
                    )
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors != {}:
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
            elif prior_dict != {}:
                if self.priors != {} and i not in self.priors["samples"].keys():
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, [])
            else:
                if self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, [])
        if config is not None:
            self._config = config
        if labels is not None:
            self.labels = labels
            self.result_files = self.result_files * len(labels)
            self.weights = {i: None for i in self.labels}
        if approximant_dict != {}:
            self._approximant = approximant_dict
        if psd_dict != {}:
            self._psd = psd_dict
        if calibration_dict != {}:
            self._calibration = calibration_dict
        if weights_dict != {}:
            self.weights = weights_dict

    @property
    def approximant(self):
        return self._approximant

    @approximant.setter
    def approximant(self, approximant):
        if not hasattr(self, "_approximant"):
            approximant_list = {i: {} for i in self.labels}
            if approximant is None:
                logger.warn(
                    "No approximant passed. Waveform plots will not be "
                    "generated"
                )
            elif approximant is not None:
                if len(approximant) != len(self.labels):
                    raise InputError(
                        "Please pass an approximant for each result file"
                    )
                approximant_list = {
                    i: j for i, j in zip(self.labels, approximant)
                }
            self._approximant = approximant_list
        else:
            for num, i in enumerate(self._approximant.keys()):
                if self._approximant[i] == {}:
                    if num == 0:
                        logger.warn(
                            "No approximant passed. Waveform plots will not be "
                            "generated"
                        )
                    self._approximant[i] = None
                    break

    @property
    def gracedb(self):
        return self._gracedb

    @gracedb.setter
    def gracedb(self, gracedb):
        self._gracedb = gracedb
        if gracedb is not None:
            first_letter = gracedb[0]
            if first_letter != "G" and first_letter != "g" and first_letter != "S":
                raise InputError(
                    "Invalid GraceDB ID passed. The GraceDB ID must be of the "
                    "form G0000 or S0000"
                )

    @property
    def detectors(self):
        return self._detectors

    @detectors.setter
    def detectors(self, detectors):
        detector = {}
        if not detectors:
            for i in self.labels:
                params = list(self.samples[i].keys())
                individual_detectors = []
                for j in params:
                    if "optimal_snr" in j and j != "network_optimal_snr":
                        det = j.split("_optimal_snr")[0]
                        individual_detectors.append(det)
                individual_detectors = sorted(
                    [str(i) for i in individual_detectors])
                if individual_detectors:
                    detector[i] = "_".join(individual_detectors)
                else:
                    detector[i] = None
        else:
            detector = detectors
        logger.debug("The detector network is %s" % (detector))
        self._detectors = detector

    @property
    def calibration(self):
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if not hasattr(self, "._calibration"):
            data = {i: {} for i in self.labels}
            if calibration is not {}:
                prior_data = self.get_psd_or_calibration_data(
                    calibration, self.extract_calibration_data_from_file
                )
                self.add_to_prior_dict("calibration", prior_data)
            for num, i in enumerate(self.result_files):
                f = GWRead(i)
                calibration_data = f.calibration_data_in_results_file
                if calibration_data is None:
                    data[self.labels[num]] = {
                        None: None
                    }
                elif isinstance(f, pesummary.gw.file.formats.pesummary.PESummary):
                    for num in range(len(calibration_data[0])):
                        data[self.labels[num]] = {
                            j: k for j, k in zip(
                                calibration_data[1][num],
                                calibration_data[0][num]
                            )
                        }
                else:
                    data[self.labels[num]] = {
                        j: k for j, k in zip(
                            calibration_data[1], calibration_data[0]
                        )
                    }
            self._calibration = data

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, psd):
        if not hasattr(self, "_psd"):
            data = {i: {} for i in self.labels}
            if psd != {}:
                data = self.get_psd_or_calibration_data(
                    psd, self.extract_psd_data_from_file
                )
            self._psd = data

    @property
    def nsamples_for_skymap(self):
        return self._nsamples_for_skymap

    @nsamples_for_skymap.setter
    def nsamples_for_skymap(self, nsamples_for_skymap):
        self._nsamples_for_skymap = nsamples_for_skymap
        if nsamples_for_skymap is not None:
            self._nsamples_for_skymap = int(nsamples_for_skymap)
            number_of_samples = [
                data.number_of_samples for label, data in self.samples.items()
            ]
            if not all(i > self._nsamples_for_skymap for i in number_of_samples):
                min_arg = np.argmin(number_of_samples)
                raise InputError(
                    "You have specified that you would like to use {} "
                    "samples to generate the skymap but the file {} only "
                    "has {} samples. Please reduce the number of samples "
                    "you wish to use for the skymap production".format(
                        self._nsamples_for_skymap, self.result_files[min_arg],
                        number_of_samples[min_arg]
                    )
                )

    @property
    def gwdata(self):
        return self._gwdata

    @gwdata.setter
    def gwdata(self, gwdata):
        from pesummary.gw.file.formats.base_read import GWRead as StrainFile

        self._gwdata = gwdata
        if gwdata is not None:
            for i in gwdata.keys():
                if not os.path.isfile(gwdata[i]):
                    raise InputError(
                        "The file {} does not exist. Please check the path to "
                        "your strain file".format(gwdata[i])
                    )
            timeseries = StrainFile.load_strain_data(gwdata)
            self._gwdata = timeseries

    @property
    def pepredicates_probs(self):
        return self._pepredicates_probs

    @pepredicates_probs.setter
    def pepredicates_probs(self, pepredicates_probs):
        from pesummary.gw.pepredicates import get_classifications

        classifications = {}
        for num, i in enumerate(self.labels):
            classifications[i] = get_classifications(self.samples[i])
        self._pepredicates_probs = classifications

    @property
    def pastro_probs(self):
        return self._pastro_probs

    @pastro_probs.setter
    def pastro_probs(self, pastro_probs):
        from pesummary.gw.p_astro import get_probabilities

        probabilities = {}
        for num, i in enumerate(self.labels):
            probabilities[i] = get_probabilities(self.samples[i])
        self._pastro_probs = probabilities

    @staticmethod
    def extract_psd_data_from_file(file):
        """Return the data stored in a psd file

        Parameters
        ----------
        file: path
            path to a file containing the psd data
        """
        if not os.path.isfile(file):
            raise InputError("The file '{}' does not exist".format(file))
        try:
            f = np.genfromtxt(file, skip_footer=2)
            return f
        except Exception as e:
            logger.info(
                "Failed to read in PSD data because {}. The PSD plot will not "
                "be generated and the PSD data will not be added to the "
                "metafile".format(e)
            )
            return {}

    @staticmethod
    def extract_calibration_data_from_file(file):
        """Return the data stored in a calibration file

        Parameters
        ----------
        file: path
            path to a file containing the calibration data
        """
        if not os.path.isfile(file):
            raise InputError("The file '{}' does not exist".format(file))
        try:
            f = np.genfromtxt(file)
            return f
        except Exception as e:
            logger.info(
                "Failed to read in calibration data because {}. The "
                "calibration plot will not be generated and the calibration "
                "data will not be added to the metafile".format(e)
            )
            return {}

    @staticmethod
    def get_ifo_from_file_name(file):
        """Return the IFO from the file name

        Parameters
        ----------
        file: str
            path to the file
        """
        file_name = file.split("/")[-1]
        if any(j in file_name for j in ["H1", "_0", "IFO0"]):
            ifo = "H1"
        elif any(j in file_name for j in ["L1", "_1", "IFO1"]):
            ifo = "L1"
        elif any(j in file_name for j in ["V1", "_2", "IFO2"]):
            ifo = "V1"
        else:
            ifo = file_name
        return ifo

    def get_psd_or_calibration_data(self, input, executable):
        """Return a dictionary containing the psd or calibration data

        Parameters
        ----------
        input: list/dict
            list/dict containing paths to calibration/psd files
        executable: func
            executable that is used to extract the data from the calibration/psd
            files
        """
        data = {}
        if input == {} or input == []:
            return data
        if isinstance(input, dict):
            keys = list(input.keys())
        if isinstance(input, dict) and isinstance(input[keys[0]], list):
            if not all(len(input[i]) != len(self.labels) for i in list(keys)):
                raise InputError(
                    "Please ensure the number of calibration/psd files matches "
                    "the number of result files passed"
                )
            for idx in range(len(input[keys[0]])):
                data[self.labels[idx]] = {
                    i: executable(input[i][idx]) for i in list(keys)
                }
        elif isinstance(input, dict):
            for i in self.labels:
                data[i] = {
                    j: executable(input[j]) for j in list(input.keys())
                }
        elif isinstance(input, list):
            for i in self.labels:
                data[i] = {
                    self.get_ifo_from_file_name(j): executable(j) for j in input
                }
        else:
            raise InputError(
                "Did not understand the psd/calibration input. Please use the "
                "following format 'H1:path/to/file'"
            )
        return data

    def grab_priors_from_inputs(self, priors):
        """
        """
        from pesummary.gw.file.read import read as GWRead

        prior_dict = {}
        if priors is not None:
            prior_dict = {"samples": {}}
            for i in priors:
                if not os.path.isfile(i):
                    raise InputError("The file {} does not exist".format(i))
            if len(priors) != len(self.labels) and len(priors) == 1:
                logger.warn(
                    "You have only specified a single prior file for {} result "
                    "files. Assuming the same prior file for all result "
                    "files".format(len(self.labels))
                )
                data = GWRead(priors[0])
                data.generate_all_posterior_samples()
                for i in self.labels:
                    prior_dict["samples"][i] = data.samples_dict
            elif len(priors) != len(self.labels):
                raise InputError(
                    "Please provide a prior file for each result file"
                )
            else:
                for num, i in enumerate(priors):
                    data = GWRead(priors[0])
                    data.generate_all_posterior_samples()
                    prior_dict["samples"][self.labels[num]] = data.samples_dict
        return prior_dict


class GWPostProcessing(PostProcessing):
    """Super class to post process the input data

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files

    Attributes
    ----------
    result_files: list
        list of result files passed
    compare_results: list
        list of labels stored in the metafile that you wish to compare
    add_to_existing: Bool
        True if we are adding to an existing web directory
    existing_samples: dict
        dictionary of samples stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_injection_data: dict
        dictionary of injection data stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_file_version: dict
        dictionary of file versions stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_config: list
        list of configuration files stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_labels: list
        list of labels stored in an existing metafile. None if
        `self.add_to_existing` is False
    user: str
        the user who submitted the job
    webdir: str
        the directory to store the webpages, plots and metafile produced
    baseurl: str
        the base url of the webpages
    labels: list
        list of labels used to distinguish the result files
    config: list
        list of configuration files for each result file
    injection_file: list
        list of injection files for each result file
    publication: Bool
        if true, publication quality plots are generated. Default False
    kde_plot: Bool
        if true, kde plots are generated instead of histograms. Default False
    samples: dict
        dictionary of posterior samples stored in the result files
    priors: dict
        dictionary of prior samples stored in the result files
    custom_plotting: list
        list containing the directory and name of python file which contains
        custom plotting functions. Default None
    email: str
        the email address of the user
    dump: Bool
        if True, all plots will be dumped onto a single html page. Default False
    hdf5: Bool
        if True, the metafile is stored in hdf5 format. Default False
    approximant: dict
        dictionary of approximants used in the analysis
    gracedb: str
        the gracedb ID for the event
    detectors: list
        the detector network used for each result file
    calibration: dict
        dictionary containing the posterior calibration envelopes for each IFO
        for each result file
    psd: dict
        dictionary containing the psd used for each IFO for each result file
    nsamples_for_skymap: int
        the number of samples to use for the skymap
    sensitivity: Bool
        if True, the sky sensitivity for HL and HLV detector networks are also
        plotted. Default False
    no_ligo_skymap: Bool
        if True, a skymap will not be generated with the ligo.skymap package.
        Default False
    multi_threading_for_skymap: Bool
        if True, multi-threading will be used to speed up skymap generation
    gwdata: dict
        dictionary containing the strain timeseries used for each result file
    maxL_samples: dict
        dictionary containing the maximum likelihood values for each parameter
        for each result file
    same_parameters: list
        list of parameters that are common in all result files
    pepredicates_probs: dict
        dictionary containing the source classification probabilities for each
        result file
    """
    def __init__(self, inputs, colors="default"):
        self.inputs = inputs
        self.result_files = self.inputs.result_files
        self.existing = self.inputs.existing
        self.compare_results = self.inputs.compare_results
        self.add_to_existing = False
        if self.existing is not None:
            self.add_to_existing = True
            self.existing_metafile = self.inputs.existing_metafile
            self.existing_samples = self.inputs.existing_samples
            self.existing_injection_data = self.inputs.existing_injection_data
            self.existing_file_version = self.inputs.existing_file_version
            self.existing_file_kwargs = self.inputs.existing_file_kwargs
            self.existing_priors = self.inputs.existing_priors
            self.existing_config = self.inputs.existing_config
            self.existing_labels = self.inputs.existing_labels
            self.existing_approximant = self.inputs.existing_approximant
            self.existing_psd = self.inputs.existing_psd
            self.existing_calibration = self.inputs.existing_calibration
        else:
            self.existing_metafile = None
            self.existing_labels = None
            self.existing_samples = None
            self.existing_file_version = None
            self.existing_file_kwargs = None
            self.existing_priors = None
            self.existing_config = None
            self.existing_injection_data = None
            self.existing_approximant = None
            self.existing_psd = None
            self.existing_calibration = None
        self.user = self.inputs.user
        self.host = self.inputs.host
        self.webdir = self.inputs.webdir
        self.baseurl = self.inputs.baseurl
        self.labels = self.inputs.labels
        self.weights = self.inputs.weights
        self.config = self.inputs.config
        self.injection_file = self.inputs.injection_file
        self.injection_data = self.inputs.injection_data
        self.publication = self.inputs.publication
        self.kde_plot = self.inputs.kde_plot
        self.samples = self.inputs.samples
        self.priors = self.inputs.priors
        self.custom_plotting = self.inputs.custom_plotting
        self.email = self.inputs.email
        self.dump = self.inputs.dump
        self.hdf5 = self.inputs.hdf5
        self.file_version = self.inputs.file_version
        self.file_kwargs = self.inputs.file_kwargs
        self.palette = self.inputs.palette
        self.approximant = self.inputs.approximant
        self.gracedb = self.inputs.gracedb
        self.detectors = self.inputs.detectors
        self.calibration = self.inputs.calibration
        self.psd = self.inputs.psd
        self.nsamples_for_skymap = self.inputs.nsamples_for_skymap
        self.sensitivity = self.inputs.sensitivity
        self.no_ligo_skymap = self.inputs.no_ligo_skymap
        self.multi_threading_for_skymap = self.inputs.multi_threading_for_skymap
        self.gwdata = self.inputs.gwdata
        self.colors = self.inputs.colors
        self.include_prior = self.inputs.include_prior
        self.notes = self.inputs.notes
        self.maxL_samples = []
        self.same_parameters = []
        self.pepredicates_probs = self.inputs.pepredicates_probs
        self.pastro_probs = self.inputs.pastro_probs

    @property
    def maxL_samples(self):
        return self._maxL_samples

    @maxL_samples.setter
    def maxL_samples(self, maxL_samples):
        key_data = self.grab_key_data_from_result_files()
        maxL_samples = {
            i: {
                j: key_data[i][j]["maxL"] for j in key_data[i].keys()
            } for i in key_data.keys()
        }
        for i in self.labels:
            maxL_samples[i]["approximant"] = self.approximant[i]
        self._maxL_samples = maxL_samples

    def grab_key_data_from_result_files(self):
        """Grab the mean, median, maxL and standard deviation for all
        parameters for all each result file
        """
        key_data = {
            i: {
                j: {
                    "mean": self.samples[i][j].average("mean"),
                    "median": self.samples[i][j].average("median"),
                    "std": self.samples[i][j].standard_deviation,
                    "maxL": self.samples[i][j].maxL
                } for j in self.samples[i].keys()
            } for i in self.labels
        }
        return key_data
