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
from pesummary.core.inputs import _Input, Input, PostProcessing
from pesummary.gw.file.read import read as GWRead
from pesummary.gw.file.psd import PSD
from pesummary.gw.file.calibration import Calibration
from pesummary.utils.exceptions import InputError
from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.utils import logger


class _GWInput(_Input):
    """Super class to handle gw specific command line inputs
    """
    @staticmethod
    def grab_data_from_metafile(
        existing_file, webdir, compare=None, nsamples=None, **kwargs
    ):
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
        data = _Input.grab_data_from_metafile(
            existing_file, webdir, compare=compare, read_function=GWRead,
            nsamples=nsamples, **kwargs
        )
        f = GWRead(existing_file)

        labels = data["labels"]

        psd = {i: {} for i in labels}
        if f.psd is not None and f.psd != {}:
            for i in labels:
                if f.psd[i] != {}:
                    psd[i] = {
                        ifo: PSD(f.psd[i][ifo]) for ifo in f.psd[i].keys()
                    }
        calibration = {i: {} for i in labels}
        if f.calibration is not None and f.calibration != {}:
            for i in labels:
                if f.calibration[i] != {}:
                    calibration[i] = {
                        ifo: Calibration(f.calibration[i][ifo]) for ifo in
                        f.calibration[i].keys()
                    }
        skymap = {i: None for i in labels}
        if hasattr(f, "skymap") and f.skymap is not None and f.skymap != {}:
            for i in labels:
                if len(f.skymap[i]):
                    skymap[i] = f.skymap[i]
        data.update(
            {
                "approximant": {
                    i: j for i, j in zip(
                        labels, [f.approximant[ind] for ind in data["indicies"]]
                    )
                },
                "psd": psd,
                "calibration": calibration,
                "skymap": skymap
            }
        )
        return data

    @property
    def grab_data_kwargs(self):
        kwargs = super(_GWInput, self).grab_data_kwargs
        if self.f_low is None:
            self._f_low = [None] * len(self.labels)
        if self.f_ref is None:
            self._f_ref = [None] * len(self.labels)
        if self.opts.approximant is None:
            approx = [None] * len(self.labels)
        else:
            approx = self.opts.approximant
        try:
            for num, label in enumerate(self.labels):
                kwargs[label].update(dict(
                    evolve_spins=self.evolve_spins, f_low=self.f_low[num],
                    approximant=approx[num], f_ref=self.f_ref[num],
                    NRSur_fits=self.NRSur_fits, return_kwargs=True,
                    waveform_fits=self.waveform_fits,
                    multi_process=self.opts.multi_process,
                    redshift_method=self.redshift_method,
                    cosmology=self.cosmology,
                    no_conversion=self.no_conversion,
                    add_zero_spin=True
                ))
            return kwargs
        except IndexError:
            logger.warn(
                "Unable to find an f_ref, f_low and approximant for each "
                "label. Using and f_ref={}, f_low={} and approximant={} "
                "for all result files".format(
                    self.f_ref[0], self.f_low[0], approx[0]
                )
            )
            for num, label in enumerate(self.labels):
                kwargs[label].update(dict(
                    evolve_spins=self.evolve_spins, f_low=self.f_low[0],
                    approximant=approx[0], f_ref=self.f_ref[0],
                    NRSur_fits=self.NRSur_fits, return_kwargs=True,
                    waveform_fits=self.waveform_fits,
                    multi_process=self.opts.multi_process,
                    redshift_method=self.redshift_method,
                    cosmology=self.cosmology,
                    no_conversion=self.no_conversion,
                    add_zero_spin=True
                ))
            return kwargs

    @staticmethod
    def grab_data_from_file(
        file, label, config=None, injection=None, file_format=None,
        nsamples=None, **kwargs
    ):
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
        file_format, str, optional
            the file format you wish to use when loading. Default None.
            If None, the read function loops through all possible options
        """
        data = _Input.grab_data_from_file(
            file, label, config=config, injection=injection,
            read_function=GWRead, file_format=file_format, nsamples=nsamples,
            **kwargs
        )
        return data

    def _set_samples(self, *args, **kwargs):
        super(_GWInput, self)._set_samples(*args, **kwargs)
        if "calibration" not in self.priors:
            self.priors["calibration"] = {
                label: {} for label in self.labels
            }

    @property
    def cosmology(self):
        return self._cosmology

    @cosmology.setter
    def cosmology(self, cosmology):
        from pesummary.gw.cosmology import available_cosmologies
        from pesummary import conf

        if cosmology.lower() not in available_cosmologies:
            logger.warn(
                "Unrecognised cosmology: {}. Using {} as default".format(
                    cosmology, conf.cosmology
                )
            )
            cosmology = conf.cosmology
        else:
            logger.debug("Using the {} cosmology".format(cosmology))
        self._cosmology = cosmology

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
            for label in self.labels:
                self.file_kwargs[label]["meta_data"]["gracedb"] = gracedb

    @property
    def detectors(self):
        return self._detectors

    @detectors.setter
    def detectors(self, detectors):
        detector = {}
        if not detectors:
            for i in self.samples.keys():
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
    def skymap(self):
        return self._skymap

    @skymap.setter
    def skymap(self, skymap):
        if not hasattr(self, "_skymap"):
            self._skymap = {i: None for i in self.labels}

    @property
    def calibration(self):
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if not hasattr(self, "_calibration"):
            data = {i: {} for i in self.labels}
            if calibration != {}:
                prior_data = self.get_psd_or_calibration_data(
                    calibration, self.extract_calibration_data_from_file
                )
                self.add_to_prior_dict("calibration", prior_data)
            else:
                prior_data = {i: {} for i in self.labels}
                for label in self.labels:
                    if hasattr(self.opts, "{}_calibration".format(label)):
                        cal_data = getattr(self.opts, "{}_calibration".format(label))
                        if cal_data != {} and cal_data is not None:
                            prior_data[label] = {
                                ifo: self.extract_calibration_data_from_file(
                                    cal_data[ifo]
                                ) for ifo in cal_data.keys()
                            }
                if not all(prior_data[i] == {} for i in self.labels):
                    self.add_to_prior_dict("calibration", prior_data)
                else:
                    self.add_to_prior_dict("calibration", {})
            for num, i in enumerate(self.result_files):
                f = GWRead(i)
                calibration_data = f.calibration_data_in_results_file
                labels = list(self.samples.keys())
                if calibration_data is None:
                    data[labels[num]] = {
                        None: None
                    }
                elif isinstance(f, pesummary.gw.file.formats.pesummary.PESummary):
                    for num in range(len(calibration_data[0])):
                        data[labels[num]] = {
                            j: k for j, k in zip(
                                calibration_data[1][num],
                                calibration_data[0][num]
                            )
                        }
                else:
                    data[labels[num]] = {
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
            else:
                for label in self.labels:
                    if hasattr(self.opts, "{}_psd".format(label)):
                        psd_data = getattr(self.opts, "{}_psd".format(label))
                        if psd_data != {} and psd_data is not None:
                            data[label] = {
                                ifo: self.extract_psd_data_from_file(
                                    psd_data[ifo]
                                ) for ifo in psd_data.keys()
                            }
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
                logger.warn(
                    "You have specified that you would like to use {} "
                    "samples to generate the skymap but the file {} only "
                    "has {} samples. Reducing the number of samples to "
                    "generate the skymap to {}".format(
                        self._nsamples_for_skymap, self.result_files[min_arg],
                        number_of_samples[min_arg], number_of_samples[min_arg]
                    )
                )
                self._nsamples_for_skymap = int(number_of_samples[min_arg])

    @property
    def gwdata(self):
        return self._gwdata

    @gwdata.setter
    def gwdata(self, gwdata):
        from pesummary.gw.file.formats.base_read import GWRead as StrainFile

        self._gwdata = gwdata
        if gwdata is not None:
            if isinstance(gwdata, dict):
                for i in gwdata.keys():
                    if not os.path.isfile(gwdata[i]):
                        raise InputError(
                            "The file {} does not exist. Please check the path "
                            "to your strain file".format(gwdata[i])
                        )
            else:
                if len(gwdata) > 1:
                    logger.warn(
                        "Multiple files passed. Only using {}".format(
                            gwdata[0]
                        )
                    )
                if not os.path.isfile(gwdata[0]):
                    raise InputError(
                        "The file {} does not exist. Please check the path "
                        "to your strain file".format(gwdata[0])
                    )
                gwdata = gwdata[0]
            timeseries = StrainFile.load_strain_data(gwdata)
            self._gwdata = timeseries

    @property
    def evolve_spins(self):
        return self._evolve_spins

    @evolve_spins.setter
    def evolve_spins(self, evolve_spins):
        self._evolve_spins = evolve_spins
        if evolve_spins:
            logger.info(
                "Spins will be evolved up to the Schwarzschild ISCO frequency"
            )
            self._evolve_spins = 6. ** -0.5

    @property
    def NRSur_fits(self):
        return self._NRSur_fits

    @NRSur_fits.setter
    def NRSur_fits(self, NRSur_fits):
        self._NRSur_fits = NRSur_fits
        base = (
            "Using the '{}' NRSurrogate model to calculate the remnant "
            "quantities"
        )
        if isinstance(NRSur_fits, (str, bytes)):
            logger.info(base.format(NRSur_fits))
            self._NRSur_fits = NRSur_fits
        elif NRSur_fits is None:
            from pesummary.gw.file.nrutils import NRSUR_MODEL

            logger.info(base.format(NRSUR_MODEL))
            self._NRSur_fits = NRSUR_MODEL

    @property
    def waveform_fits(self):
        return self._waveform_fits

    @waveform_fits.setter
    def waveform_fits(self, waveform_fits):
        self._waveform_fits = waveform_fits
        if waveform_fits:
            logger.info(
                "Evaluating the remnant quantities using the provided "
                "approximant"
            )

    @property
    def f_low(self):
        return self._f_low

    @f_low.setter
    def f_low(self, f_low):
        self._f_low = f_low
        if f_low is not None:
            self._f_low = [float(i) for i in f_low]

    @property
    def f_ref(self):
        return self._f_ref

    @f_ref.setter
    def f_ref(self, f_ref):
        self._f_ref = f_ref
        if f_ref is not None:
            self._f_ref = [float(i) for i in f_ref]

    @property
    def pepredicates_probs(self):
        return self._pepredicates_probs

    @pepredicates_probs.setter
    def pepredicates_probs(self, pepredicates_probs):
        from pesummary.gw.pepredicates import get_classifications

        classifications = {}
        for num, i in enumerate(list(self.samples.keys())):
            classifications[i] = get_classifications(self.samples[i])
        if self.mcmc_samples:
            logger.debug(
                "Averaging classification probability across mcmc samples"
            )
            classifications[self.labels[0]] = {
                prior: {
                    key: np.round(np.mean(
                        [val[prior][key] for val in classifications.values()]
                    ), 3) for key in _probs.keys()
                } for prior, _probs in list(classifications.values())[0].items()
            }
        self._pepredicates_probs = classifications

    @property
    def pastro_probs(self):
        return self._pastro_probs

    @pastro_probs.setter
    def pastro_probs(self, pastro_probs):
        from pesummary.gw.p_astro import get_probabilities

        probabilities = {}
        for num, i in enumerate(list(self.samples.keys())):
            em_bright = get_probabilities(self.samples[i])
            if em_bright is not None:
                probabilities[i] = {
                    "default": em_bright[0], "population": em_bright[1]
                }
            else:
                probabilities[i] = None
        if self.mcmc_samples:
            logger.debug(
                "Averaging em_bright probability across mcmc samples"
            )
            probabilities[self.labels[0]] = {
                prior: {
                    key: np.round(np.mean(
                        [val[prior][key] for val in probabilities.values()]
                    ), 3) for key in _probs.keys()
                } for prior, _probs in list(probabilities.values())[0].items()
            }
        self._pastro_probs = probabilities

    @staticmethod
    def extract_psd_data_from_file(file):
        """Return the data stored in a psd file

        Parameters
        ----------
        file: path
            path to a file containing the psd data
        """
        from pesummary.gw.file.psd import PSD

        general = (
            "Failed to read in PSD data because {}. The PSD plot will be "
            "generated and the PSD data will not be added to the metafile."
        )
        try:
            f = np.genfromtxt(file)
            return PSD(f)
        except ValueError:
            f = np.genfromtxt(file, skip_footer=2)
            return PSD(f)
        except FileNotFoundError:
            logger.info(
                general.format("the file {} does not exist".format(file))
            )
            return {}
        except ValueError as e:
            logger.info(general.format(e))
            return {}

    @staticmethod
    def extract_calibration_data_from_file(file):
        """Return the data stored in a calibration file

        Parameters
        ----------
        file: path
            path to a file containing the calibration data
        """
        general = (
            "Failed to read in calibration data because {}. The calibration "
            "plot will not be generated and the calibration data will not be "
            "added to the metafile"
        )
        from pesummary.gw.file.calibration import Calibration

        try:
            f = np.genfromtxt(file)
            return Calibration(f)
        except FileNotFoundError:
            logger.info(
                general.format("the file {} does not exist".format(file))
            )
            return {}
        except ValueError as e:
            logger.info(general.format(e))
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
            if not all(len(input[i]) == len(self.labels) for i in list(keys)):
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
                    logger.info(
                        "Assigning {} to {}".format(self.labels[num], i)
                    )
                    if self.labels[num] in self.grab_data_kwargs.keys():
                        grab_data_kwargs = self.grab_data_kwargs[
                            self.labels[num]
                        ]
                    else:
                        grab_data_kwargs = self.grab_data_kwargs
                    data = GWRead(priors[num])
                    data.generate_all_posterior_samples(**grab_data_kwargs)
                    prior_dict["samples"][self.labels[num]] = data.samples_dict
        return prior_dict


class GWInput(_GWInput, Input):
    """Class to handle gw specific command line inputs

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
    disable_comparison: Bool
        if True, comparison plots and pages are not produced
    disable_interactive: Bool
        if True, interactive plots are not produced
    public: Bool
        if True, public facing summarypages are produced
    """
    def __init__(self, opts):
        super(GWInput, self).__init__(
            opts, ignore_copy=True, extra_options=[
                "evolve_spins", "NRSur_fits", "f_low", "f_ref",
                "waveform_fits", "redshift_method", "cosmology",
                "no_conversion"
            ]
        )
        if self.existing is not None:
            self.existing_data = self.grab_data_from_metafile(
                self.existing_metafile, self.existing,
                compare=self.compare_results
            )
            self.existing_approximant = self.existing_data["approximant"]
            self.existing_psd = self.existing_data["psd"]
            self.existing_calibration = self.existing_data["calibration"]
            self.existing_skymap = self.existing_data["skymap"]
        else:
            self.existing_approximant = None
            self.existing_psd = None
            self.existing_calibration = None
            self.existing_skymap = None
        self.approximant = self.opts.approximant
        self.gracedb = self.opts.gracedb
        self.detectors = None
        self.skymap = None
        self.calibration = self.opts.calibration
        self.psd = self.opts.psd
        self.nsamples_for_skymap = self.opts.nsamples_for_skymap
        self.sensitivity = self.opts.sensitivity
        self.no_ligo_skymap = self.opts.no_ligo_skymap
        self.multi_threading_for_skymap = self.multi_process
        if not self.no_ligo_skymap and self.multi_process > 1:
            total = self.multi_process
            self.multi_threading_for_plots = int(total / 2.)
            self.multi_threading_for_skymap = total - self.multi_threading_for_plots
            logger.info(
                "Assigning {} process{}to skymap generation and {} process{}to "
                "other plots".format(
                    self.multi_threading_for_skymap,
                    "es " if self.multi_threading_for_skymap > 1 else " ",
                    self.multi_threading_for_plots,
                    "es " if self.multi_threading_for_plots > 1 else " "
                )
            )
        self.gwdata = self.opts.gwdata
        self.public = self.opts.public
        self.pepredicates_probs = []
        self.pastro_probs = []
        self.copy_files()

    def copy_files(self):
        """Copy the relevant file to the web directory
        """
        for label in self.labels:
            if self.psd[label] != {}:
                for ifo in self.psd[label].keys():
                    self.psd[label][ifo].save_to_file(
                        os.path.join(self.webdir, "psds", "{}_{}_psd.dat".format(
                            label, ifo
                        ))
                    )
            if label in self.priors["calibration"].keys():
                if self.priors["calibration"][label] != {}:
                    for ifo in self.priors["calibration"][label].keys():
                        self.priors["calibration"][label][ifo].save_to_file(
                            os.path.join(self.webdir, "calibration", "{}_{}_cal.txt".format(
                                label, ifo
                            ))
                        )
        self._copy_files(self.default_files_to_copy)

    def make_directories(self):
        """Make the directories to store the information
        """
        for dirs in ["psds", "calibration"]:
            self.default_directories.append(dirs)
        if self.publication:
            self.default_directories.append("plots/publication")
        self._make_directories(self.webdir, self.default_directories)


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
    disable_comparison: bool
        Whether to make comparison webpage
    public: Bool
        if True, public facing summarypages are produced
    """
    def __init__(self, inputs, colors="default"):
        super(GWPostProcessing, self).__init__(inputs, colors=colors)
        if self.existing is not None:
            self.existing_approximant = self.inputs.existing_approximant
            self.existing_psd = self.inputs.existing_psd
            self.existing_calibration = self.inputs.existing_calibration
            self.existing_skymap = self.inputs.existing_skymap
        else:
            self.existing_approximant = None
            self.existing_psd = None
            self.existing_calibration = None
            self.existing_skymap = None
        self.publication_kwargs = self.inputs.publication_kwargs
        self.approximant = self.inputs.approximant
        self.gracedb = self.inputs.gracedb
        self.detectors = self.inputs.detectors
        self.skymap = self.inputs.skymap
        self.calibration = self.inputs.calibration
        self.psd = self.inputs.psd
        self.nsamples_for_skymap = self.inputs.nsamples_for_skymap
        self.sensitivity = self.inputs.sensitivity
        self.no_ligo_skymap = self.inputs.no_ligo_skymap
        self.multi_threading_for_skymap = self.inputs.multi_threading_for_skymap
        self.gwdata = self.inputs.gwdata
        self.maxL_samples = []
        self.same_parameters = []
        self.pepredicates_probs = self.inputs.pepredicates_probs
        self.pastro_probs = self.inputs.pastro_probs
        self.public = self.inputs.public

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
        def smart_average(data, _type="mean", multiple=False):
            if not multiple and _type == "std":
                return data.standard_deviation
            elif not multiple and _type == "maxL":
                return data.maxL
            elif not multiple:
                return data.average(_type)
            else:
                _data = np.concatenate(list(data.values()))
                if _type == "mean":
                    return np.mean(_data)
                elif _type == "median":
                    return np.median(_data)
                elif _type == "std":
                    return np.std(_data)
                else:
                    return float("nan")

        key_data = {
            key: {
                j: {
                    _type: smart_average(
                        val[j], multiple=self.mcmc_samples, _type=_type
                    ) for _type in ["mean", "median", "std", "maxL"]
                } for j in val.keys()
            } for key, val in self.samples.items()
        }
        return key_data
