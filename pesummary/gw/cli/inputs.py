# Licensed under an MIT style license -- see LICENSE.md

import ast
import os
import numpy as np
import pesummary.core.cli.inputs
from pesummary.gw.file.read import read as GWRead
from pesummary.gw.file.psd import PSD
from pesummary.gw.file.calibration import Calibration
from pesummary.utils.decorators import deprecation
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import logger
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _GWInput(pesummary.core.cli.inputs._Input):
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
        _replace_kwargs = {
            "psd": "{file}.psd['{label}']"
        }
        if "psd_default" in kwargs.keys():
            _replace_kwargs["psd_default"] = kwargs["psd_default"]
        data = pesummary.core.cli.inputs._Input.grab_data_from_metafile(
            existing_file, webdir, compare=compare, read_function=GWRead,
            nsamples=nsamples, _replace_with_pesummary_kwargs=_replace_kwargs,
            **kwargs
        )
        f = GWRead(existing_file)

        labels = data["labels"]

        psd = {i: {} for i in labels}
        if f.psd is not None and f.psd != {}:
            for i in labels:
                if i in f.psd.keys() and f.psd[i] != {}:
                    psd[i] = {
                        ifo: PSD(f.psd[i][ifo]) for ifo in f.psd[i].keys()
                    }
        calibration = {i: {} for i in labels}
        if f.calibration is not None and f.calibration != {}:
            for i in labels:
                if i in f.calibration.keys() and f.calibration[i] != {}:
                    calibration[i] = {
                        ifo: Calibration(f.calibration[i][ifo]) for ifo in
                        f.calibration[i].keys()
                    }
        skymap = {i: None for i in labels}
        if hasattr(f, "skymap") and f.skymap is not None and f.skymap != {}:
            for i in labels:
                if i in f.skymap.keys() and len(f.skymap[i]):
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
        for _property in ["f_low", "f_ref", "f_final", "delta_f"]:
            if getattr(self, _property) is None:
                setattr(self, "_{}".format(_property), [None] * len(self.labels))
            elif len(getattr(self, _property)) == 1 and len(self.labels) != 1:
                setattr(
                    self, "_{}".format(_property),
                    getattr(self, _property) * len(self.labels)
                )
        if self.opts.approximant is None:
            approx = [None] * len(self.labels)
        else:
            approx = self.opts.approximant
        resume_file = [
            os.path.join(
                self.webdir, "checkpoint",
                "{}_conversion_class.pickle".format(label)
            ) for label in self.labels
        ]

        try:
            for num, label in enumerate(self.labels):
                try:
                    psd = self.psd[label]
                except KeyError:
                    psd = {}
                kwargs[label].update(dict(
                    evolve_spins_forwards=self.evolve_spins_forwards,
                    evolve_spins_backwards=self.evolve_spins_backwards,
                    f_low=self.f_low[num],
                    approximant=approx[num], f_ref=self.f_ref[num],
                    NRSur_fits=self.NRSur_fits, return_kwargs=True,
                    multipole_snr=self.calculate_multipole_snr,
                    precessing_snr=self.calculate_precessing_snr,
                    psd=psd, f_final=self.f_final[num],
                    waveform_fits=self.waveform_fits,
                    multi_process=self.opts.multi_process,
                    redshift_method=self.redshift_method,
                    cosmology=self.cosmology,
                    no_conversion=self.no_conversion,
                    add_zero_spin=True, delta_f=self.delta_f[num],
                    psd_default=self.psd_default,
                    disable_remnant=self.disable_remnant,
                    force_BBH_remnant_computation=self.force_BBH_remnant_computation,
                    resume_file=resume_file[num],
                    restart_from_checkpoint=self.restart_from_checkpoint,
                    force_BH_spin_evolution=self.force_BH_spin_evolution,
                ))
            return kwargs
        except IndexError:
            logger.warning(
                "Unable to find an f_ref, f_low and approximant for each "
                "label. Using and f_ref={}, f_low={} and approximant={} "
                "for all result files".format(
                    self.f_ref[0], self.f_low[0], approx[0]
                )
            )
            for num, label in enumerate(self.labels):
                kwargs[label].update(dict(
                    evolve_spins_forwards=self.evolve_spins_forwards,
                    evolve_spins_backwards=self.evolve_spins_backwards,
                    f_low=self.f_low[0],
                    approximant=approx[0], f_ref=self.f_ref[0],
                    NRSur_fits=self.NRSur_fits, return_kwargs=True,
                    multipole_snr=self.calculate_multipole_snr,
                    precessing_snr=self.calculate_precessing_snr,
                    psd=self.psd[self.labels[0]], f_final=self.f_final[0],
                    waveform_fits=self.waveform_fits,
                    multi_process=self.opts.multi_process,
                    redshift_method=self.redshift_method,
                    cosmology=self.cosmology,
                    no_conversion=self.no_conversion,
                    add_zero_spin=True, delta_f=self.delta_f[0],
                    psd_default=self.psd_default,
                    disable_remnant=self.disable_remnant,
                    force_BBH_remnant_computation=self.force_BBH_remnant_computation,
                    resume_file=resume_file[num],
                    restart_from_checkpoint=self.restart_from_checkpoint,
                    force_BH_spin_evolution=self.force_BH_spin_evolution
                ))
            return kwargs

    @staticmethod
    def grab_data_from_file(
        file, label, webdir, config=None, injection=None, file_format=None,
        nsamples=None, disable_prior_sampling=False, **kwargs
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
        data = pesummary.core.cli.inputs._Input.grab_data_from_file(
            file, label, webdir, config=config, injection=injection,
            read_function=GWRead, file_format=file_format, nsamples=nsamples,
            disable_prior_sampling=disable_prior_sampling, **kwargs
        )
        return data

    @property
    def reweight_samples(self):
        return self._reweight_samples

    @reweight_samples.setter
    def reweight_samples(self, reweight_samples):
        from pesummary.gw.reweight import options
        self._reweight_samples = self._check_reweight_samples(
            reweight_samples, options
        )

    def _set_samples(self, *args, **kwargs):
        super(_GWInput, self)._set_samples(*args, **kwargs)
        if "calibration" not in self.priors:
            self.priors["calibration"] = {
                label: {} for label in self.labels
            }

    def _set_corner_params(self, corner_params):
        corner_params = super(_GWInput, self)._set_corner_params(corner_params)
        if corner_params is None:
            logger.debug(
                "Using the default corner parameters: {}".format(
                    ", ".join(conf.gw_corner_parameters)
                )
            )
        else:
            _corner_params = corner_params
            corner_params = list(set(conf.gw_corner_parameters + corner_params))
            for param in _corner_params:
                _data = self.samples
                if not all(param in _data[label].keys() for label in self.labels):
                    corner_params.remove(param)
            logger.debug(
                "Generating a corner plot with the following "
                "parameters: {}".format(", ".join(corner_params))
            )
        return corner_params

    @property
    def cosmology(self):
        return self._cosmology

    @cosmology.setter
    def cosmology(self, cosmology):
        from pesummary.gw.cosmology import available_cosmologies

        if cosmology.lower() not in available_cosmologies:
            logger.warning(
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
                logger.warning(
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
                        logger.warning(
                            "No approximant passed. Waveform plots will not be "
                            "generated"
                        )
                    self._approximant[i] = None
                    break

    @property
    def gracedb_server(self):
        return self._gracedb_server

    @gracedb_server.setter
    def gracedb_server(self, gracedb_server):
        if gracedb_server is None:
            self._gracedb_server = conf.gracedb_server
        else:
            logger.debug(
                "Using '{}' as the GraceDB server".format(gracedb_server)
            )
            self._gracedb_server = gracedb_server

    @property
    def gracedb(self):
        return self._gracedb

    @gracedb.setter
    def gracedb(self, gracedb):
        self._gracedb = gracedb
        if gracedb is not None:
            from pesummary.gw.gracedb import get_gracedb_data, HTTPError
            from json.decoder import JSONDecodeError

            first_letter = gracedb[0]
            if first_letter != "G" and first_letter != "g" and first_letter != "S":
                logger.warning(
                    "Invalid GraceDB ID passed. The GraceDB ID must be of the "
                    "form G0000 or S0000. Ignoring input."
                )
                self._gracedb = None
                return
            _error = (
                "Unable to download data from Gracedb because {}. Only storing "
                "the GraceDB ID in the metafile"
            )
            try:
                logger.info(
                    "Downloading {} from gracedb for {}".format(
                        ", ".join(self.gracedb_data), gracedb
                    )
                )
                json = get_gracedb_data(
                    gracedb, info=self.gracedb_data,
                    service_url=self.gracedb_server
                )
                json["id"] = gracedb
            except (HTTPError, RuntimeError, JSONDecodeError) as e:
                logger.warning(_error.format(e))
                json = {"id": gracedb}

            for label in self.labels:
                self.file_kwargs[label]["meta_data"]["gracedb"] = json

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
                _opened = self._open_result_files
                if i in _opened.keys() and _opened[i] is not None:
                    f = self._open_result_files[i]
                else:
                    f = GWRead(i, disable_prior=True)
                try:
                    calibration_data = f.interpolate_calibration_spline_posterior()
                except Exception as e:
                    logger.warning(
                        "Failed to extract calibration data from the result "
                        "file: {} because {}".format(i, e)
                    )
                    calibration_data = None
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
                                    psd_data[ifo], IFO=ifo
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
                logger.warning(
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
        from pesummary.gw.file.strain import StrainDataDict

        self._gwdata = gwdata
        if gwdata is not None:
            if isinstance(gwdata, dict):
                for i in gwdata.keys():
                    if not os.path.isfile(gwdata[i]):
                        raise InputError(
                            "The file {} does not exist. Please check the path "
                            "to your strain file".format(gwdata[i])
                        )
                self._gwdata = StrainDataDict.read(gwdata)
            else:
                logger.warning(
                    "Please provide gwdata as a dictionary with keys "
                    "displaying the channel and item giving the path to the "
                    "strain file"
                )
                self._gwdata = None

    @property
    def evolve_spins_forwards(self):
        return self._evolve_spins_forwards

    @evolve_spins_forwards.setter
    def evolve_spins_forwards(self, evolve_spins_forwards):
        self._evolve_spins_forwards = evolve_spins_forwards
        _msg = "Spins will be evolved up to {}"
        if evolve_spins_forwards:
            logger.info(_msg.format("Schwarzschild ISCO frequency"))
            self._evolve_spins_forwards = 6. ** -0.5

    @property
    def evolve_spins_backwards(self):
        return self._evolve_spins_backwards

    @evolve_spins_backwards.setter
    def evolve_spins_backwards(self, evolve_spins_backwards):
        self._evolve_spins_backwards = evolve_spins_backwards
        _msg = (
            "Spins will be evolved backwards to an infinite separation using the '{}' "
            "method"
        )
        if isinstance(evolve_spins_backwards, (str, bytes)):
            logger.info(_msg.format(evolve_spins_backwards))
        elif evolve_spins_backwards is None:
            logger.info(_msg.format("precession_averaged"))
            self._evolve_spins_backwards = "precession_averaged"

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
            from pesummary.gw.conversions.nrutils import NRSUR_MODEL

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
    def f_final(self):
        return self._f_final

    @f_final.setter
    def f_final(self, f_final):
        self._f_final = f_final
        if f_final is not None:
            self._f_final = [float(i) for i in f_final]

    @property
    def delta_f(self):
        return self._delta_f

    @delta_f.setter
    def delta_f(self, delta_f):
        self._delta_f = delta_f
        if delta_f is not None:
            self._delta_f = [float(i) for i in delta_f]

    @property
    def psd_default(self):
        return self._psd_default

    @psd_default.setter
    def psd_default(self, psd_default):
        self._psd_default = psd_default
        if "stored:" in psd_default:
            label = psd_default.split("stored:")[1]
            self._psd_default = "{file}.psd['%s']" % (label)
            return
        try:
            from pycbc import psd
            psd_default = getattr(psd, psd_default)
        except ImportError:
            logger.warning(
                "Unable to import 'pycbc'. Unable to generate a default PSD"
            )
            psd_default = None
        except AttributeError:
            logger.warning(
                "'pycbc' does not have the '{}' psd available. Using '{}' as "
                "the default PSD".format(psd_default, conf.psd)
            )
            psd_default = getattr(psd, conf.psd)
        except ValueError as e:
            logger.warning("Setting 'psd_default' to None because {}".format(e))
            psd_default = None
        self._psd_default = psd_default

    @property
    def pepredicates_probs(self):
        return self._pepredicates_probs

    @pepredicates_probs.setter
    def pepredicates_probs(self, pepredicates_probs):
        from pesummary.gw.classification import PEPredicates

        classifications = {}
        for num, i in enumerate(list(self.samples.keys())):
            try:
                classifications[i] = PEPredicates(
                    self.samples[i]
                ).dual_classification()
            except Exception as e:
                logger.warning(
                    "Failed to generate source classification probabilities "
                    "because {}".format(e)
                )
                classifications[i] = None
        if self.mcmc_samples:
            if any(_probs is None for _probs in classifications.values()):
                classifications[self.labels[0]] = None
                logger.warning(
                    "Unable to average classification probabilities across "
                    "mcmc chains because one or more chains failed to estimate "
                    "classifications"
                )
            else:
                logger.debug(
                    "Averaging classification probability across mcmc samples"
                )
                classifications[self.labels[0]] = {
                    prior: {
                        key: np.round(np.mean(
                            [val[prior][key] for val in classifications.values()]
                        ), 3) for key in _probs.keys()
                    } for prior, _probs in
                    list(classifications.values())[0].items()
                }
        self._pepredicates_probs = classifications

    @property
    def pastro_probs(self):
        return self._pastro_probs

    @pastro_probs.setter
    def pastro_probs(self, pastro_probs):
        from pesummary.gw.classification import PAstro

        probabilities = {}
        for num, i in enumerate(list(self.samples.keys())):
            try:
                probabilities[i] = PAstro(self.samples[i]).dual_classification()
            except Exception as e:
                logger.warning(
                    "Failed to generate em_bright probabilities because "
                    "{}".format(e)
                )
                probabilities[i] = None
        if self.mcmc_samples:
            if any(_probs is None for _probs in probabilities.values()):
                probabilities[self.labels[0]] = None
                logger.warning(
                    "Unable to average em_bright probabilities across "
                    "mcmc chains because one or more chains failed to estimate "
                    "probabilities"
                )
            else:
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

    @property
    def preliminary_pages(self):
        return self._preliminary_pages

    @preliminary_pages.setter
    def preliminary_pages(self, preliminary_pages):
        required = conf.gw_reproducibility
        self._preliminary_pages = {label: False for label in self.labels}
        for num, label in enumerate(self.labels):
            for attr in required:
                _property = getattr(self, attr)
                if isinstance(_property, dict):
                    if label not in _property.keys():
                        self._preliminary_pages[label] = True
                    elif not len(_property[label]):
                        self._preliminary_pages[label] = True
                elif isinstance(_property, list):
                    if _property[num] is None:
                        self._preliminary_pages[label] = True
        if any(value for value in self._preliminary_pages.values()):
            _labels = [
                label for label, value in self._preliminary_pages.items() if
                value
            ]
            msg = (
                "Unable to reproduce the {} analys{} because no {} data was "
                "provided. 'Preliminary' watermarks will be added to the final "
                "html pages".format(
                    ", ".join(_labels), "es" if len(_labels) > 1 else "is",
                    " or ".join(required)
                )
            )
            logger.warning(msg)

    @staticmethod
    def _extract_IFO_data_from_file(file, cls, desc, IFO=None):
        """Return IFO data stored in a file

        Parameters
        ----------
        file: path
            path to a file containing the IFO data
        cls: obj
            class you wish to use when loading the file. This class must have
            a '.read' method
        desc: str
            description of the IFO data stored in the file
        IFO: str, optional
            the IFO which the data belongs to
        """
        general = (
            "Failed to read in %s data because {}. The %s plot will not be "
            "generated and the %s data will not be added to the metafile."
        ) % (desc, desc, desc)
        try:
            return cls.read(file, IFO=IFO)
        except FileNotFoundError:
            logger.warning(
                general.format("the file {} does not exist".format(file))
            )
            return {}
        except ValueError as e:
            logger.warning(general.format(e))
            return {}

    @staticmethod
    def extract_psd_data_from_file(file, IFO=None):
        """Return the data stored in a psd file

        Parameters
        ----------
        file: path
            path to a file containing the psd data
        """
        from pesummary.gw.file.psd import PSD
        return _GWInput._extract_IFO_data_from_file(file, PSD, "PSD", IFO=IFO)

    @staticmethod
    def extract_calibration_data_from_file(file, **kwargs):
        """Return the data stored in a calibration file

        Parameters
        ----------
        file: path
            path to a file containing the calibration data
        """
        from pesummary.gw.file.calibration import Calibration
        return _GWInput._extract_IFO_data_from_file(
            file, Calibration, "calibration", **kwargs
        )

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
                    i: executable(input[i][idx], IFO=i) for i in list(keys)
                }
        elif isinstance(input, dict):
            for i in self.labels:
                data[i] = {
                    j: executable(input[j], IFO=j) for j in list(input.keys())
                }
        elif isinstance(input, list):
            for i in self.labels:
                data[i] = {
                    self.get_ifo_from_file_name(j): executable(
                        j, IFO=self.get_ifo_from_file_name(j)
                    ) for j in input
                }
        else:
            raise InputError(
                "Did not understand the psd/calibration input. Please use the "
                "following format 'H1:path/to/file'"
            )
        return data

    def grab_priors_from_inputs(self, priors):
        def read_func(data, **kwargs):
            from pesummary.gw.file.read import read as GWRead
            data = GWRead(data, **kwargs)
            data.generate_all_posterior_samples()
            return data

        return super(_GWInput, self).grab_priors_from_inputs(
            priors, read_func=read_func, read_kwargs=self.grab_data_kwargs
        )

    def grab_key_data_from_result_files(self):
        """Grab the mean, median, maxL and standard deviation for all
        parameters for all each result file
        """
        from pesummary.utils.kde_list import KDEList
        from pesummary.gw.plots.plot import _return_bounds
        from pesummary.utils.credible_interval import (
            hpd_two_sided_credible_interval
        )
        from pesummary.utils.bounded_1d_kde import bounded_1d_kde
        key_data = super(_GWInput, self).grab_key_data_from_result_files()
        bounded_parameters = ["mass_ratio", "a_1", "a_2", "lambda_tilde"]
        for param in bounded_parameters:
            xlow, xhigh = _return_bounds(param, [])
            _samples = {
                key: val[param] for key, val in self.samples.items()
                if param in val.keys()
            }
            _min = [np.min(_) for _ in _samples.values() if len(_samples)]
            _max = [np.max(_) for _ in _samples.values() if len(_samples)]
            if not len(_min):
                continue
            _min = np.min(_min)
            _max = np.max(_max)
            x = np.linspace(_min, _max, 1000)
            try:
                kdes = KDEList(
                    list(_samples.values()), kde=bounded_1d_kde,
                    kde_kwargs={"xlow": xlow, "xhigh": xhigh}
                )
            except Exception as e:
                logger.warning(
                    "Unable to compute the HPD interval for {} because {}".format(
                        param, e
                    )
                )
                continue
            pdfs = kdes(x)
            for num, key in enumerate(_samples.keys()):
                [xlow, xhigh], _ = hpd_two_sided_credible_interval(
                    [], 90, x=x, pdf=pdfs[num]
                )
                key_data[key][param]["90% HPD"] = [xlow, xhigh]
                for _param in self.samples[key].keys():
                    if _param in bounded_parameters:
                        continue
                    key_data[key][_param]["90% HPD"] = float("nan")
        return key_data


class SamplesInput(_GWInput, pesummary.core.cli.inputs.SamplesInput):
    """Class to handle and store sample specific command line arguments
    """
    def __init__(self, *args, **kwargs):
        kwargs.update({"ignore_copy": True})
        super(SamplesInput, self).__init__(
            *args, gw=True, extra_options=[
                "evolve_spins_forwards",
                "evolve_spins_backwards",
                "NRSur_fits",
                "calculate_multipole_snr",
                "calculate_precessing_snr",
                "f_low",
                "f_ref",
                "f_final",
                "psd",
                "waveform_fits",
                "redshift_method",
                "cosmology",
                "no_conversion",
                "delta_f",
                "psd_default",
                "disable_remnant",
                "force_BBH_remnant_computation",
                "force_BH_spin_evolution"
            ], **kwargs
        )
        if self._restarted_from_checkpoint:
            return
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
        self.detectors = None
        self.skymap = None
        self.calibration = self.opts.calibration
        self.gwdata = self.opts.gwdata
        self.maxL_samples = []

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


class PlottingInput(SamplesInput, pesummary.core.cli.inputs.PlottingInput):
    """Class to handle and store plottig specific command line arguments
    """
    def __init__(self, *args, **kwargs):
        super(PlottingInput, self).__init__(*args, **kwargs)
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
        self.preliminary_pages = None
        self.pepredicates_probs = []
        self.pastro_probs = []


class WebpageInput(SamplesInput, pesummary.core.cli.inputs.WebpageInput):
    """Class to handle and store webpage specific command line arguments
    """
    def __init__(self, *args, **kwargs):
        super(WebpageInput, self).__init__(*args, **kwargs)
        self.gracedb_server = self.opts.gracedb_server
        self.gracedb_data = self.opts.gracedb_data
        self.gracedb = self.opts.gracedb
        self.public = self.opts.public
        if not hasattr(self, "preliminary_pages"):
            self.preliminary_pages = None
        if not hasattr(self, "pepredicates_probs"):
            self.pepredicates_probs = []
        if not hasattr(self, "pastro_probs"):
            self.pastro_probs = []


class WebpagePlusPlottingInput(PlottingInput, WebpageInput):
    """Class to handle and store webpage and plotting specific command line
    arguments
    """
    def __init__(self, *args, **kwargs):
        super(WebpagePlusPlottingInput, self).__init__(*args, **kwargs)

    @property
    def default_directories(self):
        return super(WebpagePlusPlottingInput, self).default_directories

    @property
    def default_files_to_copy(self):
        return super(WebpagePlusPlottingInput, self).default_files_to_copy


class MetaFileInput(SamplesInput, pesummary.core.cli.inputs.MetaFileInput):
    """Class to handle and store metafile specific command line arguments
    """
    @property
    def default_directories(self):
        dirs = super(MetaFileInput, self).default_directories
        dirs += ["psds", "calibration"]
        return dirs

    def copy_files(self):
        _error = "Failed to save the {} to file"
        for label in self.labels:
            if self.psd[label] != {}:
                for ifo in self.psd[label].keys():
                    if not isinstance(self.psd[label][ifo], PSD):
                        logger.warning(_error.format("{} PSD".format(ifo)))
                        continue
                    self.psd[label][ifo].save_to_file(
                        os.path.join(self.webdir, "psds", "{}_{}_psd.dat".format(
                            label, ifo
                        ))
                    )
            if label in self.priors["calibration"].keys():
                if self.priors["calibration"][label] != {}:
                    for ifo in self.priors["calibration"][label].keys():
                        _instance = isinstance(
                            self.priors["calibration"][label][ifo], Calibration
                        )
                        if not _instance:
                            logger.warning(
                                _error.format(
                                    "{} calibration envelope".format(
                                        ifo
                                    )
                                )
                            )
                            continue
                        self.priors["calibration"][label][ifo].save_to_file(
                            os.path.join(self.webdir, "calibration", "{}_{}_cal.txt".format(
                                label, ifo
                            ))
                        )
        return super(MetaFileInput, self).copy_files()


class WebpagePlusPlottingPlusMetaFileInput(MetaFileInput, WebpagePlusPlottingInput):
    """Class to handle and store webpage, plotting and metafile specific command
    line arguments
    """
    def __init__(self, *args, **kwargs):
        super(WebpagePlusPlottingPlusMetaFileInput, self).__init__(
            *args, **kwargs
        )

    @property
    def default_directories(self):
        return super(WebpagePlusPlottingPlusMetaFileInput, self).default_directories

    @property
    def default_files_to_copy(self):
        return super(WebpagePlusPlottingPlusMetaFileInput, self).default_files_to_copy


@deprecation(
    "The GWInput class is deprecated. Please use either the BaseInput, "
    "SamplesInput, PlottingInput, WebpageInput, WebpagePlusPlottingInput, "
    "MetaFileInput or the WebpagePlusPlottingPlusMetaFileInput class"
)
class GWInput(WebpagePlusPlottingPlusMetaFileInput):
    pass


class IMRCTInput(pesummary.core.cli.inputs._Input):
    """Class to handle the TGR specific command line arguments
    """
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        if len(labels) % 2 != 0:
            raise ValueError(
                "The IMRCT test requires 2 results files for each analysis. "
            )
        elif len(labels) > 2:
            cond = all(
                ":inspiral" in label or ":postinspiral" in label for label in
                labels
            )
            if not cond:
                raise ValueError(
                    "To compare 2 or more analyses, please provide labels as "
                    "'{}:inspiral' and '{}:postinspiral' where {} indicates "
                    "the analysis label"
                )
            else:
                self.analysis_label = [
                    label.split(":inspiral")[0]
                    for label in labels
                    if ":inspiral" in label and ":postinspiral" not in label
                ]
                if len(self.analysis_label) != len(self.result_files) / 2:
                    raise ValueError(
                        "When comparing more than 2 analyses, labels must "
                        "be of the form '{}:inspiral' and '{}:postinspiral'."
                    )
                logger.info(
                    "Using the labels: {} to distinguish analyses".format(
                        ", ".join(self.analysis_label)
                    )
                )
        elif sorted(labels) != ["inspiral", "postinspiral"]:
            if all(self.is_pesummary_metafile(ff) for ff in self.result_files):
                meta_file_labels = []
                for suffix in [":inspiral", ":postinspiral"]:
                    if any(suffix in label for label in labels):
                        ind = [
                            num for num, label in enumerate(labels) if
                            suffix in label
                        ]
                        if len(ind) > 1:
                            raise ValueError(
                                "Please provide a single {} label".format(
                                    suffix.split(":")[1]
                                )
                            )
                        meta_file_labels.append(
                            labels[ind[0]].split(suffix)[0]
                        )
                    else:
                        raise ValueError(
                            "Please provide labels as {inspiral_label}:inspiral "
                            "and {postinspiral_label}:postinspiral where "
                            "inspiral_label and postinspiral_label are the "
                            "PESummary labels for the inspiral and postinspiral "
                            "analyses respectively. "
                        )
                if len(self.result_files) == 1:
                    logger.info(
                        "Using the {} samples for the inspiral analysis and {} "
                        "samples for the postinspiral analysis from the file "
                        "{}".format(
                            meta_file_labels[0], meta_file_labels[1],
                            self.result_files[0]
                        )
                    )
                elif len(self.result_files) == 2:
                    logger.info(
                        "Using the {} samples for the inspiral analysis from "
                        "the file {}. Using the {} samples for the "
                        "postinspiral analysis from the file {}".format(
                            meta_file_labels[0], self.result_files[0],
                            meta_file_labels[1], self.result_files[1]
                        )
                    )
                else:
                    raise ValueError(
                        "Currently, you can only provide at most 2 pesummary "
                        "metafiles. If one is provided, both the inspiral and "
                        "postinspiral are extracted from that single file. If "
                        "two are provided, the inspiral is extracted from one "
                        "file and the postinspiral is extracted from the other."
                    )
                self._labels = ["inspiral", "postinspiral"]
                self._meta_file_labels = meta_file_labels
                self.analysis_label = ["primary"]
            else:
                raise ValueError(
                    "The IMRCT test requires an inspiral and postinspiral result "
                    "file. Please indicate which file is the inspiral and which "
                    "is postinspiral by providing these exact labels to the "
                    "summarytgr executable"
                )
        else:
            self.analysis_label = ["primary"]

    def _extract_stored_approximant(self, opened_file, label):
        """Extract the approximant used for a given analysis stored in a
        PESummary metafile

        Parameters
        ----------
        opened_file: pesummary.gw.file.formats.pesummary.PESummary
            opened metafile that contains the analysis 'label'
        label: str
            analysis label which is stored in the PESummary metafile
        """
        if opened_file.approximant is not None:
            if label not in opened_file.labels:
                raise ValueError(
                    "Invalid label {}. The list of available labels are {}".format(
                        label, ", ".join(opened_file.labels)
                    )
                )
            _index = opened_file.labels.index(label)
            return opened_file.approximant[_index]
        return

    def _extract_stored_remnant_fits(self, opened_file, label):
        """Extract the remnant fits used for a given analysis stored in a
        PESummary metafile

        Parameters
        ----------
        opened_file: pesummary.gw.file.formats.pesummary.PESummary
            opened metafile that contains the analysis 'label'
        label: str
            analysis label which is stored in the PESummary metafile
        """
        fits = {}
        fit_strings = [
            "final_spin_NR_fits", "final_mass_NR_fits"
        ]
        if label not in opened_file.labels:
            raise ValueError(
                "Invalid label {}. The list of available labels are {}".format(
                    label, ", ".join(opened_file.labels)
                )
            )
        _index = opened_file.labels.index(label)
        _meta_data = opened_file.extra_kwargs[_index]
        if "meta_data" in _meta_data.keys():
            for key in fit_strings:
                if key in _meta_data["meta_data"].keys():
                    fits[key] = _meta_data["meta_data"][key]
        if len(fits):
            return fits
        return

    def _extract_stored_cutoff_frequency(self, opened_file, label):
        """Extract the cutoff frequencies used for a given analysis stored in a
        PESummary metafile

        Parameters
        ----------
        opened_file: pesummary.gw.file.formats.pesummary.PESummary
            opened metafile that contains the analysis 'label'
        label: str
            analysis label which is stored in the PESummary metafile
        """
        frequencies = {}
        if opened_file.config is not None:
            if label not in opened_file.labels:
                raise ValueError(
                    "Invalid label {}. The list of available labels are {}".format(
                        label, ", ".join(opened_file.labels)
                    )
                )
            if opened_file.config[label] is not None:
                _config = opened_file.config[label]
                if "config" in _config.keys():
                    if "maximum-frequency" in _config["config"].keys():
                        frequencies["fhigh"] = _config["config"][
                            "maximum-frequency"
                        ]
                    if "minimum-frequency" in _config["config"].keys():
                        frequencies["flow"] = _config["config"][
                            "minimum-frequency"
                        ]
                elif "lalinference" in _config.keys():
                    if "fhigh" in _config["lalinference"].keys():
                        frequencies["fhigh"] = _config["lalinference"][
                            "fhigh"
                        ]
                    if "flow" in _config["lalinference"].keys():
                        frequencies["flow"] = _config["lalinference"][
                            "flow"
                        ]
            return frequencies
        return

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
        self._read_samples = {
            _label: GWRead(_path, disable_prior=True) for _label, _path in zip(
                self.labels, self.result_files
            )
        }
        _samples_dict = {}
        _approximant_dict = {}
        _cutoff_frequency_dict = {}
        _remnant_fits_dict = {}
        for label, _open in self._read_samples.items():
            if isinstance(_open.samples_dict, MultiAnalysisSamplesDict):
                if not len(self._meta_file_labels):
                    raise ValueError(
                        "Currently you can only pass a file containing a "
                        "single analysis or a valid PESummary metafile "
                        "containing multiple analyses"
                    )
                _labels = _open.labels
                if len(self._read_samples) == 1:
                    _samples_dict = {
                        label: _open.samples_dict[meta_file_label] for
                        label, meta_file_label in zip(
                            self.labels, self._meta_file_labels
                        )
                    }
                    for label, meta_file_label in zip(self.labels, self._meta_file_labels):
                        _stored_approx = self._extract_stored_approximant(
                            _open, meta_file_label
                        )
                        _stored_frequencies = self._extract_stored_cutoff_frequency(
                            _open, meta_file_label
                        )
                        _stored_remnant_fits = self._extract_stored_remnant_fits(
                            _open, meta_file_label
                        )
                        if _stored_approx is not None:
                            _approximant_dict[label] = _stored_approx
                        if _stored_remnant_fits is not None:
                            _remnant_fits_dict[label] = _stored_remnant_fits
                        if _stored_frequencies is not None:
                            if label == "inspiral":
                                if "fhigh" in _stored_frequencies.keys():
                                    _cutoff_frequency_dict[label] = _stored_frequencies[
                                        "fhigh"
                                    ]
                            if label == "postinspiral":
                                if "flow" in _stored_frequencies.keys():
                                    _cutoff_frequency_dict[label] = _stored_frequencies[
                                        "flow"
                                    ]
                    break
                else:
                    ind = self.labels.index(label)
                    _samples_dict[label] = _open.samples_dict[
                        self._meta_file_labels[ind]
                    ]
                    _stored_approx = self._extract_stored_approximant(
                        _open, self._meta_file_labels[ind]
                    )
                    _stored_frequencies = self._extract_stored_cutoff_frequency(
                        _open, self._meta_file_labels[ind]
                    )
                    _stored_remnant_fits = self._extract_stored_remnant_fits(
                        _open, self._meta_file_labels[ind]
                    )
                    if _stored_approx is not None:
                        _approximant_dict[label] = _stored_approx
                    if _stored_remnant_fits is not None:
                        _remnant_fits_dict[label] = _stored_remnant_fits
                    if _stored_frequencies is not None:
                        if label == "inspiral":
                            if "fhigh" in _stored_frequencies.keys():
                                _cutoff_frequency_dict[label] = _stored_frequencies[
                                    "fhigh"
                                ]
                        if label == "postinspiral":
                            if "flow" in _stored_frequencies.keys():
                                _cutoff_frequency_dict[label] = _stored_frequencies[
                                    "flow"
                                ]
            else:
                _samples_dict[label] = _open.samples_dict
                extra_kwargs = _open.extra_kwargs
                if "pe_algorithm" in extra_kwargs["sampler"].keys():
                    if extra_kwargs["sampler"]["pe_algorithm"] == "bilby":
                        try:
                            subkwargs = extra_kwargs["other"]["likelihood"][
                                "waveform_arguments"
                            ]
                            _approximant_dict[label] = (
                                subkwargs["waveform_approximant"]
                            )
                            if "inspiral" in label and "postinspiral" not in label:
                                _cutoff_frequency_dict[label] = (
                                    subkwargs["maximum_frequency"]
                                )
                            elif "postinspiral" in label:
                                _cutoff_frequency_dict[label] = (
                                    subkwargs["minimum_frequency"]
                                )
                        except KeyError:
                            pass
        self._samples = MultiAnalysisSamplesDict(_samples_dict)
        if len(_approximant_dict):
            self._approximant_dict = _approximant_dict
        if len(_cutoff_frequency_dict):
            self._cutoff_frequency_dict = _cutoff_frequency_dict
        if len(_remnant_fits_dict):
            self._remnant_fits_dict = _remnant_fits_dict

    @property
    def imrct_kwargs(self):
        return self._imrct_kwargs

    @imrct_kwargs.setter
    def imrct_kwargs(self, imrct_kwargs):
        test_kwargs = dict(N_bins=101)
        try:
            test_kwargs.update(imrct_kwargs)
        except AttributeError:
            test_kwargs = test_kwargs

        for key, value in test_kwargs.items():
            try:
                test_kwargs[key] = ast.literal_eval(value)
            except ValueError:
                pass
        self._imrct_kwargs = test_kwargs

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data):
        self._meta_data = {}
        for num, _inspiral in enumerate(self.inspiral_keys):
            frequency_dict = dict()
            approximant_dict = dict()
            remnant_dict = dict()
            zipped = zip(
                [self.cutoff_frequency, self.approximant, None],
                [frequency_dict, approximant_dict, remnant_dict],
                ["cutoff_frequency", "approximant", "remnant_fits"]
            )
            _inspiral_string = self.inspiral_keys[num]
            _postinspiral_string = self.postinspiral_keys[num]
            for _list, _dict, name in zipped:
                if _list is not None and len(_list) == len(self.labels):
                    inspiral_ind = self.labels.index(_inspiral_string)
                    postinspiral_ind = self.labels.index(_postinspiral_string)
                    _dict["inspiral"] = _list[inspiral_ind]
                    _dict["postinspiral"] = _list[postinspiral_ind]
                elif _list is not None:
                    raise ValueError(
                        "Please provide a 'cutoff_frequency' and 'approximant' "
                        "for each file"
                    )
                else:
                    try:
                        if name == "cutoff_frequency":
                            cond = (
                                "inspiral" in self._cutoff_frequency_dict.keys()
                                and "postinspiral" not in
                                self._cutoff_frequency_dict.keys()
                            )
                            if cond:
                                _dict["inspiral"] = self._cutoff_frequency_dict[
                                    "inspiral"
                                ]
                            elif "postinspiral" in self._cutoff_frequency_dict.keys():
                                _dict["postinspiral"] = self._cutoff_frequency_dict[
                                    "postinspiral"
                                ]
                        elif name == "approximant":
                            cond = (
                                "inspiral" in self._approximant_dict.keys()
                                and "postinspiral" not in
                                self._approximant_dict.keys()
                            )
                            if cond:
                                _dict["inspiral"] = self._approximant_dict[
                                    "inspiral"
                                ]
                            elif "postinspiral" in self._approximant_dict.keys():
                                _dict["postinspiral"] = self._approximant_dict[
                                    "postinspiral"
                                ]
                        elif name == "remnant_fits":
                            cond = (
                                "inspiral" in self._remnant_fits_dict.keys()
                                and "postinspiral" not in
                                self._remnant_fits_dict.keys()
                            )
                            if cond:
                                _dict["inspiral"] = self._remnant_fits_dict[
                                    "inspiral"
                                ]
                            elif "postinspiral" in self._remnant_fits_dict.keys():
                                _dict["postinspiral"] = self._remnant_fits_dict[
                                    "postinspiral"
                                ]
                    except (AttributeError, KeyError, TypeError):
                        _dict["inspiral"] = None
                        _dict["postinspiral"] = None

            self._meta_data[self.analysis_label[num]] = {
                "inspiral maximum frequency (Hz)": frequency_dict["inspiral"],
                "postinspiral minimum frequency (Hz)": frequency_dict["postinspiral"],
                "inspiral approximant": approximant_dict["inspiral"],
                "postinspiral approximant": approximant_dict["postinspiral"],
                "inspiral remnant fits": remnant_dict["inspiral"],
                "postinspiral remnant fits": remnant_dict["postinspiral"]
            }

    def __init__(self, opts):
        self.opts = opts
        self.existing = None
        self.webdir = self.opts.webdir
        self.user = None
        self.baseurl = None
        self.result_files = self.opts.samples
        self.labels = self.opts.labels
        self.samples = self.opts.samples
        self.inspiral_keys = [
            key for key in self.samples.keys() if "inspiral" in key
            and "postinspiral" not in key
        ]
        self.postinspiral_keys = [
            key.replace("inspiral", "postinspiral") for key in self.inspiral_keys
        ]
        try:
            self.imrct_kwargs = self.opts.imrct_kwargs
        except AttributeError:
            self.imrct_kwargs = {}
        for _arg in ["cutoff_frequency", "approximant", "links_to_pe_pages", "f_low"]:
            _attr = getattr(self.opts, _arg)
            if _attr is not None and len(_attr) and len(_attr) != len(self.labels):
                raise ValueError("Please provide a {} for each file".format(_arg))
            setattr(self, _arg, _attr)
        self.meta_data = None
        self.default_directories = ["samples", "plots", "js", "html", "css"]
        self.publication = False
        self.make_directories()
