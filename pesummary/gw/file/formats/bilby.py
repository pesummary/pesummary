# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from pesummary.core.file.formats.bilby import Bilby as CoreBilby
from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_bilby(
    path, disable_prior=False, latex_dict=GWlatex_labels,
    complex_params=["matched_filter_snr", "optimal_snr"], **kwargs
):
    """Grab the parameters and samples in a bilby file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `bilby` result file.
        Default False
    complex_params: list, optional
        list of parameters stored in the bilby result file which are complex
        and you wish to store the `amplitude` and `angle` as seperate
        posterior distributions
    latex_dict: dict, optional
        list of latex labels for each parameter
    """
    from pesummary.core.file.formats.bilby import (
        read_bilby as _read_bilby
    )

    return _read_bilby(
        path, disable_prior=disable_prior, latex_dict=latex_dict,
        complex_params=complex_params, _bilby_class=Bilby, **kwargs
    )


def prior_samples_from_file(path, cls="BBHPriorDict", nsamples=5000, **kwargs):
    """Return a dict of prior samples from a `bilby` prior file

    Parameters
    ----------
    path: str
        path to a `bilby` prior file
    cls: str, optional
        class you wish to read in the prior file
    nsamples: int, optional
        number of samples to draw from a prior file. Default 5000
    """
    from pesummary.core.file.formats.bilby import (
        prior_samples_from_file as _prior_samples_from_file
    )
    from bilby.gw import prior

    if isinstance(cls, str):
        cls = getattr(prior, cls)
    return _prior_samples_from_file(path, cls=cls, nsamples=nsamples, **kwargs)


class Bilby(GWSingleAnalysisRead):
    """PESummary wrapper of `bilby` (https://git.ligo.org/lscsoft/bilby). The
    path_to_results_file argument will be passed directly to
    `bilby.core.result.read_in_result`. All functions therefore use `bilby`
    methods and requires `bilby` to be installed.

    Parameters
    ----------
    path_to_results_file: str
        path to the results file that you wish to read in with `bilby`.
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `bilby` result file.
        Default False
    disable_prior_conversion: Bool, optional
        if True, disable the conversion module from deriving alternative prior
        distributions. Default False
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples

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
    prior: dict
        dictionary of prior samples extracted from the bilby result file. The
        analytic priors are evaluated for 5000 samples
    injection_parameters: dict
        dictionary of injection parameters stored in the result file
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
    def __init__(self, path_to_results_file, **kwargs):
        super(Bilby, self).__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_bilby_file, **kwargs)

    @staticmethod
    def grab_priors(bilby_object, nsamples=5000):
        """Draw samples from the prior functions stored in the bilby file
        """
        from pesummary.utils.array import Array

        f = bilby_object
        try:
            samples = f.priors.sample(size=nsamples)
            priors = {key: Array(samples[key]) for key in samples}
        except Exception as e:
            logger.info("Failed to draw prior samples because {}".format(e))
            priors = {}
        return priors

    @staticmethod
    def grab_extra_kwargs(bilby_object):
        """Grab any additional information stored in the bilby file
        """
        from pesummary.core.file.formats.bilby import config_from_object

        f = bilby_object
        kwargs = CoreBilby.grab_extra_kwargs(bilby_object)
        try:
            kwargs["meta_data"]["f_ref"] = \
                f.meta_data["likelihood"]["waveform_arguments"]["reference_frequency"]
        except Exception:
            pass
        for key, item in f.meta_data["likelihood"].items():
            if not isinstance(item, dict):
                try:
                    if isinstance(item, bool):
                        kwargs["meta_data"][key] = str(item)
                    else:
                        kwargs["meta_data"][key] = item
                except Exception:
                    pass
        try:
            kwargs["meta_data"]["approximant"] = \
                f.meta_data["likelihood"]["waveform_arguments"]["waveform_approximant"]
        except Exception:
            pass
        try:
            kwargs["meta_data"]["IFOs"] = \
                " ".join(f.meta_data["likelihood"]["interferometers"].keys())
        except Exception:
            pass
        _config = config_from_object(f)
        if len(_config):
            if "config" in _config.keys():
                _config = _config["config"]
            options = [
                "minimum_frequency", "reference_frequency",
                "waveform_approximant"
            ]
            pesummary_names = ["f_low", "f_ref", "approximant"]
            for opt, name in zip(options, pesummary_names):
                if opt in _config.keys():
                    try:
                        import ast
                        _option = ast.literal_eval(_config[opt])
                    except ValueError:
                        _option = _config[opt]
                    if isinstance(_option, dict):
                        _value = np.min(list(_option.values()))
                    else:
                        _value = _option
                    kwargs["meta_data"][name] = _value
        return kwargs

    @property
    def calibration_spline_posterior(self):
        if not any("recalib_" in i for i in self.parameters):
            return super(Bilby, self).calibration_spline_posterior
        ifos = np.unique(
            [
                param.split('_')[1] for param in self.parameters if 'recalib_'
                in param and "non_reweighted" not in param
            ]
        )
        amp_params, phase_params, log_freqs = {}, {}, {}
        for ifo in ifos:
            amp_params[ifo], phase_params[ifo] = [], []
            freq_params = np.sort(
                [
                    param for param in self.parameters if
                    'recalib_%s_frequency_' % (ifo) in param
                    and "non_reweighted" not in param
                ]
            )
            posterior = self.samples_dict
            log_freqs[ifo] = np.log(
                [posterior[param][0] for param in freq_params]
            )
            amp_parameters = np.sort(
                [
                    param for param in self.parameters if
                    'recalib_%s_amplitude_' % (ifo) in param
                    and "non_reweighted" not in param
                ]
            )
            amplitude = np.array([posterior[param] for param in amp_parameters])
            phase_parameters = np.sort(
                [
                    param for param in self.parameters if
                    'recalib_%s_phase_' % (ifo) in param
                    and "non_reweighted" not in param
                ]
            )
            phase = np.array([posterior[param] for param in phase_parameters])
            for num, i in enumerate(amplitude):
                amp_params[ifo].append(i)
                phase_params[ifo].append(phase[num])
        return log_freqs, amp_params, phase_params

    @staticmethod
    def load_strain_data(path_to_strain_file):
        """Load the strain data

        Parameters
        ----------
        path_to_strain_file: str
            path to the strain file that you wish to load
        """
        Bilby.load_from_function(
            Bilby._timeseries_from_bilby_pickle, path_to_strain_file)

    @staticmethod
    def _timeseries_from_bilby_pickle(path_to_strain_file):
        """Load a bilby pickle file containing the strain data

        Parameters
        ----------
        path_to_strain_file: str
            path to the strain file that you wish to load
        """
        import pickle
        import gwpy

        with open(path_to_strain_file, "rb") as f:
            data = pickle.load(f)

        strain_data = {}
        for i in data.interferometers:
            strain_data[i.name] = gwpy.timeseries.TimeSeries(
                data=i.strain_data.time_domain_strain,
                times=i.strain_data.time_array)
        return strain_data

    @staticmethod
    def _grab_data_from_bilby_file(path, disable_prior=False, **kwargs):
        """
        Load the results file using the `bilby` library

        Complex matched filter SNRs are stored in the result file.
        The amplitude and angle are extracted here.
        """
        return read_bilby(path, disable_prior=disable_prior, **kwargs)

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass
