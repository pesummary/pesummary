# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.dingo import Dingo as CoreDingo
from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_dingo(path, disable_prior=False, **kwargs):
    """Grab the parameters and samples in a dingo file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `dingo` result file.
        Default False
    """
    from pesummary.core.file.formats.dingo import read_dingo as _read_dingo

    return _read_dingo(path, disable_prior=disable_prior, _dingo_class=Dingo, **kwargs)


class Dingo(GWSingleAnalysisRead):
    """PESummary wrapper of `dingo` (https://github.com/dingo-gw/dingo). The
    path_to_results_file argument will be passed directly to
    `bilby.gw.result.Result`. All functions therefore use `dingo`
    methods and requires `dingo` to be installed.

    Parameters
    ----------
    path_to_results_file: str
        path to the results file that you wish to read in with `dingo`.
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `dingo` result file.
        Default False
    disable_prior_conversion: Bool, optional
        if True, disable the conversion module from deriving alternative prior
        distributions. Default False
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples
    remove_nan_likelihood_samples: Bool, optional
        if True, remove samples which have log_likelihood='nan'. Default True

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
        super().__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_dingo_file, **kwargs)

    @staticmethod
    def grab_priors(dingo_object, nsamples=5000):
        """Draw samples from the prior functions stored in the dingo object"""
        from pesummary.utils.array import Array

        f = dingo_object
        try:
            samples = f.pesummary_prior.sample(size=nsamples)
            priors = {key: Array(samples[key]) for key in samples}
        except Exception as e:
            logger.info("Failed to draw prior samples because {}".format(e))
            priors = {}
        return priors

    @staticmethod
    def grab_extra_kwargs(dingo_object):
        """Grab any additional information stored in the dingo file"""
        f = dingo_object
        kwargs = CoreDingo.grab_extra_kwargs(dingo_object)
        try:
            kwargs["meta_data"]["f_ref"] = f.f_ref
        except Exception:
            pass
        try:
            kwargs["meta_data"]["approximant"] = f.approximant
        except Exception:
            pass
        try:
            kwargs["meta_data"]["IFOs"] = " ".join(f.interferometers)
        except Exception:
            pass
        try:
            kwargs["meta_data"]["f_low"] = f.domain.f_min
        except Exception:
            pass
        try:
            kwargs["meta_data"]["delta_f"] = f.domain.delta_f
        except Exception:
            pass
        try:
            kwargs["meta_data"]["f_final"] = f.domain.f_max
        except Exception:
            pass
        return kwargs

    @staticmethod
    def _grab_data_from_dingo_file(path, disable_prior=False, **kwargs):
        """
        Load the results file using the `dingo` library

        Complex matched filter SNRs are stored in the result file.
        The amplitude and angle are extracted here.
        """
        return read_dingo(path, disable_prior=disable_prior, **kwargs)

    # TODO: Load strain data and PSD? Calibration spline posterior? Complex params?
    #  Latex labels?
