# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.core.file.formats.base_read import SingleAnalysisRead
from pesummary.core.file.formats.bilby import _bilby_prior_dict_to_pesummary_samples_dict
from pesummary import conf
from pesummary.utils.utils import logger

__author__ = [
    "Stephen Green <stephen.green@ligo.org>",
    "Nihar Gupte <nihar.gupte@ligo.org>",
]


def _load_dingo(path):
    """Wrapper for `dingo.gw.result.Result`

    Parameters
    ----------
    path: str
        path to the dingo result file you wish to load
    """
    from dingo.gw.result import Result

    return Result(file_name=path)


# TODO: Set up for complex parameters (but first enable output of complex SNR from Dingo)


def read_dingo(
    path, disable_prior=False, nsamples_for_prior=None, _dingo_class=None, **kwargs
):
    """Grab the parameters and samples in a dingo file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `bilby` result file.
        Default False
    nsamples_for_prior: int, optional
        number of samples to draw from the analytic priors
    """
    if _dingo_class is None:
        _dingo_class = Dingo
    dingo_object = _load_dingo(path)
    posterior = dingo_object.get_pesummary_samples(num_processes=kwargs.get("num_processes", 1))
    parameters = list(posterior.keys())
    samples = posterior.to_numpy()
    injection = dingo_object.injection_parameters
    if injection is None:
        injection = {i: j for i, j in zip(parameters, [float("nan")] * len(parameters))}
    else:
        for i in parameters:
            if i not in injection.keys():
                injection[i] = float("nan")

    try:
        extra_kwargs = _dingo_class.grab_extra_kwargs(dingo_object)
    except Exception:
        extra_kwargs = {"sampler": {}, "meta_data": {}}
    extra_kwargs["sampler"]["nsamples"] = len(samples)
    if "weights" in dingo_object.samples:
        extra_kwargs["sampler"]["pe_algorithm"] = "dingo-is"
    else:
        extra_kwargs["sampler"]["pe_algorithm"] = "dingo"
    try:
        version = dingo_object.version  
    except Exception as e:
        version = None

    data = {
        "parameters": parameters,
        "samples": samples.tolist(),
        "injection": injection,
        "version": version,
        "kwargs": extra_kwargs,
    }
    if not disable_prior:
        logger.debug("Drawing prior samples from dingo result file")
        if nsamples_for_prior is None:
            nsamples_for_prior = len(samples)
        prior_samples = Dingo.grab_priors(dingo_object, nsamples=nsamples_for_prior)
        data["prior"] = {"samples": prior_samples}
        if len(prior_samples):
            data["prior"]["analytic"] = prior_samples.analytic
    else:
        try:
            _prior = dingo_object.pesummary_prior
            data["prior"] = {
                "samples": {},
                "analytic": {key: str(item) for key, item in _prior.items()},
            }
        except (AttributeError, KeyError):
            pass
    return data


def prior_samples_from_dingo_object(dingo_object, nsamples=5000, **kwargs):
    """Return a dict of prior samples from a `dingo.core.result.Result`
    object

    Parameters
    ----------
    dingo_object: dingo.gw.result.Result
        a dingo.core.result.Result object you wish to draw prior samples from
    nsamples: int, optional
        number of samples to draw from a prior file. Default 5000
    """
    samples = dingo_object.pesummary_prior.sample(size=nsamples)
    return _bilby_prior_dict_to_pesummary_samples_dict(
        samples, prior=dingo_object.pesummary_prior
    )


class Dingo(SingleAnalysisRead):
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
    remove_nan_likelihood_samples: Bool, optional
        if True, remove samples which have log_likelihood='nan'. Default True

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
    prior: dict
        dictionary of prior samples keyed by parameters. The prior functions
        are evaluated for 5000 samples.
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

    def __init__(self, path_to_results_file, **kwargs):
        super().__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_dingo_file, **kwargs)

    @staticmethod
    def grab_priors(dingo_object, nsamples=5000):
        """Draw samples from the prior functions stored in the dingo object"""
        try:
            return prior_samples_from_dingo_object(dingo_object, nsamples=nsamples)
        except Exception as e:
            logger.info("Failed to draw prior samples because {}".format(e))
            return {}

    @staticmethod
    def grab_extra_kwargs(dingo_object):
        """Grab any additional information stored in the dingo object"""
        f = dingo_object
        kwargs = {
            "sampler": {},
            "meta_data": {},
            "event_metadata": f.event_metadata,
            "other": f.metadata,
        }
        if f.log_evidence:
            kwargs["sampler"][conf.log_evidence] = np.round(f.log_evidence, 4)
        if f.log_evidence_std:
            kwargs["sampler"][conf.log_evidence_error] = np.round(f.log_evidence_std, 4)
        if f.log_bayes_factor:
            kwargs["sampler"][conf.log_bayes_factor] = np.round(f.log_bayes_factor, 4)
        if f.log_noise_evidence:
            kwargs["sampler"][conf.log_noise_evidence] = np.round(f.log_noise_evidence, 4)
        return kwargs

    @staticmethod
    def _grab_data_from_dingo_file(path, **kwargs):
        """Load the results file using the `dingo` library"""
        return read_dingo(path, **kwargs)
