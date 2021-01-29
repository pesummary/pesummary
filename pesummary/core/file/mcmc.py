# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger
import numpy as np
import copy

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
STEP_NUMBER_PARAMS = ["cycle"]
algorithms = ["burnin_by_step_number", "burnin_by_first_n"]


def _number_of_negative_steps(samples, logger_level="debug"):
    """Return the number of samples that have step < 0 for each dictionary

    Parameters
    ----------
    samples: pesummary.utils.samples_dict.MCMCSamplesDict
        MCMCSamplesDict object containing the samples for multiple mcmc chains
    logger_level: str, optional
        logger level to use when printing information to stdout. Default debug
    """
    _samples = copy.deepcopy(samples)
    parameters = list(_samples.parameters)
    step_param = [
        alternative for alternative in STEP_NUMBER_PARAMS if alternative
        in parameters
    ]
    if not len(step_param):
        logger.warning(
            "Unable to find a step number in the MCMCSamplesDict object. "
            "Aborting discard"
        )
        return {key: None for key in _samples.keys()}
    elif len(step_param) > 1:
        step_param = step_param[0]
        getattr(logger, logger_level)(
            "Multiple columns found with possible step numbers. Using "
            "{}".format(step_param)
        )
    else:
        step_param = step_param[0]
    keys = _samples.keys()
    step_idx = [
        np.arange(_samples[key].number_of_samples)[_samples[key][step_param] > 0]
        for key in _samples.keys()
    ]
    return {
        key: step[0] if len(step_idx) else 0 for key, step in
        zip(keys, step_idx)
    }


def burnin_by_step_number(samples, logger_level="debug"):
    """Discard all samples with step number < 0 as burnin

    Parameters
    ----------
    samples: pesummary.utils.samples_dict.MCMCSamplesDict
        MCMCSamplesDict object containing the samples for multiple mcmc chains
    logger_level: str, optional
        logger level to use when printing information to stdout. Default debug
    """
    _samples = copy.deepcopy(samples)
    n_samples = _number_of_negative_steps(_samples, logger_level=logger_level)
    return _samples.discard_samples(n_samples)


def burnin_by_first_n(samples, N, step_number=False, logger_level="debug"):
    """Discard the first N samples as burnin

    Parameters
    ----------
    samples: pesummary.utils.samples_dict.MCMCSamplesDict
        MCMCSamplesDict object containing the samples for multiple mcmc chains
    N: int
        Number of samples to discard as burnin
    step_number: Bool, optional
        If True, discard all samples that have step number < N
    logger_level: str, optional
        logger level to use when printing information to stdout. Default debug
    """
    _samples = copy.deepcopy(samples)
    n_samples = {key: N for key in _samples.keys()}
    if step_number:
        n_samples = {
            key: item + N for key, item in _number_of_negative_steps(
                _samples, logger_level=logger_level
            ).items()
        }
    return _samples.discard_samples(n_samples)
