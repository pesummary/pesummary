# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
options = {}


def rejection_sampling(data, weights):
    """Reweight an input using rejection sampling

    Parameters
    ----------
    data: np.ndarray/pesummary.utils.samples_dict.SamplesDict
        posterior table you wish to reweight
    weights: np.ndarray
        a set of weights for each sample
    """
    weights = np.asarray(weights)
    idx = weights > np.random.uniform(0, np.max(weights), len(weights))
    logger.info(
        "Rejection sampling resulted in {} samples ({} input)".format(
            idx.sum(), len(idx)
        )
    )
    return data[idx]
