# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from scipy.stats import gaussian_kde
import scipy.integrate as integrate


def weighted_credible_interval(samples, percentile, weights):
    """Compute the credible interval from a set of weighted samples.

    Parameters
    ----------
    samples: np.ndarray
        array of samples you wish to calculate the credible interval for
    percentile: float, list
        either a float, or a list of floats, giving the percentile you wish to
        use
    weights: np.ndarray
        array of weights of length samples
    """
    percentile = np.array(percentile).astype(float)
    if percentile.ndim < 1:
        percentile = np.array([percentile])
    ind_sorted = np.argsort(samples)
    sorted_data = samples[ind_sorted]
    sorted_weights = weights[ind_sorted]
    Sn = 100 * sorted_weights.cumsum() / sorted_weights.sum()
    data = np.zeros_like(percentile)
    for num, p in enumerate(percentile):
        inds = np.argwhere(Sn >= p)[0]
        data[num] = np.interp(percentile, Sn[inds], sorted_data[inds])[0]

    if len(percentile) == 1:
        return float(data[0])
    return data


def two_sided_credible_interval(samples, percentile, weights=None):
    """Compute the 2-sided credible interval from a set of samples.

    Parameters
    ----------
    samples: np.ndarray
        array of samples you wish to calculate the credible interval for
    percentile: list
        list of floats, giving the upper and lower percentile you wish to use
    weights: np.ndarray, optional
        array of weights of length samples
    """
    percentile = np.array(percentile).astype(float)
    if percentile.ndim and len(percentile) > 2:
        raise ValueError(
            "Please provide only an upper and lower credible interval"
        )
    if not weights:
        return np.percentile(samples, percentile)
    return weighted_credible_interval(samples, percentile, weights)


def hpd_two_sided_credible_interval(
    samples, percentile, weights=None, xlow=None, xhigh=None, xN=1000,
    kde=gaussian_kde, kde_kwargs={}, x=None, pdf=None
):
    """Compute the highest posterior density (HPD) 2-sided credible interval
    from a set of samples. Code adapted from BNS_plots.ipynb located here:
    https://git.ligo.org/publications/O2/cbc-catalog/-/blob/master/event_plots

    Parameters
    ----------
    samples: np.ndarray
        array of samples you wish to calculate the credible interval for
    percentile: float
        the percentile you wish to use
    weights: np.ndarray, optional
        array of weights of length samples. These weights are added to
        `kde_kwargs` and passed to kde. Only used if pdf=None
    xlow: float, optional
        minimum value to evaluate the KDE. If not provided, and x=None, xlow is
        set to the minimum sample
    xhigh: float, optional
        maximum value to evaluate the KDE. If not provided, and x=None, xhigh is
        set to the maximum sample
    xN: float, optional
        number of points to evaluate the KDE between xlow and xhigh. Only used
        if x=None. Default 1000
    kde: func, optional
        function to use to generate the KDE. Default scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        kwargs to pass to kde. Default {}
    x: np.ndarray, optional
        array of points to evaluate the KDE. If not provided,
        x=np.linspace(xlow, xhigh, xN). Default None.
    pdf: np.ndarray, optional
        the PDF to use when calculating the 2-sided credible interval. If
        provided, kde and kde_kwargs are not used. The array of points used to
        evaluate the PDF must be provided. Default None
    """
    percentile = np.array(percentile).astype(float)
    if percentile.ndim >= 1:
        raise ValueError(
            "Unable to pass more than one percentile when using the highest"
            "posterior density method"
        )
    if xlow is None and x is None:
        xlow = np.min(samples)
    elif xlow is None:
        xlow = np.min(x)
    if xhigh is None and x is None:
        xhigh = np.max(samples)
    elif xhigh is None:
        xhigh = np.max(x)
    if pdf is not None and x is None:
        raise ValueError(
            "Please provide the array of points used to evaluate the PDF"
        )
    if x is None:
        x = np.linspace(xlow, xhigh, xN)
    if pdf is None:
        if weights is not None:
            kde_kwargs.update({"weights": weights})
        pdf = kde(samples, **kde_kwargs)(x)

    credible_interval = []
    ilow = 0
    ihigh = len(x) - 1
    xlow, xhigh = x[ilow], x[ihigh]
    area = 1.0
    while area > (percentile / 100.0):
        xlow, xhigh = x[ilow], x[ihigh]
        x_interval = x[ilow:ihigh + 1]
        pdf_interval = pdf[ilow:ihigh + 1]
        if pdf[ilow] > pdf[ihigh]:
            ihigh -= 1
        else:
            ilow += 1
        area = integrate.simps(pdf_interval, x_interval)
    return np.array([xlow, xhigh]), area
