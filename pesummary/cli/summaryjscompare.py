#!/usr/bin/env python3
"""
Interface to generate JS test comparison between two results files.

This plots the difference in CDF between two sets of samples and adds the JS test
statistic with uncertainty estimated by bootstrapping.
"""

import argparse
from collections import namedtuple

import numpy as np
import os
import pandas as pd
from scipy.stats import binom
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt

from pesummary.io import read
from pesummary.core.parser import parser as pesummary_parser
from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde
from pesummary.core.plots.figure import figure
from pesummary.gw.plots.bounds import default_bounds
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.utils.tqdm import tqdm
from pesummary.utils.utils import jensen_shannon_divergence
from pesummary.utils.utils import _check_latex_install, get_matplotlib_style_file, logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

matplotlib.style.use(get_matplotlib_style_file())
_check_latex_install()


def load_data(data_file):
    """ Read in a data file and return samples dictionary """
    f = read(data_file, package="gw", disable_prior=True)
    return f.samples_dict


def js_bootstrap(key, resultA, resultB, nsamples, ntests):
    """
    Evaluates mean JS divergence with bootstrapping
    key: string posterior parameter
    result_A: first full posterior samples set
    result_B: second full posterior samples set
    nsamples: number for downsampling full sample set
    ntests: number of iterations over different nsamples realisations
    returns: 1 dim array (of lenght ntests)
    """

    samplesA = resultA[key]
    samplesB = resultB[key]

    # Get minimum number of samples to use
    nsamples = min([nsamples, len(samplesA), len(samplesB)])

    xlow, xhigh = None, None
    if key in default_bounds.keys():
        bounds = default_bounds[key]
        if "low" in bounds.keys():
            xlow = bounds["low"]
        if "high" in bounds.keys():
            if isinstance(bounds["high"], str) and "mass_1" in bounds["high"]:
                xhigh = np.min([np.max(samplesA), np.max(samplesB)])
            else:
                xhigh = bounds["high"]

    js_array = np.zeros(ntests)

    for j in tqdm(range(ntests)):
        bootA = np.random.choice(samplesA, size=nsamples, replace=False)
        bootB = np.random.choice(samplesB, size=nsamples, replace=False)
        js_array[j] = np.nan_to_num(
            jensen_shannon_divergence([bootA, bootB], kde=Bounded_1d_kde, xlow=xlow, xhigh=xhigh)
        )
    return js_array


def calc_median_error(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    summary = namedtuple("summary", ["median", "lower", "upper"])
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    return summary


def bin_series_and_calc_cdf(x, y, bins=100):
    """
    Bin two unequal length series into equal bins
    and calculate their cumulative distibution function
    in order to generate pp-plots
    """
    boundaries = sorted(x)[:: round(len(x) / bins) + 1]
    labels = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    # Bin two series into equal bins
    try:
        xb = pd.cut(x, bins=boundaries, labels=labels)
        yb = pd.cut(y, bins=boundaries, labels=labels)
        # Get value counts for each bin and sort by bin
        xhist = xb.value_counts().sort_index(ascending=True) / len(xb)
        yhist = yb.value_counts().sort_index(ascending=True) / len(yb)
        # Make cumulative
        for ser in [xhist, yhist]:
            ttl = 0
            for idx, val in ser.iteritems():
                ttl += val
                ser.loc[idx] = ttl
    except ValueError:
        xhist = np.linspace(0, 1, 1000)
        yhist = np.linspace(0, 1, 1000)
    return xhist, yhist


def calculate_CI(len_samples, confidence_level=0.95, n_points=1001):
    """
    Calculate confidence intervals
    (https://git.ligo.org/lscsoft/bilby/blob/master/bilby/core/result.py#L1578)
    len_samples: lenght of posterior samples
    confidence level: default 90%
    n_points: number of points over which evaluating confidence region
    """
    x_values = np.linspace(0, 1, n_points)
    N = len_samples
    edge_of_bound = (1.0 - confidence_level) / 2.0
    lower = binom.ppf(1 - edge_of_bound, N, x_values) / N
    upper = binom.ppf(edge_of_bound, N, x_values) / N
    lower[0] = 0
    upper[0] = 0
    return x_values, upper, lower


def pp_plot(event, resultA, resultB, labelA, labelB, main_keys, nsamples, js_data, webdir):
    """
    Produce PP plot between sampleA and samplesB
    for a set of paramaters (main keys) for a given event.
    The JS divergence for each pair of samples is shown in legend.
    """
    # Creating dict where ks_data for each event will be saved
    fig, ax = figure(figsize=(6, 5), gca=True)

    latex_labels.update(GWlatex_labels)
    for key_index, key in enumerate(main_keys):

        # Check the key exists in both sets of samples
        if key not in resultA or key not in resultB:
            logger.debug(f"Missing key {key}")
            continue
        # Get minimum number of samples to use
        nsamples = min([nsamples, len(resultA[key]), len(resultB[key])])

        # Resample to nsamples
        lp = np.random.choice(resultA[key], size=nsamples, replace=False)
        bp = np.random.choice(resultB[key], size=nsamples, replace=False)

        # Bin posterior samples into equal lenght bins and calculate cumulative
        xhist, yhist = bin_series_and_calc_cdf(bp, lp)

        summary = js_data[key]
        logger.debug(f"JS {key}: {summary.median}, {summary.minus}, {summary.plus}")
        fmt = "{{0:{0}}}".format(".5f").format

        if key not in list(latex_labels.keys()):
            latex_labels[key] = key.replace("_", " ")
        ax.plot(
            xhist,
            yhist - xhist,
            label=latex_labels[key]
            + r" ${{{0}}}_{{-{1}}}^{{+{2}}}$".format(fmt(summary.median), fmt(summary.minus), fmt(summary.plus)),
            linewidth=1,
            linestyle="-",
        )

    for confidence in [0.68, 0.95, 0.997]:
        x_values, upper, lower = calculate_CI(nsamples, confidence_level=confidence)
        ax.fill_between(
            x_values, lower - x_values, upper - x_values, linewidth=1, color="k", alpha=0.1,
        )
    ax.set_xlabel(f"{labelA} CDF")
    ax.set_ylabel(f"{labelA} CDF - {labelB} CDF")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", ncol=4, fontsize=6)
    ax.set_title(r"{} N samples={:.0f}".format(event, nsamples))
    ax.grid()
    fig.tight_layout()
    plt.savefig(os.path.join(webdir, "{}-comparison-{}-{}.png".format(event, labelA, labelB)))


def parse_cmd_line(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--event", type=str, required=True, help="Label, e.g. the event name")
    parser.add_argument(
        "-r", "--samples", type=str, nargs=2, help=("Paths to a pair of results files to compare."),
    )
    parser.add_argument("-l", "--labels", type=str, nargs=2, help="Pair of labels for each result file")
    parser.add_argument(
        "--main_keys",
        nargs="+",
        default=[
            "theta_jn",
            "chirp_mass",
            "mass_ratio",
            "tilt_1",
            "tilt_2",
            "luminosity_distance",
            "ra",
            "dec",
            "a_1",
            "a_2",
        ],
        required=False,
        help="List of parameter names",
    )
    parser.add_argument(
        "--ntests", type=int, default=100, required=False, help="Number of iteration for bootstrapping",
    )
    parser.add_argument(
        "--nsamples", type=int, default=10000, required=False, help="Number of samples to use",
    )
    parser.add_argument("--random-seed", type=int, default=150914)
    parser.add_argument(
        "--webdir", type=str, default=".", required=False, help="Path to webdirectory where plots will be saved"
    )

    _parser = pesummary_parser(existing_parser=parser)
    args, unknown = _parser.parse_known_args(args=args)
    return args


def main(args=None):
    args = parse_cmd_line(args=args)

    # Set random seed
    np.random.seed(seed=args.random_seed)

    # Read in the keys to apply
    main_keys = args.main_keys

    # Read in the results and labels
    resultA = load_data(args.samples[0])
    labelA = args.labels[0]
    resultB = load_data(args.samples[1])
    labelB = args.labels[1]

    logger.debug("Evaluating JS divergence..")
    js_data = dict()
    js = np.zeros((args.ntests, len(main_keys)))
    for i, key in enumerate(main_keys):
        js[:, i] = js_bootstrap(key, resultA, resultB, args.nsamples, ntests=args.ntests,)
        js_data[key] = calc_median_error(js[:, i])
    logger.debug("Making pp-plot..")
    pp_plot(args.event, resultA, resultB, labelA, labelB, main_keys, args.nsamples, js_data=js_data, webdir=args.webdir)


if __name__ == "__main__":
    main()
