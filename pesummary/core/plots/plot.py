# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from pesummary.utils.utils import logger
from pesummary.core.plots.kde import kdeplot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

import numpy as np
from scipy import signal


def _autocorrelation_plot(param, samples):
    """Generate the autocorrelation function for a set of samples for a given
    parameter for a given approximant.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: list
        list of samples for param
    """
    logger.debug("Generating the autocorrelation function for %s" % (param))
    samples = samples[int(len(samples) / 2):]
    x = samples - np.mean(samples)
    y = np.conj(x[::-1])
    acf = np.fft.ifftshift(signal.fftconvolve(y, x, mode='full'))
    N = np.array(samples).shape[0]
    acf = acf[0:N]
    fig = plt.figure()
    plt.plot(acf / acf[0], linestyle=' ', marker='o', markersize=0.5)
    plt.ticklabel_format(axis="x", style="plain")
    plt.xlabel("lag", fontsize=16)
    plt.ylabel("ACF", fontsize=16)
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _sample_evolution_plot(param, samples, latex_label, inj_value=None):
    """Generate a scatter plot showing the evolution of the samples for a
    given parameter for a given approximant.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: list
        list of samples for param
    latex_label: str
        latex label for param
    inj_value: float
        value that was injected
    """
    logger.debug("Generating the sample scatter plot for %s" % (param))
    fig = plt.figure()
    n_samples = len(samples)
    plt.plot(range(n_samples), samples, linestyle=' ', marker='o',
             markersize=0.5)
    plt.ticklabel_format(axis="x", style="plain")
    plt.xlabel("samples", fontsize=16)
    plt.ylabel(latex_label, fontsize=16)
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _1d_cdf_plot(param, samples, latex_label):
    """Generate the cumulative distribution function for a given parameter for
    a given approximant.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: list
        list of samples for param
    latex_label: str
        latex label for param
    """
    logger.debug("Generating the 1d CDF for %s" % (param))
    fig = plt.figure()
    samples.sort()
    plt.xlabel(latex_label, fontsize=16)
    plt.ylabel("Cumulative Density Function", fontsize=16)
    upper_percentile = np.percentile(samples, 90)
    lower_percentile = np.percentile(samples, 10)
    median = np.median(samples)
    upper = np.round(upper_percentile - median, 2)
    lower = np.round(median - lower_percentile, 2)
    median = np.round(median, 2)
    plt.title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower), fontsize=18)
    plt.plot(samples, np.linspace(0, 1, len(samples)), color='b')
    plt.grid(b=True)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    return fig


def _1d_cdf_comparison_plot(param, samples, colors, latex_label, labels):
    """Generate a plot to compare the cdfs for a given parameter for different
    approximants.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    approximants: list
        list of approximant names that you would like to compare
    samples: 2d list
        list of samples for param for each approximant
    colors: list
        list of colors to be used to differentiate the different approximants
    latex_label: str
        latex label for param
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    """
    logger.debug("Generating the 1d comparison CDF for %s" % (param))
    fig = plt.figure(figsize=(8, 6))
    for num, i in enumerate(samples):
        i.sort()
        plt.plot(i, np.linspace(0, 1, len(i)), color=colors[num],
                 label=labels[num])
    plt.xlabel(latex_label, fontsize=16)
    plt.ylabel("Cumulative Density Function", fontsize=16)
    plt.grid(b=True)
    plt.ylim([0, 1.05])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    return fig


def _1d_histogram_plot(param, samples, latex_label, inj_value=None, kde=False):
    """Generate the 1d histogram plot for a given parameter for a given
    approximant.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: list
        list of samples for param
    latex_label: str
        latex label for param
    inj_value: float
        value that was injected
    kde: Bool
        if true, a kde is plotted instead of a histogram
    """
    logger.debug("Generating the 1d histogram plot for %s" % (param))
    fig = plt.figure()
    if np.ptp(samples) == 0:
        plt.axvline(samples[0], color='b')
    elif not kde:
        plt.hist(samples, histtype="step", bins=50, color='b', density=True,
                 linewidth=1.75)
    else:
        kdeplot(samples, color='b', shade=True, alpha_shade=0.1,
                clip=[np.min(samples), np.max(samples)], linewidth=1.0)
    plt.xlabel(latex_label, fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    upper_percentile = np.percentile(samples, 90)
    lower_percentile = np.percentile(samples, 10)
    if inj_value:
        plt.axvline(inj_value, color='r', linestyle='--', linewidth=1.75)
    plt.axvline(upper_percentile, color='b', linestyle='--', linewidth=1.75)
    plt.axvline(lower_percentile, color='b', linestyle='--', linewidth=1.75)
    median = np.median(samples)
    upper = np.round(upper_percentile - median, 2)
    lower = np.round(median - lower_percentile, 2)
    median = np.round(median, 2)
    plt.title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower), fontsize=18)
    plt.grid(b=True)
    plt.tight_layout()
    return fig


def _1d_comparison_histogram_plot(param, samples, colors,
                                  latex_label, labels, kde=False):
    """Generate the a plot to compare the 1d_histogram plots for a given
    parameter for different approximants.

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    approximants: list
        list of approximant names that you would like to compare
    samples: 2d list
        list of samples for param for each approximant
    colors: list
        list of colors to be used to differentiate the different approximants
    latex_label: str
        latex label for param
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    kde: Bool
        if true, a kde is plotted instead of a histogram
    """
    logger.debug("Generating the 1d comparison histogram plot for %s" % (param))
    fig = plt.figure(figsize=(8, 6))
    for num, i in enumerate(samples):
        if np.ptp(i) == 0:
            plt.axvline(i[0], color=colors[num], label=labels[num])
        elif not kde:
            plt.hist(i, histtype="step", bins=50, color=colors[num],
                     label=labels[num], linewidth=2.5, density=True)
        else:
            kdeplot(i, color=colors[num], shade=True, alpha_shade=0.05,
                    clip=[np.min(i), np.max(i)], linewidth=1.5,
                    label=labels[num])
        plt.axvline(x=np.percentile(i, 90), color=colors[num], linestyle='--',
                    linewidth=2.5)
        plt.axvline(x=np.percentile(i, 10), color=colors[num], linestyle='--',
                    linewidth=2.5)
    plt.xlabel(latex_label, fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.grid(b=True)
    plt.tight_layout()
    return fig


def _comparison_box_plot(param, samples, colors, latex_label, labels):
    """Generate a box plot to compare 1d_histograms for a given parameter

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    approximants: list
        list of approximant names that you would like to compare
    samples: 2d list
        list of samples for param for each approximant
    colors: list
        list of colors to be used to differentiate the different approximants
    latex_label: str
        latex label for param
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    """
    logger.debug("Generating the 1d comparison boxplot plot for %s" % (param))
    fig = plt.figure()
    maximum = np.max([np.max(i) for i in samples])
    minimum = np.min([np.min(i) for i in samples])
    middle = (maximum + minimum) * 0.5
    plt.boxplot(samples, widths=0.2, vert=False, whis=np.inf, labels=labels)
    for num, i in enumerate(labels):
        plt.annotate(i, xy=(middle, 1), xytext=(middle, num + 1. + 0.2),
                     ha="center")
    plt.yticks([])
    plt.xlabel(latex_label, fontsize=16)
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _make_corner_plot(samples, params, latex_labels, **kwargs):
    """Generate the corner plots for a given approximant

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from the command line
    samples: nd list
        nd list of samples for each parameter for a given approximant
    params: list
        list of parameters associated with each element in samples
    approximant: str
        name of approximant that was used to generate the samples
    latex_labels: dict
        dictionary of latex labels for each parameter
    """
    logger.debug("Generating the corner plot")
    # set the default kwargs
    default_kwargs = dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3)
    xs = np.zeros([len(params), len(samples)])
    for num, i in enumerate(params):
        xs[num] = [j[params.index("%s" % (i))] for j in samples]
    default_kwargs['range'] = [1.0] * len(params)
    default_kwargs["labels"] = [latex_labels[i] for i in params]
    figure = corner.corner(xs.T, **default_kwargs)
    # grab the axes of the subplots
    axes = figure.get_axes()
    extent = axes[0].get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    width, height = extent.width, extent.height
    width *= figure.dpi
    height *= figure.dpi
    return figure, params
