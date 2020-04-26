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

from pesummary.utils.utils import (
    logger, number_of_columns_for_legend, _check_latex_install,
    get_matplotlib_style_file, gelman_rubin, get_matplotlib_backend
)
from pesummary.core.plots.kde import kdeplot
from pesummary import conf

import matplotlib
matplotlib.use(get_matplotlib_backend())
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
import copy
from itertools import cycle

import numpy as np
from scipy import signal
plt.style.use(get_matplotlib_style_file())
_check_latex_install()


def _autocorrelation_plot(
    param, samples, fig=None, color=conf.color, markersize=0.5
):
    """Generate the autocorrelation function for a set of samples for a given
    parameter for a given approximant.

     Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: list
        list of samples for param
    fig: matplotlib.pyplot.figure
        existing figure you wish to use
    color: str, optional
        color you wish to use for the autocorrelation plot
    """
    logger.debug("Generating the autocorrelation function for %s" % (param))
    if fig is None:
        fig = plt.figure()
    samples = samples[int(len(samples) / 2):]
    x = samples - np.mean(samples)
    y = np.conj(x[::-1])
    acf = np.fft.ifftshift(signal.fftconvolve(y, x, mode='full'))
    N = np.array(samples).shape[0]
    acf = acf[0:N]
    plt.plot(acf / acf[0], linestyle=' ', marker='o', markersize=markersize,
             color=color)
    plt.ticklabel_format(axis="x", style="plain")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _autocorrelation_plot_mcmc(param, samples, colorcycle=conf.colorcycle):
    """Generate the autocorrelation function for a set of samples for a given
    parameter for a given set of mcmc chains

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: np.ndarray
        2d array containing a list of samples for param for each mcmc chain
    colorcycle: list, str
        color cycle you wish to use for the different mcmc chains
    """
    cycol = cycle(colorcycle)
    fig = plt.figure()
    for ss in samples:
        fig = _autocorrelation_plot(
            param, ss, fig=fig, markersize=1.25, color=next(cycol)
        )
    return fig


def _sample_evolution_plot(
    param, samples, latex_label, inj_value=None, fig=None, color=conf.color,
    markersize=0.5
):
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
    fig: matplotlib.pyplot.figure, optional
        existing figure you wish to use
    color: str, optional
        color you wish to use to plot the scatter points
    """
    logger.debug("Generating the sample scatter plot for %s" % (param))
    if fig is None:
        fig = plt.figure()
    n_samples = len(samples)
    plt.plot(range(n_samples), samples, linestyle=' ', marker='o',
             markersize=markersize, color=color)
    plt.ticklabel_format(axis="x", style="plain")
    plt.xlabel("samples")
    plt.ylabel(latex_label)
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _sample_evolution_plot_mcmc(
    param, samples, latex_label, inj_value=None, colorcycle=conf.colorcycle
):
    """Generate a scatter plot showing the evolution of the samples in each
    mcmc chain for a given parameter

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: np.ndarray
        2d array containing the samples for param for each mcmc chain
    latex_label: str
        latex label for param
    inj_value: float
        value that was injected
    colorcycle: list, str
        color cycle you wish to use for the different mcmc chains
    """
    cycol = cycle(colorcycle)
    fig = plt.figure()
    for ss in samples:
        fig = _sample_evolution_plot(
            param, ss, latex_label, inj_value=None, fig=fig, markersize=1.25,
            color=next(cycol)
        )
    return fig


def _1d_cdf_plot(
    param, samples, latex_label, fig=None, color=conf.color, title=True
):
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
    fig: matplotlib.pyplot.figure, optional
        existing figure you wish to use
    color: str, optional09
        color you wish to use to plot the scatter points
    title: Bool, optional
        if True, add a title to the 1d cdf plot showing giving the median
        and symmetric 90% credible intervals
    """
    logger.debug("Generating the 1d CDF for %s" % (param))
    if fig is None:
        fig = plt.figure()
    sorted_samples = copy.deepcopy(samples)
    sorted_samples.sort()
    plt.xlabel(latex_label)
    plt.ylabel("Cumulative Density Function")
    upper_percentile = np.percentile(samples, 95)
    lower_percentile = np.percentile(samples, 5)
    median = np.median(samples)
    upper = np.round(upper_percentile - median, 2)
    lower = np.round(median - lower_percentile, 2)
    median = np.round(median, 2)
    if title:
        plt.title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower))
    plt.plot(sorted_samples, np.linspace(0, 1, len(sorted_samples)),
             color=color)
    plt.grid(b=True)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    return fig


def _1d_cdf_plot_mcmc(param, samples, latex_label, colorcycle=conf.colorcycle):
    """Generate the cumulative distribution function for a given parameter
    for a given set of mcmc chains

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: np.ndarray
        2d array containing the samples for param for each mcmc chain
    latex_label: str
        latex label for param
    colorcycle: list, str
        color cycle you wish to use for the different mcmc chains
    """
    cycol = cycle(colorcycle)
    fig = plt.figure()
    for ss in samples:
        fig = _1d_cdf_plot(
            param, ss, latex_label, fig=fig, color=next(cycol), title=False
        )
    gelman = gelman_rubin(samples)
    plt.title("Gelman-Rubin: {}".format(gelman))
    return fig


def _1d_cdf_comparison_plot(
        param, samples, colors, latex_label, labels, linestyles=None
):
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
    if linestyles is None:
        linestyles = ["-"] * len(samples)
    fig = plt.figure(figsize=(8, 6))
    handles = []
    for num, i in enumerate(samples):
        sorted_samples = copy.deepcopy(samples[num])
        sorted_samples = sorted(sorted_samples)
        plt.plot(sorted_samples, np.linspace(0, 1, len(sorted_samples)),
                 color=colors[num], label=labels[num],
                 linestyle=linestyles[num])
        handles.append(
            mlines.Line2D([], [], color=colors[num], label=labels[num])
        )
    ncols = number_of_columns_for_legend(labels)
    legend = plt.legend(
        handles=handles, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        handlelength=3, ncol=ncols, mode="expand", borderaxespad=0.
    )
    for num, legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(1.75)
        legobj.set_linestyle(linestyles[num])
    plt.xlabel(latex_label)
    plt.ylabel("Cumulative Density Function")
    plt.grid(b=True)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    return fig


def _1d_histogram_plot(param, samples, latex_label, inj_value=None, kde=False,
                       prior=None, weights=None, xlow=None, xhigh=None,
                       fig=None, title=True, color=conf.color,
                       autoscale=True, bins=50, histtype="step"):
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
    prior: list
        list of prior samples for param
    weights: list
        list of weights for each sample
    fig: matplotlib.pyplot.figure, optional
        existing figure you wish to use
    color: str, optional09
        color you wish to use to plot the scatter points
    title: Bool, optional
        if True, add a title to the 1d cdf plot showing giving the median
        and symmetric 90% credible intervals
    autoscale: Bool, optional
        autoscale the x axis
    bins: int, optional
        number of bins to use for histogram
    histtype: str, optional
        histogram type to use when plotting
    """
    logger.debug("Generating the 1d histogram plot for %s" % (param))
    if fig is None:
        fig = plt.figure()
    if np.ptp(samples) == 0:
        plt.axvline(samples[0], color=conf.color)
        xlims = plt.gca().get_xlim()
    elif not kde:
        plt.hist(samples, histtype=histtype, bins=bins, color=color,
                 density=True, linewidth=1.75, weights=weights)
        xlims = plt.gca().get_xlim()
        if prior is not None:
            plt.hist(prior, color=conf.prior_color, alpha=0.2, edgecolor="w",
                     density=True, linewidth=1.75, histtype="bar", bins=bins)
    else:
        kwargs = {"shade": True, "alpha_shade": 0.1, "linewidth": 1.0}
        if xlow is not None or xhigh is not None:
            kwargs.update({"xlow": xlow, "xhigh": xhigh})
        else:
            kwargs.update({"clip": [samples.minimum, samples.maximum]})
        x = kdeplot(samples, color=color, **kwargs)
        xlims = plt.gca().get_xlim()
        if prior is not None:
            kdeplot(prior, color=conf.prior_color, **kwargs)
    plt.xlabel(latex_label)
    plt.ylabel("Probability Density")
    percentile = samples.confidence_interval([5, 95])
    if inj_value is not None:
        plt.axvline(inj_value, color=conf.injection_color, linestyle='-',
                    linewidth=2.5)
    plt.axvline(percentile[0], color=color, linestyle='--', linewidth=1.75)
    plt.axvline(percentile[1], color=color, linestyle='--', linewidth=1.75)
    median = samples.average("median")
    if title:
        upper = np.round(percentile[1] - median, 2)
        lower = np.round(median - percentile[0], 2)
        median = np.round(median, 2)
        plt.title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower))
    plt.grid(b=True)
    if autoscale:
        plt.xlim(xlims)
    plt.tight_layout()
    return fig


def _1d_histogram_plot_mcmc(
    param, samples, latex_label, inj_value=None, kde=False, prior=None,
    weights=None, xlow=None, xhigh=None, colorcycle=conf.colorcycle
):
    """Generate a 1d histogram plot for a given parameter for a given
    set of mcmc chains

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: np.ndarray
        2d array of samples for param for each mcmc chain
    latex_label: str
        latex label for param
    inj_value: float
        value that was injected
    kde: Bool
        if true, a kde is plotted instead of a histogram
    prior: list
        list of prior samples for param
    weights: list
        list of weights for each sample
    colorcycle: list, str
        color cycle you wish to use for the different mcmc chains
    """
    cycol = cycle(colorcycle)
    fig = plt.figure()
    for ss in samples:
        fig = _1d_histogram_plot(
            param, ss, latex_label, inj_value=inj_value, kde=kde, prior=prior,
            weights=weights, xlow=xlow, xhigh=xhigh, fig=fig,
            color=next(cycol), title=False, autoscale=False
        )
    gelman = gelman_rubin(samples)
    plt.title("Gelman-Rubin: {}".format(gelman))
    return fig


def _1d_comparison_histogram_plot(
    param, samples, colors, latex_label, labels, kde=False, linestyles=None,
    xlow=None, xhigh=None
):
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
    linestyles: list
        list of linestyles for each set of samples
    """
    logger.debug("Generating the 1d comparison histogram plot for %s" % (param))
    if linestyles is None:
        linestyles = ["-"] * len(samples)
    fig = plt.figure(figsize=(8, 6))
    handles = []
    for num, i in enumerate(samples):
        if np.ptp(i) == 0:
            plt.axvline(i[0], color=colors[num], label=labels[num])
        elif not kde:
            plt.hist(i, histtype="step", bins=50, color=colors[num],
                     label=labels[num], linewidth=2.5, density=True,
                     linestyle=linestyles[num])
        else:
            kwargs = {
                "shade": True, "alpha_shade": 0.05, "linewidth": 1.5,
                "label": labels[num]
            }
            if xlow is not None or xhigh is not None:
                kwargs.update({"xlow": xlow, "xhigh": xhigh})
            else:
                kwargs.update({"clip": [np.min(i), np.max(i)]})
            kdeplot(i, color=colors[num], **kwargs)
        plt.axvline(x=np.percentile(i, 95), color=colors[num], linestyle='--',
                    linewidth=2.5)
        plt.axvline(x=np.percentile(i, 5), color=colors[num], linestyle='--',
                    linewidth=2.5)
        handles.append(
            mlines.Line2D([], [], color=colors[num], label=labels[num])
        )
    ncols = number_of_columns_for_legend(labels)
    legend = plt.legend(
        handles=handles, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        handlelength=3, ncol=ncols, mode="expand", borderaxespad=0.
    )
    for num, legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(1.75)
        legobj.set_linestyle(linestyles[num])
    plt.xlabel(latex_label)
    plt.ylabel("Probability Density")
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
    plt.xlabel(latex_label)
    plt.tight_layout()
    plt.grid(b=True)
    return fig


def _make_corner_plot(samples, latex_labels, corner_parameters=None, **kwargs):
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
    default_kwargs = conf.corner_kwargs
    parameters = list(samples.keys())
    if corner_parameters is not None:
        included_parameters = [i for i in parameters if i in corner_parameters]
    else:
        included_parameters = parameters
    xs = np.zeros([len(included_parameters), len(samples[parameters[0]])])
    for num, i in enumerate(included_parameters):
        xs[num] = samples[i]
    default_kwargs['range'] = [1.0] * len(included_parameters)
    default_kwargs["labels"] = [latex_labels[i] for i in included_parameters]
    figure = corner.corner(xs.T, **default_kwargs)
    # grab the axes of the subplots
    axes = figure.get_axes()
    axes_of_interest = axes[:2]
    location = []
    for i in axes_of_interest:
        extent = i.get_window_extent().transformed(
            figure.dpi_scale_trans.inverted()
        )
        location.append([extent.x0 * figure.dpi, extent.y0 * figure.dpi])
    width, height = extent.width, extent.height
    width *= figure.dpi
    height *= figure.dpi
    try:
        seperation = abs(location[0][0] - location[1][0]) - width
    except IndexError:
        seperation = None
    data = {
        "width": width, "height": height, "seperation": seperation,
        "x0": location[0][0], "y0": location[0][0]
    }
    return figure, included_parameters, data
