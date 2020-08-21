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
    get_matplotlib_style_file, gelman_rubin,
)
from pesummary.core.plots.kde import kdeplot
from pesummary.core.plots.figure import figure, subplots, ExistingFigure
from pesummary import conf

import matplotlib.style
import matplotlib.lines as mlines
import corner
import copy
from itertools import cycle

import numpy as np
from scipy import signal

matplotlib.style.use(get_matplotlib_style_file())
_check_latex_install()

_default_legend_kwargs = dict(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, handlelength=3, mode="expand",
    borderaxespad=0.0,
)


def _autocorrelation_plot(
    param, samples, fig=None, color=conf.color, markersize=0.5, grid=True
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
    grid: Bool, optional
        if True, plot a grid
    """
    logger.debug("Generating the autocorrelation function for %s" % (param))
    if fig is None:
        fig, ax = figure(gca=True)
    else:
        ax = fig.gca()
    samples = samples[int(len(samples) / 2):]
    x = samples - np.mean(samples)
    y = np.conj(x[::-1])
    acf = np.fft.ifftshift(signal.fftconvolve(y, x, mode="full"))
    N = np.array(samples).shape[0]
    acf = acf[0:N]
    ax.plot(
        acf / acf[0], linestyle=" ", marker="o", markersize=markersize,
        color=color
    )
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.grid(b=grid)
    fig.tight_layout()
    return fig


def _autocorrelation_plot_mcmc(
    param, samples, colorcycle=conf.colorcycle, grid=True
):
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
    grid: Bool, optional
        if True, plot a grid
    """
    cycol = cycle(colorcycle)
    fig, ax = figure(gca=True)
    for ss in samples:
        fig = _autocorrelation_plot(
            param, ss, fig=fig, markersize=1.25, color=next(cycol), grid=grid
        )
    return fig


def _sample_evolution_plot(
    param, samples, latex_label, inj_value=None, fig=None, color=conf.color,
    markersize=0.5, grid=True
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
    grid: Bool, optional
        if True, plot a grid
    """
    logger.debug("Generating the sample scatter plot for %s" % (param))
    if fig is None:
        fig, ax = figure(gca=True)
    else:
        ax = fig.gca()
    n_samples = len(samples)
    ax.plot(
        range(n_samples), samples, linestyle=" ", marker="o",
        markersize=markersize, color=color
    )
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xlabel("samples")
    ax.set_ylabel(latex_label)
    ax.grid(b=grid)
    fig.tight_layout()
    return fig


def _sample_evolution_plot_mcmc(
    param, samples, latex_label, inj_value=None, colorcycle=conf.colorcycle,
    grid=True
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
    grid: Bool, optional
        if True, plot a grid
    """
    cycol = cycle(colorcycle)
    fig, ax = figure(gca=True)
    for ss in samples:
        fig = _sample_evolution_plot(
            param, ss, latex_label, inj_value=None, fig=fig, markersize=1.25,
            color=next(cycol), grid=grid
        )
    return fig


def _1d_cdf_plot(
    param, samples, latex_label, fig=None, color=conf.color, title=True,
    grid=True
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
    grid: Bool, optional
        if True, plot a grid
    """
    logger.debug("Generating the 1d CDF for %s" % (param))
    if fig is None:
        fig, ax = figure(gca=True)
    else:
        ax = fig.gca()
    sorted_samples = copy.deepcopy(samples)
    sorted_samples.sort()
    ax.set_xlabel(latex_label)
    ax.set_ylabel("Cumulative Density Function")
    upper_percentile = np.percentile(samples, 95)
    lower_percentile = np.percentile(samples, 5)
    median = np.median(samples)
    upper = np.round(upper_percentile - median, 2)
    lower = np.round(median - lower_percentile, 2)
    median = np.round(median, 2)
    if title:
        ax.set_title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower))
    ax.plot(sorted_samples, np.linspace(0, 1, len(sorted_samples)), color=color)
    ax.grid(b=grid)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    return fig


def _1d_cdf_plot_mcmc(
    param, samples, latex_label, colorcycle=conf.colorcycle, grid=True
):
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
    grid: Bool, optional
        if True, plot a grid
    """
    cycol = cycle(colorcycle)
    fig, ax = figure(gca=True)
    for ss in samples:
        fig = _1d_cdf_plot(
            param, ss, latex_label, fig=fig, color=next(cycol), title=False,
            grid=grid
        )
    gelman = gelman_rubin(samples)
    ax.set_title("Gelman-Rubin: {}".format(gelman))
    return fig


def _1d_cdf_comparison_plot(
    param, samples, colors, latex_label, labels, linestyles=None, grid=True,
    legend_kwargs=_default_legend_kwargs, latex_friendly=False
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
    grid: Bool, optional
        if True, plot a grid
    legend_kwargs: dict, optional
        optional kwargs to pass to ax.legend()
    latex_friendly: Bool, optional
        if True, make the label latex friendly. Default False
    """
    logger.debug("Generating the 1d comparison CDF for %s" % (param))
    if linestyles is None:
        linestyles = ["-"] * len(samples)
    fig, ax = figure(figsize=(8, 6), gca=True)
    handles = []
    for num, i in enumerate(samples):
        fig = _1d_cdf_plot(
            param, i, latex_label, fig=fig, color=colors[num], title=False,
            grid=grid
        )
        if latex_friendly:
            labels = copy.deepcopy(labels)
            labels[num] = labels[num].replace("_", "\_")
        handles.append(mlines.Line2D([], [], color=colors[num], label=labels[num]))
    ncols = number_of_columns_for_legend(labels)
    legend = ax.legend(handles=handles, ncol=ncols, **legend_kwargs)
    for num, legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(1.75)
        legobj.set_linestyle(linestyles[num])
    ax.set_xlabel(latex_label)
    ax.set_ylabel("Cumulative Density Function")
    ax.grid(b=grid)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    return fig


def _1d_histogram_plot(
    param, samples, latex_label, inj_value=None, kde=False, prior=None,
    weights=None, xlow=None, xhigh=None, fig=None, title=True, color=conf.color,
    autoscale=True, bins=50, histtype="step", grid=True
):
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
    grid: Bool, optional
        if True, plot a grid
    """
    from pesummary.utils.samples_dict import Array

    logger.debug("Generating the 1d histogram plot for %s" % (param))
    samples = Array(samples)
    if fig is None:
        fig, ax = figure(gca=True)
    else:
        ax = fig.gca()
    if np.ptp(samples) == 0:
        ax.axvline(samples[0], color=conf.color)
        xlims = ax.get_xlim()
    elif not kde:
        ax.hist(
            samples, histtype=histtype, bins=bins, color=color, density=True,
            linewidth=1.75, weights=weights
        )
        xlims = ax.get_xlim()
        if prior is not None:
            ax.hist(
                prior, color=conf.prior_color, alpha=0.2, edgecolor="w",
                density=True, linewidth=1.75, histtype="bar", bins=bins,
            )
    else:
        kwargs = {"shade": True, "alpha_shade": 0.1, "linewidth": 1.0}
        if xlow is not None or xhigh is not None:
            kwargs.update({"xlow": xlow, "xhigh": xhigh})
        else:
            kwargs.update({"clip": [samples.minimum, samples.maximum]})
        x = kdeplot(samples, color=color, ax=ax, **kwargs)
        xlims = ax.get_xlim()
        if prior is not None:
            kdeplot(prior, color=conf.prior_color, ax=ax, **kwargs)
    ax.set_xlabel(latex_label)
    ax.set_ylabel("Probability Density")
    percentile = samples.confidence_interval([5, 95])
    if inj_value is not None:
        ax.axvline(
            inj_value, color=conf.injection_color, linestyle="-", linewidth=2.5
        )
    ax.axvline(percentile[0], color=color, linestyle="--", linewidth=1.75)
    ax.axvline(percentile[1], color=color, linestyle="--", linewidth=1.75)
    median = samples.average("median")
    if title:
        upper = np.round(percentile[1] - median, 2)
        lower = np.round(median - percentile[0], 2)
        median = np.round(median, 2)
        ax.set_title(r"$%s^{+%s}_{-%s}$" % (median, upper, lower))
    ax.grid(b=grid)
    if autoscale:
        ax.set_xlim(xlims)
    fig.tight_layout()
    return fig


def _1d_histogram_plot_mcmc(
    param, samples, latex_label, inj_value=None, kde=False, prior=None,
    weights=None, xlow=None, xhigh=None, colorcycle=conf.colorcycle, grid=True
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
    grid: Bool, optional
        if True, plot a grid
    """
    cycol = cycle(colorcycle)
    fig, ax = figure(gca=True)
    for ss in samples:
        fig = _1d_histogram_plot(
            param, ss, latex_label, inj_value=inj_value, kde=kde, prior=prior,
            weights=weights, xlow=xlow, xhigh=xhigh, fig=fig, color=next(cycol),
            title=False, autoscale=False, grid=grid
        )
    gelman = gelman_rubin(samples)
    ax.set_title("Gelman-Rubin: {}".format(gelman))
    return fig


def _1d_comparison_histogram_plot(
    param, samples, colors, latex_label, labels, kde=False, linestyles=None,
    xlow=None, xhigh=None, max_vline=1, figsize=(8, 6), grid=True,
    legend_kwargs=_default_legend_kwargs, latex_friendly=False
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
    grid: Bool, optional
        if True, plot a grid
    legend_kwargs: dict, optional
        optional kwargs to pass to ax.legend()
    latex_friendly: Bool, optional
        if True, make the label latex friendly. Default False
    """
    logger.debug("Generating the 1d comparison histogram plot for %s" % (param))
    if linestyles is None:
        linestyles = ["-"] * len(samples)
    fig, ax = figure(figsize=figsize, gca=True)
    handles = []
    for num, i in enumerate(samples):
        if latex_friendly:
            labels = copy.deepcopy(labels)
            labels[num] = labels[num].replace("_", "\_")
        if len(set(i)) <= max_vline:
            for _ind, _sample in enumerate(set(i)):
                _label = None
                if _ind == 0:
                    _label = labels[num]
                ax.axvline(_sample, color=colors[num], label=_label)
        elif not kde:
            ax.hist(
                i, histtype="step", bins=50, color=colors[num],
                label=labels[num], linewidth=2.5, density=True,
                linestyle=linestyles[num],
            )
        else:
            kwargs = {
                "shade": True, "alpha_shade": 0.05, "linewidth": 1.5,
                "label": labels[num]
            }
            if xlow is not None or xhigh is not None:
                kwargs.update({"xlow": xlow, "xhigh": xhigh})
            else:
                kwargs.update({"clip": [np.min(i), np.max(i)]})
            kdeplot(i, color=colors[num], ax=ax, **kwargs)
        ax.axvline(
            x=np.percentile(i, 95), color=colors[num], linestyle="--",
            linewidth=2.5
        )
        ax.axvline(
            x=np.percentile(i, 5), color=colors[num], linestyle="--",
            linewidth=2.5
        )
        handles.append(mlines.Line2D([], [], color=colors[num], label=labels[num]))
    ncols = number_of_columns_for_legend(labels)
    legend = ax.legend(handles=handles, ncol=ncols, **legend_kwargs)
    for num, legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(1.75)
        legobj.set_linestyle(linestyles[num])
    ax.set_xlabel(latex_label)
    ax.set_ylabel("Probability Density")
    ax.grid(b=grid)
    fig.tight_layout()
    return fig


def _comparison_box_plot(param, samples, colors, latex_label, labels, grid=True):
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
    grid: Bool, optional
        if True, plot a grid
    """
    logger.debug("Generating the 1d comparison boxplot plot for %s" % (param))
    fig, ax = figure(gca=True)
    maximum = np.max([np.max(i) for i in samples])
    minimum = np.min([np.min(i) for i in samples])
    middle = (maximum + minimum) * 0.5
    ax.boxplot(samples, widths=0.2, vert=False, whis=np.inf, labels=labels)
    for num, i in enumerate(labels):
        ax.annotate(i, xy=(middle, 1), xytext=(middle, num + 1.0 + 0.2), ha="center")
    ax.set_yticks([])
    ax.set_xlabel(latex_label)
    fig.tight_layout()
    ax.grid(b=grid)
    return fig


def _make_corner_plot(
    samples, latex_labels, corner_parameters=None, parameters=None, **kwargs
):
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
    if parameters is None:
        parameters = list(samples.keys())
    if corner_parameters is not None:
        included_parameters = [i for i in parameters if i in corner_parameters]
    else:
        included_parameters = parameters
    xs = np.zeros([len(included_parameters), len(samples[parameters[0]])])
    for num, i in enumerate(included_parameters):
        xs[num] = samples[i]
    default_kwargs.update(kwargs)
    default_kwargs["range"] = [1.0] * len(included_parameters)
    default_kwargs["labels"] = [latex_labels[i] for i in included_parameters]

    _figure = ExistingFigure(corner.corner(xs.T, **default_kwargs))
    # grab the axes of the subplots
    axes = _figure.get_axes()
    axes_of_interest = axes[:2]
    location = []
    for i in axes_of_interest:
        extent = i.get_window_extent().transformed(_figure.dpi_scale_trans.inverted())
        location.append([extent.x0 * _figure.dpi, extent.y0 * _figure.dpi])
    width, height = extent.width, extent.height
    width *= _figure.dpi
    height *= _figure.dpi
    try:
        seperation = abs(location[0][0] - location[1][0]) - width
    except IndexError:
        seperation = None
    data = {
        "width": width, "height": height, "seperation": seperation,
        "x0": location[0][0], "y0": location[0][0]
    }
    return _figure, included_parameters, data


def _make_comparison_corner_plot(
    samples, latex_labels, corner_parameters=None, colors=conf.corner_colors,
    latex_friendly=True, **kwargs
):
    """Generate a corner plot which contains multiple datasets

    Parameters
    ----------
    samples: dict
        nested dictionary containing the label as key and SamplesDict as item
        for each dataset you wish to plot
    latex_labels: dict
        dictionary of latex labels for each parameter
    corner_parameters: list, optional
        corner parameters you wish to include in the plot
    colors: list, optional
        unique colors for each dataset
    latex_friendly: Bool, optional
        if True, make the label latex friendly. Default True
    **kwargs: dict
        all kwargs are passed to `corner.corner`
    """
    parameters = corner_parameters
    if corner_parameters is None:
        _parameters = [list(_samples.keys()) for _samples in samples.values()]
        parameters = [
            i for i in _parameters[0] if all(i in _params for _params in _parameters)
        ]
    if len(samples.keys()) > len(colors):
        raise ValueError("Please provide a unique color for each dataset")

    hist_kwargs = kwargs.get("hist_kwargs", dict())
    hist_kwargs["density"] = True
    lines = []
    for num, (label, posterior) in enumerate(samples.items()):
        if latex_friendly:
            label = copy.deepcopy(label)
            label = label.replace("_", "\_")
        lines.append(mlines.Line2D([], [], color=colors[num], label=label))
        _samples = {
            param: value for param, value in posterior.items() if param in
            parameters
        }
        hist_kwargs["color"] = colors[num]
        kwargs.update({"hist_kwargs": hist_kwargs})
        if num == 0:
            fig, _, _ = _make_corner_plot(
                _samples, latex_labels, corner_parameters=corner_parameters,
                parameters=parameters, color=colors[num], **kwargs
            )
        else:
            fig, _, _ = _make_corner_plot(
                _samples, latex_labels, corner_parameters=corner_parameters,
                fig=fig, parameters=parameters, color=colors[num], **kwargs
            )
    fig.legend(handles=lines, loc="upper right")
    lines = []
    return fig
