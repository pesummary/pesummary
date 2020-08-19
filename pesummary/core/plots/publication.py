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

import numpy as np
from matplotlib import gridspec
from scipy.stats import gaussian_kde
import corner
import copy

from pesummary.core.plots.figure import figure
from pesummary import conf


def _triangle_axes(
    figsize=(8, 8), width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0,
    hspace=0.0,
):
    """Initialize the axes for a 2d triangle plot

    Parameters
    ----------
    figsize: tuple, optional
        figure size you wish to use. Default (8, 8)
    width_ratios: list, optional
        ratio of widths for the triangular axis. Default 4:1
    height_ratios: list, optional
        ratio of heights for the triangular axis. Default 1:4
    wspace: float, optional
        horizontal space between the axis. Default 0.0
    hspace: float, optional
        vertical space between the axis. Default 0.0
    """
    high1d = 1.0
    fig = figure(figsize=figsize, gca=False)
    gs = gridspec.GridSpec(
        2, 2, width_ratios=width_ratios, height_ratios=height_ratios,
        wspace=wspace, hspace=hspace
    )
    ax1, ax2, ax3, ax4 = (
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
        fig.add_subplot(gs[3]),
    )
    ax1.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()
    ax1.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])
    return fig, ax1, ax2, ax3, ax4


def triangle_plot(
    x, y, kde=gaussian_kde, npoints=100, kde_kwargs={}, fill=True,
    fill_alpha=0.5, levels=[0.9], smooth=7, colors=list(conf.colorcycle),
    xlabel=None, ylabel=None, fontsize={"legend": 12, "label": 12},
    linestyles=None, linewidths=None, plot_density=True,
    percentiles=None, percentile_plot=None, fig_kwargs={}, labels=None,
    rangex=None, rangey=None, grid=False, latex_friendly=False, **kwargs
):
    """Generate a triangular plot made of 3 axis. One central axis showing the
    2d marginalized posterior and two smaller axes showing the marginalized 1d
    posterior distribution (above and to the right of central axis)

    Parameters
    ----------
    x: list
        list of samples for the x axis
    y: list
        list of samples for the y axis
    kde: Bool/func, optional
        kde to use for smoothing the 1d marginalized posterior distribution. If
        you do not want to use KDEs, simply pass kde=False. Default
        scipy.stats.gaussian_kde
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function
    fill: Bool, optional
        whether or not to fill the 1d posterior distributions
    fill_alpha: float, optional
        alpha to use for fill
    levels: list, optional
        levels you wish to use for the 2d contours
    smooth: dict/float, optional
        how much smoothing you wish to use for the 2d contours. If you wish
        to use different smoothing for different contours, then provide a dict
        with keys given by the label
    colors: list, optional
        list of colors you wish to use for each analysis
    xlabel: str, optional
        xlabel you wish to use for the plot
    ylabel: str, optional
        ylabel you wish to use for the plot
    fontsize: dict, optional
        dictionary giving the fontsize for the labels and legend. Default
        {'legend': 12, 'label': 12}
    linestyles: list, optional
        linestyles you wish to use for each analysis
    linewidths: list, optional
        linewidths you wish to use for each analysis
    plot_density: Bool, optional
        whether or not to plot the density on the 2d contour. Default True
    percentiles: list, optional
        percentiles you wish to plot. Default None
    percentile_plot: list, optional
        list of analyses to plot percentiles. Default all analyses
    fig_kwargs: dict, optional
        optional kwargs passed directly to the _triangle_axes function
    labels: list, optional
        label associated with each set of samples
    rangex: tuple, optional
        range over which to plot the x axis
    rangey: tuple, optional
        range over which to plot the y axis
    grid: Bool, optional
        if True, show a grid on all axes. Default False
    **kwargs: dict
        all additional kwargs are passed to the corner.hist2d function
    """
    fig, ax1, ax2, ax3, ax4 = _triangle_axes(**fig_kwargs)
    ax2.axis("off")
    return _triangle_plot(
        fig, [ax1, ax3, ax4], x, y, kde=kde, npoints=npoints,
        smooth=smooth, kde_kwargs=kde_kwargs, fill=fill, fill_alpha=fill_alpha,
        levels=levels, colors=colors, linestyles=linestyles,
        linewidths=linewidths, plot_density=plot_density,
        percentiles=percentiles, fig_kwargs=fig_kwargs, labels=labels,
        xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, rangex=rangex,
        rangey=rangey, percentile_plot=percentile_plot, grid=grid,
        latex_friendly=latex_friendly, **kwargs
    )


def _triangle_plot(
    fig, axes, x, y, kde=gaussian_kde, npoints=100, kde_kwargs={}, fill=True,
    fill_alpha=0.5, levels=[0.9], smooth=7, colors=list(conf.colorcycle),
    xlabel=None, ylabel=None, fontsize={"legend": 12, "label": 12},
    linestyles=None, linewidths=None, plot_density=True, percentiles=None,
    percentile_plot=None, fig_kwargs={}, labels=None, plot_datapoints=False,
    rangex=None, rangey=None, grid=False, latex_friendly=False, **kwargs
):
    """Base function to generate a triangular plot

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure on which to make the plots
    axes: list
        list of subplots associated with the figure
    x: list
        list of samples for the x axis
    y: list
        list of samples for the y axis
    kde: Bool/func, optional
        kde to use for smoothing the 1d marginalized posterior distribution. If
        you do not want to use KDEs, simply pass kde=False. Default
        scipy.stats.gaussian_kde
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function
    fill: Bool, optional
        whether or not to fill the 1d posterior distributions
    fill_alpha: float, optional
        alpha to use for fill
    levels: list, optional
        levels you wish to use for the 2d contours
    smooth: dict/float, optional
        how much smoothing you wish to use for the 2d contours. If you wish
        to use different smoothing for different contours, then provide a dict
        with keys given by the label
    colors: list, optional
        list of colors you wish to use for each analysis
    xlabel: str, optional
        xlabel you wish to use for the plot
    ylabel: str, optional
        ylabel you wish to use for the plot
    fontsize: dict, optional
        dictionary giving the fontsize for the labels and legend. Default
        {'legend': 12, 'label': 12}
    linestyles: list, optional
        linestyles you wish to use for each analysis
    linewidths: list, optional
        linewidths you wish to use for each analysis
    plot_density: Bool, optional
        whether or not to plot the density on the 2d contour. Default True
    percentiles: list, optional
        percentiles you wish to plot. Default None
    percentile_plot: list, optional
        list of analyses to plot percentiles. Default all analyses
    fig_kwargs: dict, optional
        optional kwargs passed directly to the _triangle_axes function
    labels: list, optional
        label associated with each set of samples
    rangex: tuple, optional
        range over which to plot the x axis
    rangey: tuple, optional
        range over which to plot the y axis
    grid: Bool, optional
        if True, show a grid on all axes
    **kwargs: dict
        all kwargs are passed to the corner.hist2d function
    """
    ax1, ax3, ax4 = axes
    if not isinstance(x[0], (list, np.ndarray)):
        x, y = np.atleast_2d(x), np.atleast_2d(y)
    _base_error = "Please provide {} for each analysis"
    if len(colors) < len(x):
        raise ValueError(_base_error.format("a single color"))
    if linestyles is None:
        linestyles = ["-"] * len(x)
    elif len(linestyles) < len(x):
        raise ValueError(_base_error.format("a single linestyle"))
    if linewidths is None:
        linewidths = [None] * len(x)
    elif len(linewidths) < len(x):
        raise ValueError(_base_error.format("a single linewidth"))
    if labels is None:
        labels = [None] * len(x)
    elif len(labels) != len(x):
        raise ValueError(_base_error.format("a label"))

    xlow = np.min([np.min(_x) for _x in x])
    xhigh = np.max([np.max(_x) for _x in x])
    ylow = np.min([np.min(_y) for _y in y])
    yhigh = np.max([np.max(_y) for _y in y])
    if rangex is not None:
        xlow, xhigh = rangex
    if rangey is not None:
        ylow, yhigh = rangey
    for num in range(len(x)):
        plot_kwargs = dict(
            color=colors[num], linewidth=linewidths[num],
            linestyle=linestyles[num]
        )
        if kde:
            _kde = kde(x[num], **kde_kwargs)
            _x = np.linspace(xlow, xhigh, npoints)
            _y = _kde(_x)
            ax1.plot(_x, _y, **plot_kwargs)
            if fill:
                ax1.fill_between(_x, 0, _y, alpha=fill_alpha, **plot_kwargs)
            if percentiles is not None:
                if percentile_plot is not None and labels[num] in percentile_plot:
                    _percentiles = np.percentile(x[num], percentiles)
                    ax1.axvline(_percentiles[0], **plot_kwargs)
                    ax1.axvline(_percentiles[1], **plot_kwargs)
            _y = np.linspace(ylow, yhigh, npoints)
            _kde = kde(y[num], **kde_kwargs)
            _x = _kde(_y)
            if latex_friendly:
                labels = copy.deepcopy(labels)
                labels[num] = labels[num].replace("_", "\_")
            ax4.plot(_x, _y, label=labels[num], **plot_kwargs)
            if fill:
                ax4.fill_betweenx(_y, 0, _x, alpha=fill_alpha, **plot_kwargs)
            if percentiles is not None:
                if percentile_plot is not None and labels[num] in percentile_plot:
                    _percentiles = np.percentile(y[num], percentiles)
                    ax4.axhline(_percentiles[0], **plot_kwargs)
                    ax4.axhline(_percentiles[1], **plot_kwargs)
        else:
            if fill:
                histtype = "stepfilled"
            else:
                histtype = "step"
            ax1.hist(x[num], histtype=histtype, **plot_kwargs)
            ax4.hist(
                y[num], histtype=histtype, orientation="horizontal",
                **plot_kwargs
            )
        if isinstance(smooth, dict):
            _smooth = smooth[labels[num]]
        else:
            _smooth = smooth
        corner.hist2d(
            x[num], y[num], bins=300, ax=ax3, levels=levels, smooth=_smooth,
            range=[[xlow, xhigh], [ylow, yhigh]], color=colors[num],
            plot_density=plot_density, contour_kwargs=dict(
                linestyles=[linestyles[num]], linewidths=linewidths[num]
            ), plot_datapoints=plot_datapoints, **kwargs
        )
    if xlabel is not None:
        ax3.set_xlabel(xlabel, fontsize=fontsize["label"])
    if ylabel is not None:
        ax3.set_ylabel(ylabel, fontsize=fontsize["label"])
    if not all(label is None for label in labels):
        ax3.legend(
            *ax4.get_legend_handles_labels(), loc="best", frameon=False,
            fontsize=fontsize["legend"]
        )
    ax1.grid(grid)
    ax3.grid(grid)
    ax4.grid(grid)
    return fig, ax1, ax3, ax4


def reverse_triangle_plot(
    x, y, kde=gaussian_kde, npoints=100, kde_kwargs={}, fill=True,
    fill_alpha=0.5, levels=[0.9], smooth=7, colors=list(conf.colorcycle),
    xlabel=None, ylabel=None, fontsize={"legend": 12, "label": 12},
    linestyles=None, linewidths=None, plot_density=True,
    percentiles=None, percentile_plot=None, fig_kwargs={}, labels=None,
    plot_datapoints=False, rangex=None, rangey=None, grid=False,
    latex_friendly=False, **kwargs
):
    """Generate a triangular plot made of 3 axis. One central axis showing the
    2d marginalized posterior and two smaller axes showing the marginalized 1d
    posterior distribution (below and to the left of central axis). Only two
    axes are plotted, each below the 1d marginalized posterior distribution

    Parameters
    ----------
    x: list
        list of samples for the x axis
    y: list
        list of samples for the y axis
    kde: Bool/func, optional
        kde to use for smoothing the 1d marginalized posterior distribution. If
        you do not want to use KDEs, simply pass kde=False. Default
        scipy.stats.gaussian_kde
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function
    fill: Bool, optional
        whether or not to fill the 1d posterior distributions
    fill_alpha: float, optional
        alpha to use for fill
    levels: list, optional
        levels you wish to use for the 2d contours
    smooth: dict/float, optional
        how much smoothing you wish to use for the 2d contours. If you wish
        to use different smoothing for different contours, then provide a dict
        with keys given by the label
    colors: list, optional
        list of colors you wish to use for each analysis
    xlabel: str, optional
        xlabel you wish to use for the plot
    ylabel: str, optional
        ylabel you wish to use for the plot
    fontsize: dict, optional
        dictionary giving the fontsize for the labels and legend. Default
        {'legend': 12, 'label': 12}
    linestyles: list, optional
        linestyles you wish to use for each analysis
    linewidths: list, optional
        linewidths you wish to use for each analysis
    plot_density: Bool, optional
        whether or not to plot the density on the 2d contour. Default True
    percentiles: list, optional
        percentiles you wish to plot. Default None
    percentile_plot: list, optional
        list of analyses to plot percentiles. Default all analyses
    fig_kwargs: dict, optional
        optional kwargs passed directly to the _triangle_axes function
    labels: list, optional
        label associated with each set of samples
    rangex: tuple, optional
        range over which to plot the x axis
    rangey: tuple, optional
        range over which to plot the y axis
    **kwargs: dict
        all kwargs are passed to the corner.hist2d function
    """
    fig, ax1, ax2, ax3, ax4 = _triangle_axes(
        width_ratios=[1, 4], height_ratios=[4, 1]
    )
    ax3.axis("off")
    fig, ax4, ax2, ax1 = _triangle_plot(
        fig, [ax4, ax2, ax1], x, y, kde=kde, npoints=npoints, smooth=smooth,
        kde_kwargs=kde_kwargs, fill=fill, fill_alpha=fill_alpha,
        levels=levels, colors=colors, linestyles=linestyles,
        linewidths=linewidths, plot_density=plot_density,
        percentiles=percentiles, fig_kwargs=fig_kwargs, labels=labels,
        fontsize=fontsize, plot_datapoints=plot_datapoints, rangex=rangex,
        rangey=rangey, percentile_plot=percentile_plot,
        latex_friendly=latex_friendly, **kwargs
    )
    ax2.axis("off")
    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.set_yticks([])

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_xticks([])

    if xlabel is not None:
        ax4.set_xlabel(xlabel, fontsize=fontsize["label"])
    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=fontsize["label"])
    return fig, ax1, ax2, ax4
