# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from matplotlib import gridspec
from scipy.stats import gaussian_kde
import copy

from pesummary.core.plots.figure import figure
from .corner import hist2d
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
DEFAULT_LEGEND_KWARGS = {"loc": "best", "frameon": False}


def pcolormesh(
    x, y, density, ax=None, levels=None, smooth=None, bins=None, label=None,
    level_kwargs={}, range=None, grid=True, legend=False, legend_kwargs={},
    **kwargs
):
    """Generate a colormesh plot on a given axis

    Parameters
    ----------
    x: np.ndarray
        array of floats for the x axis
    y: np.ndarray
        array of floats for the y axis
    density: np.ndarray
        2d array of probabilities
    ax: matplotlib.axes._subplots.AxesSubplot, optional
        axis you wish to use for plotting
    levels: list, optional
        contour levels to show on the plot. Default None
    smooth: float, optional
        sigma to use for smoothing. Default, no smoothing applied
    level_kwargs: dict, optional
        optional kwargs to use for ax.contour
    **kwargs: dict, optional
        all additional kwargs passed to ax.pcolormesh
    """
    if smooth is not None:
        import scipy.ndimage.filters as filter
        density = filter.gaussian_filter(density, sigma=smooth)
    _cmap = kwargs.get("cmap", None)
    _off = False
    if _cmap is not None and isinstance(_cmap, str) and _cmap.lower() == "off":
        _off = True
    if grid and "zorder" not in kwargs:
        _zorder = -10
    else:
        _zorder = kwargs.pop("zorder", 10)
    if not _off:
        ax.pcolormesh(x, y, density, zorder=_zorder, **kwargs)
    if levels is not None:
        CS = ax.contour(x, y, density, levels=levels, **level_kwargs)
        if legend:
            _legend_kwargs = DEFAULT_LEGEND_KWARGS.copy()
            _legend_kwargs.update(legend_kwargs)
            CS.collections[0].set_label(label)
            ax.legend(**_legend_kwargs)
    return ax


def analytic_twod_contour_plot(*args, smooth=None, **kwargs):
    """Generate a 2d contour plot given an analytic PDF

    Parameters
    ----------
    *args: tuple
        all args passed to twod_contour_plot
    smooth: float, optional
        degree of smoothing to apply to probabilities
    **kwargs: dict, optional
        all additional kwargs passed to twod_contour_plot
    """
    return twod_contour_plot(
        *args, smooth=smooth, _function=pcolormesh, **kwargs
    )


def twod_contour_plot(
    x, y, *args, rangex=None, rangey=None, fig=None, ax=None, return_ax=False,
    levels=[0.9], bins=300, smooth=7, xlabel=None, ylabel=None,
    fontsize={"label": 12}, grid=True, label=None, truth=None,
    _function=hist2d, truth_lines=True, truth_kwargs={},
    _default_truth_kwargs={
        "marker": 'o', "markeredgewidth": 2, "markersize": 6, "color": 'k'
    }, **kwargs
):
    """Generate a 2d contour contour plot for 2 marginalized posterior
    distributions

    Parameters
    ----------
    x: np.array
        array of posterior samples to use for the x axis
    y: np.array
        array of posterior samples to use for the y axis
    rangex: tuple, optional
        range over which to plot the x axis
    rangey: tuple, optional
        range over which to plot the y axis
    fig: matplotlib.figure.Figure, optional
        figure you wish to use for plotting
    ax: matplotlib.axes._subplots.AxesSubplot, optional
        axis you wish to use for plotting
    return_ax: Bool, optional
        if True return the axis used for plotting. Else return the figure
    levels: list, optional
        levels you wish to use for the 2d contours. Default [0.9]
    bins: int, optional
        number of bins to use for gridding 2d parameter space. Default 300
    smooth: int, optional
        how much smoothing you wish to use for the 2d contours
    xlabel: str, optional
        label to use for the xaxis
    ylabel: str, optional
        label to use for the yaxis
    fontsize: dict, optional
        dictionary containing the fontsize to use for the plot
    grid: Bool, optional
        if True, add a grid to the plot
    label: str, optional
        label to use for a given contour
    truth: list, optional
        the true value of the posterior. `truth` is a list of length 2 with
        first element being the true x value and second element being the true
        y value
    truth_lines: Bool, optional
        if True, add vertical and horizontal lines spanning the 2d space to show
        injected value
    truth_kwargs: dict, optional
        kwargs to use to indicate truth
    **kwargs: dict, optional
        all additional kwargs are passed to the
        `pesummary.core.plots.corner.hist2d` function
    """
    if fig is None and ax is None:
        fig, ax = figure(gca=True)
    elif fig is None and ax is not None:
        return_ax = True
    elif ax is None:
        ax = fig.gca()

    xlow, xhigh = np.min(x), np.max(x)
    ylow, yhigh = np.min(y), np.max(y)
    if rangex is not None:
        xlow, xhigh = rangex
    if rangey is not None:
        ylow, yhigh = rangey
    if "range" not in list(kwargs.keys()):
        kwargs["range"] = [[xlow, xhigh], [ylow, yhigh]]

    _function(
        x, y, *args, ax=ax, levels=levels, bins=bins, smooth=smooth,
        label=label, grid=grid, **kwargs
    )
    if truth is not None:
        _default_truth_kwargs.update(truth_kwargs)
        ax.plot(*truth, **_default_truth_kwargs)
        if truth_lines:
            ax.axvline(
                truth[0], color=_default_truth_kwargs["color"], linewidth=0.5
            )
            ax.axhline(
                truth[1], color=_default_truth_kwargs["color"], linewidth=0.5
            )
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize["label"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize["label"])
    ax.grid(grid)
    if fig is not None:
        fig.tight_layout()
    if return_ax:
        return ax
    return fig


def comparison_twod_contour_plot(
    x, y, labels=None, plot_density=None, rangex=None, rangey=None,
    legend_kwargs={"loc": "best", "frameon": False},
    colors=list(conf.colorcycle), linestyles=None, **kwargs
):
    """Generate a comparison 2d contour contour plot for 2 marginalized
    posterior distributions from multiple analyses

    Parameters
    ----------
    x: np.ndarray
        2d array of posterior samples to use for the x axis; array for each
        analysis
    y: np.ndarray
        2d array of posterior samples to use for the y axis; array for each
        analysis
    labels: list, optional
        labels to assign to each contour
    plot_density: str, optional
        label of the analysis you wish to plot the density for. If you wish
        to plot both, simply pass `plot_density='both'`
    rangex: tuple, optional
        range over which to plot the x axis
    rangey: tuple, optional
        range over which to plot the y axis
    legend_kwargs: dict, optional
        kwargs to use for the legend
    colors: list, optional
        list of colors to use for each contour
    linestyles: list, optional
        linestyles to use for each contour
    **kwargs: dict, optional
        all additional kwargs are passed to the
        `pesummary.core.plots.publication.twod_contour_plot` function
    """
    if labels is None and plot_density is not None:
        plot_density = None
    if labels is None:
        labels = [None] * len(x)

    xlow = np.min([np.min(_x) for _x in x])
    xhigh = np.max([np.max(_x) for _x in x])
    ylow = np.min([np.min(_y) for _y in y])
    yhigh = np.max([np.max(_y) for _y in y])
    if rangex is None:
        rangex = [xlow, xhigh]
    if rangey is None:
        rangey = [ylow, yhigh]

    fig = None
    for num, (_x, _y) in enumerate(zip(x, y)):
        if plot_density is not None and plot_density == labels[num]:
            plot_density = True
        elif plot_density is not None and isinstance(plot_density, list):
            if labels[num] in plot_density:
                plot_density = True
            else:
                plot_density = False
        elif plot_density is not None and plot_density == "both":
            plot_density = True
        else:
            plot_density = False

        _label = _color = _linestyle = None
        if labels is not None:
            _label = labels[num]
        if colors is not None:
            _color = colors[num]
        if linestyles is not None:
            _linestyle = linestyles[num]
        fig = twod_contour_plot(
            _x, _y, plot_density=plot_density, label=_label, fig=fig,
            rangex=rangex, rangey=rangey, color=_color, linestyles=_linestyle,
            **kwargs
        )
    ax = fig.gca()
    legend = ax.legend(**legend_kwargs)
    return fig


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


def _generate_triangle_plot(
    *args, function=None, fig_kwargs={}, existing_figure=None, **kwargs
):
    """Generate a triangle plot according to a given function

    Parameters
    ----------
    *args: tuple
        all args passed to function
    function: func, optional
        function you wish to use to generate triangle plot. Default
        _triangle_plot
    **kwargs: dict, optional
        all kwargs passed to function
    """
    if existing_figure is None:
        fig, ax1, ax2, ax3, ax4 = _triangle_axes(**fig_kwargs)
        ax2.axis("off")
    else:
        fig, ax1, ax3, ax4 = existing_figure
    if function is None:
        function = _triangle_plot
    return function(fig, [ax1, ax3, ax4], *args, **kwargs)


def triangle_plot(*args, **kwargs):
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
    kde_2d: func, optional
        kde to use for smoothing the 2d marginalized posterior distribution.
        default None
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function
    kde_2d_kwargs: dict, optional
        optional kwargs which are passed directly to the 2d kde function
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
    legend_kwargs: dict, optional
        optional kwargs for the legend. Default {"loc": "best", "frameon": False}
    **kwargs: dict
        all additional kwargs are passed to the corner.hist2d function
    """
    return _generate_triangle_plot(*args, function=_triangle_plot, **kwargs)


def analytic_triangle_plot(*args, **kwargs):
    """Generate a triangle plot given probability densities for x, y and xy.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure on which to make the plots
    axes: list
        list of subplots associated with the figure
    x: list
        list of points to use for the x axis
    y: list
        list of points to use for the y axis
    prob_x: list
        list of probabilities associated with x
    prob_y: list
        list of probabilities associated with y
    probs_xy: list
        2d list of probabilities for xy
    smooth: float, optional
        degree of smoothing to apply to probs_xy. Default no smoothing applied
    cmap: str, optional
        name of cmap to use for plotting
    """
    return _generate_triangle_plot(
        *args, function=_analytic_triangle_plot, **kwargs
    )


def _analytic_triangle_plot(
    fig, axes, x, y, probs_x, probs_y, probs_xy, smooth=None, xlabel=None,
    ylabel=None, grid=True, **kwargs
):
    """Generate a triangle plot given probability densities for x, y and xy.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure on which to make the plots
    axes: list
        list of subplots associated with the figure
    x: list
        list of points to use for the x axis
    y: list
        list of points to use for the y axis
    prob_x: list
        list of probabilities associated with x
    prob_y: list
        list of probabilities associated with y
    probs_xy: list
        2d list of probabilities for xy
    smooth: float, optional
        degree of smoothing to apply to probs_xy. Default no smoothing applied
    xlabel: str, optional
        label to use for the x axis
    ylabel: str, optional
        label to use for the y axis
    grid: Bool, optional
        if True, add a grid to the plot
    """
    ax1, ax3, ax4 = axes
    analytic_twod_contour_plot(
        x, y, probs_xy, ax=ax3, smooth=smooth, grid=grid, **kwargs
    )
    level_kwargs = kwargs.get("level_kwargs", None)
    if level_kwargs is not None and "colors" in level_kwargs.keys():
        color = level_kwargs["colors"][0]
    else:
        color = None
    ax1.plot(x, probs_x, color=color)
    ax4.plot(probs_y, y, color=color)
    fontsize = kwargs.get("fontsize", {"label": 12})
    if xlabel is not None:
        ax3.set_xlabel(xlabel, fontsize=fontsize["label"])
    if ylabel is not None:
        ax3.set_ylabel(ylabel, fontsize=fontsize["label"])
    ax1.grid(grid)
    if grid:
        ax3.grid(grid, zorder=10)
    ax4.grid(grid)
    xlims = ax3.get_xlim()
    ax1.set_xlim(xlims)
    ylims = ax3.get_ylim()
    ax4.set_ylim(ylims)
    fig.tight_layout()
    return fig, ax1, ax3, ax4


def _triangle_plot(
    fig, axes, x, y, kde=gaussian_kde, npoints=100, kde_kwargs={}, fill=True,
    fill_alpha=0.5, levels=[0.9], smooth=7, colors=list(conf.colorcycle),
    xlabel=None, ylabel=None, fontsize={"legend": 12, "label": 12},
    linestyles=None, linewidths=None, plot_density=True, percentiles=None,
    percentile_plot=None, fig_kwargs={}, labels=None, plot_datapoints=False,
    rangex=None, rangey=None, grid=False, latex_friendly=False, kde_2d=None,
    kde_2d_kwargs={}, legend_kwargs={"loc": "best", "frameon": False},
    truth=None, _contour_function=twod_contour_plot, **kwargs
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
    kde_2d: func, optional
        kde to use for smoothing the 2d marginalized posterior distribution.
        default None
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function.
        kde_kwargs to be passed to the kde on the y axis may be specified
        by the dictionary entry 'y_axis'. kde_kwargs to be passed to the kde on
        the x axis may be specified by the dictionary entry 'x_axis'.
    kde_2d_kwargs: dict, optional
        optional kwargs which are passed directly to the 2d kde function
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
    legend_kwargs: dict, optional
        optional kwargs for the legend. Default {"loc": "best", "frameon": False}
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
            if "x_axis" in kde_kwargs.keys():
                _kde = kde(x[num], **kde_kwargs["x_axis"])
            else:
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
            if "y_axis" in kde_kwargs.keys():
                _kde = kde(y[num], **kde_kwargs["y_axis"])
            else:
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
        _contour_function(
            x[num], y[num], ax=ax3, levels=levels, smooth=_smooth,
            rangex=[xlow, xhigh], rangey=[ylow, yhigh], color=colors[num],
            linestyles=linestyles[num],
            plot_density=plot_density, contour_kwargs=dict(
                linestyles=[linestyles[num]], linewidths=linewidths[num]
            ), plot_datapoints=plot_datapoints, kde=kde_2d,
            kde_kwargs=kde_2d_kwargs, grid=False, truth=truth, **kwargs
        )

    if truth is not None:
        ax1.axvline(truth[0], color='k', linewidth=0.5)
        ax4.axhline(truth[1], color='k', linewidth=0.5)
    if xlabel is not None:
        ax3.set_xlabel(xlabel, fontsize=fontsize["label"])
    if ylabel is not None:
        ax3.set_ylabel(ylabel, fontsize=fontsize["label"])
    if not all(label is None for label in labels):
        legend_kwargs["fontsize"] = fontsize["legend"]
        ax3.legend(*ax4.get_legend_handles_labels(), **legend_kwargs)
    ax1.grid(grid)
    ax3.grid(grid)
    ax4.grid(grid)
    xlims = ax1.get_xlim()
    ax3.set_xlim(xlims)
    ylims = ax4.get_ylim()
    ax3.set_ylim(ylims)
    return fig, ax1, ax3, ax4


def _generate_reverse_triangle_plot(
    *args, xlabel=None, ylabel=None, function=None, existing_figure=None, **kwargs
):
    """Generate a reverse triangle plot according to a given function

    Parameters
    ----------
    *args: tuple
        all args passed to function
    xlabel: str, optional
        label to use for the x axis
    ylabel: str, optional
        label to use for the y axis
    function: func, optional
        function to use to generate triangle plot. Default _triangle_plot
    **kwargs: dict, optional
        all kwargs passed to function
    """
    if existing_figure is None:
        fig, ax1, ax2, ax3, ax4 = _triangle_axes(
            width_ratios=[1, 4], height_ratios=[4, 1]
        )
        ax3.axis("off")
    else:
        fig, ax1, ax2, ax4 = existing_figure
    if function is None:
        function = _triangle_plot
    fig, ax4, ax2, ax1 = function(fig, [ax4, ax2, ax1], *args, **kwargs)
    ax2.axis("off")
    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.set_yticks([])

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_xticks([])

    _fontsize = kwargs.get("fontsize", {"label": 12})["label"]
    if xlabel is not None:
        ax4.set_xlabel(xlabel, fontsize=_fontsize)
    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=_fontsize)
    return fig, ax1, ax2, ax4


def reverse_triangle_plot(*args, **kwargs):
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
    kde_2d: func, optional
        kde to use for smoothing the 2d marginalized posterior distribution.
        default None
    npoints: int, optional
        number of points to use for the 1d kde
    kde_kwargs: dict, optional
        optional kwargs which are passed directly to the kde function.
        kde_kwargs to be passed to the kde on the y axis may be specified
        by the dictionary entry 'y_axis'. kde_kwargs to be passed to the kde on
        the x axis may be specified by the dictionary entry 'x_axis'.
    kde_2d_kwargs: dict, optional
        optional kwargs which are passed directly to the 2d kde function
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
    legend_kwargs: dict, optional
        optional kwargs for the legend. Default {"loc": "best", "frameon": False}
    **kwargs: dict
        all kwargs are passed to the corner.hist2d function
    """
    return _generate_reverse_triangle_plot(
        *args, function=_triangle_plot, **kwargs
    )


def analytic_reverse_triangle_plot(*args, **kwargs):
    """Generate a triangle plot given probability densities for x, y and xy.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure on which to make the plots
    axes: list
        list of subplots associated with the figure
    x: list
        list of points to use for the x axis
    y: list
        list of points to use for the y axis
    prob_x: list
        list of probabilities associated with x
    prob_y: list
        list of probabilities associated with y
    probs_xy: list
        2d list of probabilities for xy
    smooth: float, optional
        degree of smoothing to apply to probs_xy. Default no smoothing applied
    cmap: str, optional
        name of cmap to use for plotting
    """
    return _generate_reverse_triangle_plot(
        *args, function=_analytic_triangle_plot, **kwargs
    )
