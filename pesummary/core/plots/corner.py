# Licensed under an MIT style license -- see LICENSE.md

import copy
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, colorConverter

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def hist2d(
    *args, kde=None, kde_kwargs={}, levels=None, contour_kwargs=None,
    weights=None, no_fill_contours=True, **kwargs
):
    """Wrapper for corner.hist2d which adds additional functionality to plot
    custom KDEs

    Parameters
    ----------
    *args: tuple
        Array(s) to plot
    kde: func, optional
        KDE function to use. Default None
    kde_kwargs: dict, optional
        kwargs to pass to the KDE function
    levels: list, optional
        Contour levels to plot
    contour_kwargs: dict, optional
        kwargs to pass to the contour function
    **kwargs: dict, optional
        kwargs to pass to corner.hist2d
    """
    from corner import hist2d as corner_hist2d
    if kde is None:
        return corner_hist2d(
            *args, levels=levels, contour_kwargs=contour_kwargs,
            weights=weights, no_fill_contours=no_fill_contours, **kwargs
        )
    from pesummary.core.plots.plot import (
        _calculate_density_with_kde_for_contour, _get_contour_levels
    )
    from pesummary.core.plots.figure import get_current_axis

    _kwargs = kwargs.copy()
    if "ax" not in kwargs.keys():
        ax = get_current_axis()
        _kwargs["ax"] = ax
    else:
        ax = _kwargs["ax"]
    _kwargs["plot_contours"] = False
    _kwargs["fill_contours"] = False
    _kwargs["no_fill_contours"] = True
    corner_hist2d(
        *args, levels=levels, contour_kwargs=contour_kwargs, weights=weights,
        **_kwargs
    )
    X, Y, H = _calculate_density_with_kde_for_contour(
        *args, kde=kde, kde_kwargs=kde_kwargs, smooth=kwargs.get("smooth", None),
        weights=weights, gridsize=kwargs.get("gridsize", 300),
        cut=kwargs.get("cut", 0)
    )
    V = _get_contour_levels(H, levels)
    if kwargs.get("fill_contours", False) and not no_fill_contours:
        contourf_kwargs = kwargs.get("contourf_kwargs", {})
        if "colors" not in contourf_kwargs:
            contourf_kwargs["colors"] = kwargs.get("color")
        ax.contourf(X, Y, H, V, **contourf_kwargs)
    if kwargs.get("plot_contours", True):
        _contour_kwargs = kwargs.get("contour_kwargs", {})
        if "colors" not in _contour_kwargs:
            _contour_kwargs["colors"] = kwargs.get("color")
        ax.contour(X, Y, H, V, **_contour_kwargs)


def corner(
    samples, parameters, bins=20, *,
    # Original corner parameters
    range=None, axes_scale="linear", weights=None, color='k',
    hist_bin_factor=1, smooth=None, smooth1d=None, labels=None,
    label_kwargs=None, titles=None, show_titles=False,
    title_quantiles=None, title_fmt=".2f", title_kwargs=None,
    truths=None, truth_color="#4682b4", scale_hist=False,
    quantiles=None, verbose=False, fig=None, max_n_ticks=5,
    top_ticks=False, use_math_text=False, reverse=False,
    labelpad=0.0, hist_kwargs={},
    # Arviz parameters
    group="posterior", var_names=None, filter_vars=None,
    coords=None, divergences=False, divergences_kwargs=None,
    labeller=None,
    # New parameters
    kde=None, kde_kwargs={}, kde_2d=None, kde_2d_kwargs={},
    N=100, **hist2d_kwargs,
):
    """Wrapper for corner.corner which adds additional functionality
    to plot custom KDEs along the leading diagonal and custom 2D
    KDEs in the 2D panels
    """
    from corner import core
    core.hist2d = hist2d
    if kde is not None:
        hist_kwargs["linewidth"] = 0.
    if kde_2d is not None:
        linewidths = [1.]
        hist2d_kwargs = hist2d_kwargs.copy()
        if hist2d_kwargs.get("plot_contours", False):
            if "contour_kwargs" not in hist2d_kwargs.keys():
                hist2d_kwargs["contour_kwargs"] = {}
            linewidths = hist2d_kwargs["contour_kwargs"].get("linewidths", None)
            hist2d_kwargs["contour_kwargs"]["linewidths"] = 0.
        plot_density = hist2d_kwargs.get("plot_density", True)
        fill_contours = hist2d_kwargs.get("fill_contours", False)
        plot_contours = hist2d_kwargs.get("plot_contours", True)
        if plot_density:
            hist2d_kwargs["plot_density"] = False
        if fill_contours:
            hist2d_kwargs["fill_contours"] = False
        hist2d_kwargs["plot_contours"] = False

    fig = core.corner_impl(
        samples, range=range, axes_scale=axes_scale, weights=weights,
        color=color, hist_bin_factor=hist_bin_factor, smooth=smooth,
        smooth1d=smooth1d, labels=labels, label_kwargs=label_kwargs,
        titles=titles, show_titles=show_titles, title_quantiles=title_quantiles,
        title_fmt=title_fmt, title_kwargs=title_kwargs, truths=truths,
        truth_color=truth_color, scale_hist=scale_hist,
        quantiles=quantiles, verbose=verbose, fig=fig,
        max_n_ticks=max_n_ticks, top_ticks=top_ticks,
        use_math_text=use_math_text, reverse=reverse,
        labelpad=labelpad, hist_kwargs=hist_kwargs, bins=bins,
        # Arviz parameters
        group=group, var_names=var_names, filter_vars=filter_vars,
        coords=coords, divergences=divergences,
        divergences_kwargs=divergences_kwargs, labeller=labeller,
        **hist2d_kwargs
    )
    if kde is None and kde_2d is None:
        return fig
    axs = np.array(fig.get_axes(), dtype=object).reshape(
        len(parameters), len(parameters)
    )
    if kde is not None:
        for num, param in enumerate(parameters):
            if param in kde_kwargs.keys():
                _kwargs = kde_kwargs[param]
            else:
                _kwargs = {}
            for key, item in kde_kwargs.items():
                if key not in parameters:
                    _kwargs[key] = item
            _kde = kde(samples[:,num], weights=weights, **_kwargs)
            xs = np.linspace(np.min(samples[:,num]), np.max(samples[:,num]), N)
            axs[num, num].plot(
                xs, _kde(xs), color=color
            )
    if kde_2d is not None:
        _hist2d_kwargs = hist2d_kwargs.copy()
        _contour_kwargs = hist2d_kwargs.pop("contour_kwargs", {})
        _contour_kwargs["linewidths"] = linewidths
        _hist2d_kwargs.update(
            {
                "plot_contours": plot_contours,
                "plot_density": plot_density,
                "fill_contours": fill_contours,
                "levels": hist2d_kwargs.pop("levels"),
                "contour_kwargs": _contour_kwargs
            }
        )
        for i, x in enumerate(parameters):
            for j, y in enumerate(parameters):
                if j >= i:
                    continue
                _kde_2d_kwargs = {}
                _xkwargs = kde_2d_kwargs.get(x, kde_2d_kwargs)
                if "low" in _xkwargs.keys():
                    _xkwargs["ylow"] = _xkwargs.pop("low")
                if "high" in _xkwargs.keys():
                     _xkwargs["yhigh"] = _xkwargs.pop("high")
                _kde_2d_kwargs.update(_xkwargs)
                _ykwargs = kde_2d_kwargs.get(y, kde_2d_kwargs)
                if "low" in _ykwargs.keys():
                    _ykwargs["xlow"] = _ykwargs.pop("low")
                if "high" in _ykwargs.keys():
                     _ykwargs["xhigh"] = _ykwargs.pop("high")
                _kde_2d_kwargs.update(_ykwargs)
                for key, item in kde_2d_kwargs.items():
                    if key not in parameters:
                        _kde_2d_kwargs[key] = item
                hist2d(
                    samples[:,j], samples[:,i],
                    ax=axs[i, j], color=color,
                    kde=kde_2d, kde_kwargs=_kde_2d_kwargs,
                    bins=bins, **_hist2d_kwargs
                )
    return fig
