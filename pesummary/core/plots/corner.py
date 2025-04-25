# Licensed under an MIT style license -- see LICENSE.md

import copy
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, colorConverter

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _set_xlim(new_fig, ax, new_xlim):
    if new_fig:
        return ax.set_xlim(new_xlim)
    xlim = ax.get_xlim()
    return ax.set_xlim([min(xlim[0], new_xlim[0]), max(xlim[1], new_xlim[1])])


def _set_ylim(new_fig, ax, new_ylim):
    if new_fig:
        return ax.set_ylim(new_ylim)
    ylim = ax.get_ylim()
    return ax.set_ylim([min(ylim[0], new_ylim[0]), max(ylim[1], new_ylim[1])])


def hist2d(
    x, y, bins=20, range=None, weights=None, levels=None, smooth=None, ax=None,
    color=None, quiet=False, plot_datapoints=True, plot_density=True,
    plot_contours=True, no_fill_contours=False, fill_contours=False,
    contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
    pcolor_kwargs=None, new_fig=True, kde=None, kde_kwargs={},
    density_cmap=None, label=None, grid=True, **kwargs
):
    """Extension of the corner.hist2d function. Allows the user to specify the
    kde used when estimating the 2d probability density

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    y : array_like[nsamples,]
       The samples.
    quiet : bool
        If true, suppress warnings for small datasets.
    levels : array_like
        The contour levels to draw.
    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.
    plot_datapoints : bool
        Draw the individual data points.
    plot_density : bool
        Draw the density colormap.
    plot_contours : bool
        Draw the contours.
    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).
    fill_contours : bool
        Fill the contours.
    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.
    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.
    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.
    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.
    kde: func, optional
        KDE you wish to use to work out the 2d probability density
    kde_kwargs: dict, optional
        kwargs passed directly to kde
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if kde is None:
        kde = gaussian_kde

    if ax is None:
        raise ValueError("Please provide an axis to plot")
    # Set the default range based on the data range if not provided.
    if range is None:
        range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    if density_cmap is None:
        density_cmap = LinearSegmentedColormap.from_list(
            "density_cmap", [color, (1, 1, 1, 0)]
        )
    elif isinstance(density_cmap, str):
        from matplotlib import cm

        density_cmap = cm.get_cmap(density_cmap)

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        _, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            range=list(map(np.sort, range)),
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )

    values = np.vstack([x.flatten(), y.flatten()])
    kernel = kde(values, weights=weights, **kde_kwargs)
    xmin, xmax = np.min(x.flatten()), np.max(x.flatten())
    ymin, ymax = np.min(y.flatten()), np.max(y.flatten())
    X, Y = np.meshgrid(X, Y)
    pts = np.vstack([X.ravel(), Y.ravel()])
    z = kernel(pts)
    H = z.reshape(X.shape)
    if smooth is not None:
        if kde_kwargs.get("transform", None) is not None:
            from pesummary.utils.utils import logger
            logger.warning(
                "Smoothing PDF. This may give unwanted effects especially near "
                "any boundaries"
            )
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    if plot_contours or plot_density:
        pass

    if kde_kwargs is None:
        kde_kwargs = dict()
    if contour_kwargs is None:
        contour_kwargs = dict()

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        if weights is None:
            ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)
        else:
            _weights = copy.deepcopy(weights)
            _weights /= np.max(_weights)
            idxs = np.argsort(_weights)
            for num, (xx, yy) in enumerate(zip(x[idxs], y[idxs])):
                _data_kwargs = data_kwargs.copy()
                _data_kwargs["alpha"] *= _weights[num]
                ax.plot(xx, yy, "o", zorder=-1, rasterized=True, **_data_kwargs)

    # Plot the base fill to hide the densest data points.
    cs = ax.contour(
        X, Y, H, levels=(1 - np.array(levels)) * np.max(H), alpha=0.
    )
    contour_set = []
    for _contour in cs.collections:
        _contour_set = []
        for _path in _contour.get_paths():
            data = _path.vertices
            transpose = data.T
            for idx, axis in enumerate(["x", "y"]):
                limits = [
                    kde_kwargs.get("{}low".format(axis), -np.inf),
                    kde_kwargs.get("{}high".format(axis), np.inf)
                ]
                if kde_kwargs.get("transform", None) is None:
                    if limits[0] is not None:
                        transpose[idx][
                            np.argwhere(transpose[idx] < limits[0])
                        ] = limits[0]
                    if limits[1] is not None:
                        transpose[idx][
                            np.argwhere(transpose[idx] > limits[1])
                        ] = limits[1]
                else:
                    _transform = kde_kwargs["transform"](transpose)
            _contour_set.append(transpose)
        contour_set.append(_contour_set)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    if plot_density:
        if pcolor_kwargs is None:
            pcolor_kwargs = dict()
        pcolor_kwargs["shading"] = "auto"
        ax.pcolor(X, Y, np.max(H) - H, cmap=density_cmap, **pcolor_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        colors = contour_kwargs.pop("colors", color)
        linestyles = kwargs.pop("linestyles", "-")
        _list = [colors, linestyles]
        for num, (prop, default) in enumerate(zip(_list, ['k', '-'])):
            if prop is None:
                _list[num] = default * len(contour_set)
            elif isinstance(prop, str):
                _list[num] = [prop] * len(contour_set)
            elif len(prop) < len(contour_set):
                raise ValueError(
                    "Please provide a color/linestyle for each contour"
                )
        for idx, _contour in enumerate(contour_set):
            for _idx, _path in enumerate(_contour):
                if idx == 0 and _idx == 0:
                    _label = label
                else:
                    _label = None
                ax.plot(
                    *_path, color=_list[0][idx], label=_label,
                    linestyle=_list[1][idx]
                )

    _set_xlim(new_fig, ax, range[0])
    _set_ylim(new_fig, ax, range[1])


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
    from corner import corner
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

    fig = corner(
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
            _kde = kde(samples[:,num], **_kwargs)
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
                "levels": hist2d_kwargs.pop("levels")[::-1],
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
