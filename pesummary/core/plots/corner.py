# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import corner

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
    kernel = kde(values, **kde_kwargs)
    xmin, xmax = np.min(x.flatten()), np.max(x.flatten())
    ymin, ymax = np.min(y.flatten()), np.max(y.flatten())
    X, Y = np.meshgrid(X, Y)
    pts = np.vstack([X.ravel(), Y.ravel()])
    z = kernel(pts)
    H = z.reshape(X.shape)
    if smooth is not None:
        if kde_kwargs.get("transform", None) is not None:
            from pesummary.utils.utils import logger
            logger.warn(
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
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

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
