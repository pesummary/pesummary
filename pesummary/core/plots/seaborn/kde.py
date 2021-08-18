# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import warnings
from scipy import stats
from seaborn.distributions import (
    _DistributionPlotter as SeabornDistributionPlotter, KDE as SeabornKDE,
)
from seaborn.utils import _normalize_kwargs, _check_argument

import pandas as pd

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>", "Seaborn authors"]


class KDE(SeabornKDE):
    """Extension of the `seaborn._statistics.KDE` to allow for custom
    kde_kernel

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn._statistics.KDE` class
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    **kwargs: dict
        all kwargs passed to the `seaborn._statistics.KDE` class
    """
    def __init__(
        self, *args, kde_kernel=stats.gaussian_kde, kde_kwargs={}, **kwargs
    ):
        super(KDE, self).__init__(*args, **kwargs)
        self._kde_kernel = kde_kernel
        self._kde_kwargs = kde_kwargs

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = self._kde_kwargs
        fit_kws["bw_method"] = self.bw_method
        if weights is not None:
            fit_kws["weights"] = weights

        kde = self._kde_kernel(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)
        return kde


class _DistributionPlotter(SeabornDistributionPlotter):
    """Extension of the `seaborn._statistics._DistributionPlotter` to allow for
    the custom KDE method to be used

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn._statistics._DistributionPlotter` class
    **kwargs: dict
        all kwargs passed to the `seaborn._statistics._DistributionPlotter`
        class
    """
    def __init__(self, *args, **kwargs):
        super(_DistributionPlotter, self).__init__(*args, **kwargs)

    def plot_univariate_density(
        self,
        multiple,
        common_norm,
        common_grid,
        fill,
        legend,
        estimate_kws,
        variance_atol,
        **plot_kws,
    ):

        import matplotlib as mpl
        # Handle conditional defaults
        if fill is None:
            fill = multiple in ("stack", "fill")

        # Preprocess the matplotlib keyword dictionaries
        if fill:
            artist = mpl.collections.PolyCollection
        else:
            artist = mpl.lines.Line2D
        plot_kws = _normalize_kwargs(plot_kws, artist)

        # Input checking
        _check_argument("multiple", ["layer", "stack", "fill"], multiple)

        # Always share the evaluation grid when stacking
        subsets = bool(set(self.variables) - {"x", "y"})
        if subsets and multiple in ("stack", "fill"):
            common_grid = True

        # Check if the data axis is log scaled
        log_scale = self._log_scaled(self.data_variable)

        # Do the computation
        densities = self._compute_univariate_density(
            self.data_variable,
            common_norm,
            common_grid,
            estimate_kws,
            log_scale,
            variance_atol,
        )

        # Note: raises when no hue and multiple != layer. A problem?
        densities, baselines = self._resolve_multiple(densities, multiple)

        # Control the interaction with autoscaling by defining sticky_edges
        # i.e. we don't want autoscale margins below the density curve
        sticky_density = (0, 1) if multiple == "fill" else (0, np.inf)

        if multiple == "fill":
            # Filled plots should not have any margins
            sticky_support = densities.index.min(), densities.index.max()
        else:
            sticky_support = []

        # Handle default visual attributes
        if "hue" not in self.variables:
            if self.ax is None:
                color = plot_kws.pop("color", None)
                default_color = "C0" if color is None else color
            else:
                if fill:
                    if self.var_types[self.data_variable] == "datetime":
                        # Avoid drawing empty fill_between on date axis
                        # https://github.com/matplotlib/matplotlib/issues/17586
                        scout = None
                        default_color = plot_kws.pop(
                            "color", plot_kws.pop("facecolor", None)
                        )
                        if default_color is None:
                            default_color = "C0"
                    else:
                        alpha_shade = plot_kws.pop("alpha_shade", 0.25)
                        scout = self.ax.fill_between([], [], **plot_kws)
                        default_color = tuple(scout.get_facecolor().squeeze())
                    plot_kws.pop("color", None)
                else:
                    plot_kws.pop("alpha_shade", 0.25)
                    scout, = self.ax.plot([], [], **plot_kws)
                    default_color = scout.get_color()
                if scout is not None:
                    scout.remove()

        plot_kws.pop("color", None)

        default_alpha = .25 if multiple == "layer" else .75
        alpha = plot_kws.pop("alpha", default_alpha)  # TODO make parameter?

        # Now iterate through the subsets and draw the densities
        # We go backwards so stacked densities read from top-to-bottom
        for sub_vars, _ in self.iter_data("hue", reverse=True):

            # Extract the support grid and density curve for this level
            key = tuple(sub_vars.items())
            try:
                density = densities[key]
            except KeyError:
                continue
            support = density.index
            fill_from = baselines[key]

            ax = self._get_axes(sub_vars)

            # Modify the matplotlib attributes from semantic mapping
            if "hue" in self.variables:
                color = self._hue_map(sub_vars["hue"])
            else:
                color = default_color

            artist_kws = self._artist_kws(
                plot_kws, fill, False, multiple, color, alpha
            )

            # Either plot a curve with observation values on the x axis
            if "x" in self.variables:

                if fill:
                    artist = ax.fill_between(
                        support, fill_from, density, **artist_kws
                    )
                else:
                    artist, = ax.plot(support, density, **artist_kws)

                artist.sticky_edges.x[:] = sticky_support
                artist.sticky_edges.y[:] = sticky_density

            # Or plot a curve with observation values on the y axis
            else:
                if fill:
                    artist = ax.fill_betweenx(
                        support, fill_from, density, **artist_kws
                    )
                else:
                    artist, = ax.plot(density, support, **artist_kws)

                artist.sticky_edges.x[:] = sticky_density
                artist.sticky_edges.y[:] = sticky_support

        # --- Finalize the plot ----

        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        default_x = default_y = ""
        if self.data_variable == "x":
            default_y = "Density"
        if self.data_variable == "y":
            default_x = "Density"
        self._add_axis_labels(ax, default_x, default_y)

        if "hue" in self.variables and legend:
            from functools import partial
            if fill:
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])

            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, fill, False, multiple, alpha, plot_kws, {},
            )

    def _compute_univariate_density(
        self,
        data_variable,
        common_norm,
        common_grid,
        estimate_kws,
        log_scale,
        variance_atol,
    ):

        # Initialize the estimator object
        estimator = KDE(**estimate_kws)

        all_data = self.plot_data.dropna()

        if set(self.variables) - {"x", "y"}:
            if common_grid:
                all_observations = self.comp_data.dropna()
                estimator.define_support(all_observations[data_variable])
        else:
            common_norm = False

        densities = {}

        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            # Extract the data points from this sub set and remove nulls
            sub_data = sub_data.dropna()
            observations = sub_data[data_variable]

            observation_variance = observations.var()
            if np.isclose(observation_variance, 0, atol=variance_atol) or np.isnan(observation_variance):
                msg = "Dataset has 0 variance; skipping density estimate."
                warnings.warn(msg, UserWarning)
                continue

            # Extract the weights for this subset of observations
            if "weights" in self.variables:
                weights = sub_data["weights"]
            else:
                weights = None

            # Estimate the density of observations at this level
            density, support = estimator(observations, weights=weights)

            if log_scale:
                support = np.power(10, support)

            # Apply a scaling factor so that the integral over all subsets is 1
            if common_norm:
                density *= len(sub_data) / len(all_data)

            # Store the density for this level
            key = tuple(sub_vars.items())
            densities[key] = pd.Series(density, index=support)

        return densities


def kdeplot(
    x=None,  # Allow positional x, because behavior will not change with reorg
    *,
    y=None,
    shade=None,  # Note "soft" deprecation, explained below
    vertical=False,  # Deprecated
    kernel=None,  # Deprecated
    bw=None,  # Deprecated
    gridsize=200,  # TODO maybe depend on uni/bivariate?
    cut=3, clip=None, legend=True, cumulative=False,
    shade_lowest=None,  # Deprecated, controlled with levels now
    cbar=False, cbar_ax=None, cbar_kws=None,
    ax=None,

    # New params
    weights=None,  # TODO note that weights is grouped with semantics
    hue=None, palette=None, hue_order=None, hue_norm=None,
    multiple="layer", common_norm=True, common_grid=False,
    levels=10, thresh=.05,
    bw_method="scott", bw_adjust=1, log_scale=None,
    color=None, fill=None, kde_kernel=stats.gaussian_kde, kde_kwargs={},
    variance_atol=1e-8,

    # Renamed params
    data=None, data2=None,

    **kwargs,
):

    if kde_kernel is None:
        kde_kernel = stats.gaussian_kde
    # Handle deprecation of `data2` as name for y variable
    if data2 is not None:

        y = data2

        # If `data2` is present, we need to check for the `data` kwarg being
        # used to pass a vector for `x`. We'll reassign the vectors and warn.
        # We need this check because just passing a vector to `data` is now
        # technically valid.

        x_passed_as_data = (
            x is None
            and data is not None
            and np.ndim(data) == 1
        )

        if x_passed_as_data:
            msg = "Use `x` and `y` rather than `data` `and `data2`"
            x = data
        else:
            msg = "The `data2` param is now named `y`; please update your code"

        warnings.warn(msg, FutureWarning)

    # Handle deprecation of `vertical`
    if vertical:
        msg = (
            "The `vertical` parameter is deprecated and will be removed in a "
            "future version. Assign the data to the `y` variable instead."
        )
        warnings.warn(msg, FutureWarning)
        x, y = y, x

    # Handle deprecation of `bw`
    if bw is not None:
        msg = (
            "The `bw` parameter is deprecated in favor of `bw_method` and "
            f"`bw_adjust`. Using {bw} for `bw_method`, but please "
            "see the docs for the new parameters and update your code."
        )
        warnings.warn(msg, FutureWarning)
        bw_method = bw

    # Handle deprecation of `kernel`
    if kernel is not None:
        msg = (
            "Support for alternate kernels has been removed. "
            "Using Gaussian kernel."
        )
        warnings.warn(msg, UserWarning)

    # Handle deprecation of shade_lowest
    if shade_lowest is not None:
        if shade_lowest:
            thresh = 0
        msg = (
            "`shade_lowest` is now deprecated in favor of `thresh`. "
            f"Setting `thresh={thresh}`, but please update your code."
        )
        warnings.warn(msg, UserWarning)

    # Handle `n_levels`
    # This was never in the formal API but it was processed, and appeared in an
    # example. We can treat as an alias for `levels` now and deprecate later.
    levels = kwargs.pop("n_levels", levels)

    # Handle "soft" deprecation of shade `shade` is not really the right
    # terminology here, but unlike some of the other deprecated parameters it
    # is probably very commonly used and much hard to remove. This is therefore
    # going to be a longer process where, first, `fill` will be introduced and
    # be used throughout the documentation. In 0.12, when kwarg-only
    # enforcement hits, we can remove the shade/shade_lowest out of the
    # function signature all together and pull them out of the kwargs. Then we
    # can actually fire a FutureWarning, and eventually remove.
    if shade is not None:
        fill = shade

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    p = _DistributionPlotter(
        data=data,
        variables=_DistributionPlotter.get_semantics(locals()),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    # Check for a specification that lacks x/y data and return early
    if not p.has_xy_data:
        return ax

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
        kde_kernel=kde_kernel,
        kde_kwargs=kde_kwargs
    )

    p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    if p.univariate:

        plot_kws = kwargs.copy()
        if color is not None:
            plot_kws["color"] = color

        p.plot_univariate_density(
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            fill=fill,
            legend=legend,
            estimate_kws=estimate_kws,
            variance_atol=variance_atol,
            **plot_kws,
        )

    else:

        p.plot_bivariate_density(
            common_norm=common_norm,
            fill=fill,
            levels=levels,
            thresh=thresh,
            legend=legend,
            color=color,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            estimate_kws=estimate_kws,
            **kwargs,
        )

    return ax
