# Licensed under an MIT style license -- see LICENSE.md

from seaborn.categorical import _ViolinPlotter
import matplotlib as mpl
from textwrap import dedent
import colorsys
import numpy as np
import math
from scipy import stats
import pandas as pd
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings

from seaborn import utils
from seaborn.utils import iqr, remove_na
from seaborn.algorithms import bootstrap
from seaborn.palettes import color_palette, husl_palette, light_palette, dark_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
from scipy.stats import gaussian_kde

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>", "Seaborn authors"]


class ViolinPlotter(_ViolinPlotter):
    """A class to extend the _ViolinPlotter class provided by Seaborn
    """
    def __init__(self, x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                 bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
                 width=.8, inner="box", split=False, dodge=True, orient=None,
                 linewidth=None, color=None, palette=None, saturation=.75,
                 ax=None, outer=None, inj=None, kde=gaussian_kde, kde_kwargs={},
                 weights=None, **kwargs):
        self.multi_color = False
        self.kde = kde
        self.kde_kwargs = kde_kwargs
        self.establish_variables(
            x, y, hue, data, orient, order, hue_order, weights=weights
        )
        self.establish_colors(color, palette, saturation)
        self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge
        self.inj = inj

        if inner is not None:
            if not any([inner.startswith("quart"),
                        inner.startswith("box"),
                        inner.startswith("stick"),
                        inner.startswith("point"),
                        inner.startswith("line")]):
                err = "Inner style '{}' not recognized".format(inner)
                raise ValueError(err)
        self.inner = inner

        if outer is not None:
            if isinstance(outer, dict):
                for i in outer.keys():
                    if not any([i.startswith("percent"),
                                i.startswith("inject")]):
                        err = "Outer style '{}' not recognized".format(outer)
                        raise ValueError(err)
            else:
                if not any([outer.startswith("percent"),
                            outer.startswith("injection")]):
                    err = "Outer style '{}' not recognized".format(outer)
                    raise ValueError(err)
        self.outer = outer

        if split and self.hue_names is not None and len(self.hue_names) != 2:
            msg = "There must be exactly two hue levels to use `split`.'"
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

    def establish_variables(self, x, y, hue, data, orient, order, hue_order,
                            weights=None, **kwargs):
        """Convert input specification into a common representation."""
        super(ViolinPlotter, self).establish_variables(
            x, y, hue, data, orient, order, hue_order, **kwargs
        )
        if weights is None:
            weights_data = []
            if isinstance(data, pd.DataFrame):
                colname = None
                if "weights" in data.columns:
                    colname = "weights"
                elif "weight" in data.columns:
                    colname = "weight"
                if colname is None:
                    colname = "weights"
                    data[colname] = np.ones(len(data))
                for _data in self.plot_data:
                    weights_data.append(data[colname][_data.index])
            else:
                for _data in self.plot_data:
                    weights_data.append(np.ones_like(_data))
        else:
            if hasattr(weights, "shape"):
                if len(data.shape) != len(weights.shape):
                    raise ValueError("weights shape must equal data shape")
                if len(weights.shape) == 1:
                    if np.isscalar(weights[0]):
                        weights_data = [weights]
                    else:
                        weights_data = list(weights)
                elif len(weights.shape) == 2:
                    nr, nc = weights.shape
                    if nr == 1 or nc == 1:
                        weights_data = [weights.ravel()]
                    else:
                        weights_data = [weights[:, i] for i in range(nc)]
                else:
                    error = "weights can have no more than 2 dimensions"
                    raise ValueError(error)
            elif np.isscalar(weights[0]):
                weights_data = [weights]
            else:
                weights_data = weights
            weights_data = [np.asarray(d, np.float) for d in weights_data]
        self.weights_data = weights_data

    def establish_colors(self, color, palette, saturation):
        """Get a list of colors for the main component of the plots."""
        if self.hue_names is None:
            n_colors = len(self.plot_data)

        else:
            n_colors = len(self.hue_names)
        if color is None and palette is None:
            # Determine whether the current palette will have enough values
            # If not, we'll default to the husl palette so each is distinct
            current_palette = utils.get_color_cycle()
            if n_colors <= len(current_palette):
                colors = color_palette(n_colors=n_colors)
            else:
                colors = husl_palette(n_colors, l=.7)
        elif palette is None:
            if self.hue_names:
                if self.default_palette == "light":
                    colors = light_palette(color, n_colors)
                elif self.default_palette == "dark":
                    colors = dark_palette(color, n_colors)
                else:
                    raise RuntimeError("No default palette specified")
            else:
                colors = [color] * n_colors
        else:
            colors = self.colors_from_palette(palette)
        rgb_colors = color_palette(colors)

        light_vals = [colorsys.rgb_to_hls(*c)[1] for c in rgb_colors]
        lum = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((lum, lum, lum))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray

    def colors_from_palette(self, palette):
        """grab the colors from the chosen palette"""
        if self.hue_names is None:
            n_colors = len(self.plot_data)
        else:
            n_colors = len(self.hue_names)

        if isinstance(palette, dict):
            keys = list(palette.keys())
            n_colors = len(self.plot_data)

            if "left" in keys and "right" in keys or all(
                    j in keys for j in self.hue_names):
                self.multi_color = True
                colors = [self._palette_or_color(palette[i], n_colors) for i in
                          keys]
                colors = [[colors[0][i], colors[1][i]] for i in range(n_colors)]
                colors = [y for x in colors for y in x]

            return colors
        else:
            colors = self._palette_or_color(palette, n_colors)
            return colors

    def _palette_or_color(self, palette_entry, n_colors):
        """Determine if the palette is a block color or a palette
        """
        if isinstance(palette_entry, list):
            while len(palette_entry) < n_colors:
                palette_entry += palette_entry

            return palette_entry

        elif "color:" in palette_entry:
            color = palette_entry.split("color:")[1]
            color = self._flatten_string(color)

            return [color] * n_colors

        else:
            return color_palette(palette_entry, n_colors)

    @staticmethod
    def _flatten_string(string):
        """Remove the trailing white space from a string"""
        return string.lstrip(" ")

    def estimate_densities(self, bw, cut, scale, scale_hue, gridsize):
        """Find the support and density for all of the data."""
        # Initialize data structures to keep track of plotting data
        if self.hue_names is None:
            support = []
            density = []
            counts = np.zeros(len(self.plot_data))
            max_density = np.zeros(len(self.plot_data))
        else:
            support = [[] for _ in self.plot_data]
            density = [[] for _ in self.plot_data]
            size = len(self.group_names), len(self.hue_names)
            counts = np.zeros(size)
            max_density = np.zeros(size)

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single level of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                # Strip missing datapoints
                kde_data = remove_na(group_data)

                # Handle special case of no data at this level
                if kde_data.size == 0:
                    support.append(np.array([]))
                    density.append(np.array([1.]))
                    counts[i] = 0
                    max_density[i] = 0
                    continue

                # Handle special case of a single unique datapoint
                elif np.unique(kde_data).size == 1:
                    support.append(np.unique(kde_data))
                    density.append(np.array([1.]))
                    counts[i] = 1
                    max_density[i] = 0
                    continue

                # Fit the KDE and get the used bandwidth size
                kde, bw_used = self.fit_kde(
                    kde_data, bw, weights=self.weights_data[i]
                )

                # Determine the support grid and get the density over it
                support_i = self.kde_support(kde_data, bw_used, cut, gridsize)
                density_i = kde(support_i)
                if np.array(density_i).ndim == 2:
                    support_i, density_i = density_i

                # Update the data structures with these results
                support.append(support_i)
                density.append(density_i)
                counts[i] = kde_data.size
                max_density[i] = density_i.max()

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                for j, hue_level in enumerate(self.hue_names):

                    # Handle special case of no data at this category level
                    if not group_data.size:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Select out the observations for this hue level
                    hue_mask = self.plot_hues[i] == hue_level

                    # Strip missing datapoints
                    kde_data = remove_na(group_data[hue_mask])

                    # Handle special case of no data at this level
                    if kde_data.size == 0:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Handle special case of a single unique datapoint
                    elif np.unique(kde_data).size == 1:
                        support[i].append(np.unique(kde_data))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 1
                        max_density[i, j] = 0
                        continue

                    # Fit the KDE and get the used bandwidth size
                    kde, bw_used = self.fit_kde(
                        kde_data, bw, weights=self.weights_data[i][hue_mask]
                    )
                    # Determine the support grid and get the density over it
                    support_ij = self.kde_support(kde_data, bw_used,
                                                  cut, gridsize)
                    density_ij = kde(support_ij)
                    if np.array(density_ij).ndim == 2:
                        support_ij, density_ij = density_ij

                    # Update the data structures with these results
                    support[i].append(support_ij)
                    density[i].append(density_ij)
                    counts[i, j] = kde_data.size
                    max_density[i, j] = density_ij.max()

        # Scale the height of the density curve.
        # For a violinplot the density is non-quantitative.
        # The objective here is to scale the curves relative to 1 so that
        # they can be multiplied by the width parameter during plotting.

        if scale == "area":
            self.scale_area(density, max_density, scale_hue)

        elif scale == "width":
            self.scale_width(density)

        elif scale == "count":
            self.scale_count(density, counts, scale_hue)

        else:
            raise ValueError("scale method '{}' not recognized".format(scale))

        # Set object attributes that will be used while plotting
        self.support = support
        self.density = density

    def draw_violins(self, ax):
        """Draw the violins onto `ax`."""
        fill_func = ax.fill_betweenx if self.orient == "v" else ax.fill_between
        checkpoint = 0
        for i, group_data in enumerate(self.plot_data):

            kws = dict(edgecolor=self.gray, linewidth=self.linewidth)

            # Option 1: we have a single level of grouping
            # --------------------------------------------
            if self.plot_hues is None:

                support, density = self.support[i], self.density[i]

                # Handle special case of no observations in this bin
                if support.size == 0:
                    continue

                # Handle special case of a single observation
                elif support.size == 1:
                    val = np.asscalar(support)
                    d = np.asscalar(density)
                    self.draw_single_observation(ax, i, val, d)
                    continue

                # Draw the violin for this group
                grid = np.ones(self.gridsize) * i
                fill_func(support,
                          grid - density * self.dwidth,
                          grid + density * self.dwidth,
                          facecolor=self.colors[i],
                          **kws)

                # Draw the interior representation of the data
                if self.inner is None:
                    continue

                # Get a nan-free vector of datapoints
                violin_data = remove_na(group_data)

                # Draw box and whisker information
                if self.inner.startswith("box"):
                    self.draw_box_lines(ax, violin_data, support, density, i)

                # Draw quartile lines
                elif self.inner.startswith("quart"):
                    self.draw_quartiles(ax, violin_data, support, density, i)

                # Draw stick observations
                elif self.inner.startswith("stick"):
                    self.draw_stick_lines(ax, violin_data, support, density, i)

                # Draw point observations
                elif self.inner.startswith("point"):
                    self.draw_points(ax, violin_data, i)

                # Draw single line
                elif self.inner.startswith("line"):
                    self.draw_single_line(ax, violin_data, i)

                if self.outer is None:
                    continue

                else:
                    self.draw_external_range(ax, violin_data, support, density, i)

                if self.inj is None:
                    continue

                else:
                    self.draw_injected_line(
                        ax, self.inj[i], violin_data, support, density, i
                    )

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):
                    support, density = self.support[i][j], self.density[i][j]
                    kws["facecolor"] = self.colors[j]
                    if self.multi_color:
                        kws["facecolor"] = self.colors[checkpoint]
                        checkpoint += 1

                    # Add legend data, but just for one set of violins
                    if not i and not self.multi_color:
                        self.add_legend_data(ax, self.colors[j], hue_level)

                    # Handle the special case where we have no observations
                    if support.size == 0:
                        continue

                    # Handle the special case where we have one observation
                    elif support.size == 1:
                        val = np.asscalar(support)
                        d = np.asscalar(density)
                        if self.split:
                            d = d / 2
                        at_group = i + offsets[j]
                        self.draw_single_observation(ax, at_group, val, d)
                        continue

                    # Option 2a: we are drawing a single split violin
                    # -----------------------------------------------

                    if self.split:

                        grid = np.ones(self.gridsize) * i
                        if j:
                            fill_func(support,
                                      grid,
                                      grid + density * self.dwidth,
                                      **kws)
                        else:
                            fill_func(support,
                                      grid - density * self.dwidth,
                                      grid,
                                      **kws)

                        # Draw the interior representation of the data
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])

                        # Draw quartile lines
                        if self.inner.startswith("quart"):
                            self.draw_quartiles(ax, violin_data,
                                                support, density, i,
                                                ["left", "right"][j])

                        # Draw stick observations
                        elif self.inner.startswith("stick"):
                            self.draw_stick_lines(ax, violin_data,
                                                  support, density, i,
                                                  ["left", "right"][j])

                        if self.outer is None:
                            continue

                        else:
                            self.draw_external_range(ax, violin_data,
                                                     support, density, i,
                                                     ["left", "right"][j],
                                                     weights=self.weights_data[i][hue_mask])

                        if self.inj is None:
                            continue

                        else:
                            self.draw_injected_line(
                                ax, self.inj[i], violin_data, support, density, i,
                                ["left", "right"][j]
                            )

                        # The box and point interior plots are drawn for
                        # all data at the group level, so we just do that once
                        if not j:
                            continue

                        # Get the whole vector for this group level
                        violin_data = remove_na(group_data)

                        # Draw box and whisker information
                        if self.inner.startswith("box"):
                            self.draw_box_lines(ax, violin_data,
                                                support, density, i)

                        # Draw point observations
                        elif self.inner.startswith("point"):
                            self.draw_points(ax, violin_data, i)

                        elif self.inner.startswith("line"):
                            self.draw_single_line(ax, violin_data, i)

                    # Option 2b: we are drawing full nested violins
                    # -----------------------------------------------

                    else:
                        grid = np.ones(self.gridsize) * (i + offsets[j])
                        fill_func(support,
                                  grid - density * self.dwidth,
                                  grid + density * self.dwidth,
                                  **kws)

                        # Draw the interior representation
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])

                        # Draw box and whisker information
                        if self.inner.startswith("box"):
                            self.draw_box_lines(ax, violin_data,
                                                support, density,
                                                i + offsets[j])

                        # Draw quartile lines
                        elif self.inner.startswith("quart"):
                            self.draw_quartiles(ax, violin_data,
                                                support, density,
                                                i + offsets[j])

                        # Draw stick observations
                        elif self.inner.startswith("stick"):
                            self.draw_stick_lines(ax, violin_data,
                                                  support, density,
                                                  i + offsets[j])

                        # Draw point observations
                        elif self.inner.startswith("point"):
                            self.draw_points(ax, violin_data, i + offsets[j])

    def fit_kde(self, x, bw, weights=None):
        """Estimate a KDE for a vector of data with flexible bandwidth."""
        kde = self.kde(x, bw_method=bw, weights=weights, **self.kde_kwargs)
        # Extract the numeric bandwidth from the KDE object
        bw_used = kde.factor

        # At this point, bw will be a numeric scale factor.
        # To get the actual bandwidth of the kernel, we multiple by the
        # unbiased standard deviation of the data, which we will use
        # elsewhere to compute the range of the support.
        bw_used = bw_used * x.std(ddof=1)

        return kde, bw_used

    def annotate_axes(self, ax):
        """Add descriptive labels to an Axes object."""
        if self.orient == "v":
            xlabel, ylabel = self.group_label, self.value_label
        else:
            xlabel, ylabel = self.value_label, self.group_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if self.orient == "v":
            ax.set_xticks(np.arange(len(self.plot_data)))
            ax.set_xticklabels(self.group_names)
        else:
            ax.set_yticks(np.arange(len(self.plot_data)))
            ax.set_yticklabels(self.group_names)

        if self.orient == "v":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, len(self.plot_data) - .5, auto=None)
        else:
            ax.yaxis.grid(False)
            ax.set_ylim(-.5, len(self.plot_data) - .5, auto=None)

        if self.hue_names is not None:
            if not self.multi_color:
                leg = ax.legend(loc="best")
                if self.hue_title is not None:
                    leg.set_title(self.hue_title)

                    # Set the title size a roundabout way to maintain
                    # compatibility with matplotlib 1.1
                    # TODO no longer needed
                    try:
                        title_size = mpl.rcParams["axes.labelsize"] * .85
                    except TypeError:  # labelsize is something like "large"
                        title_size = mpl.rcParams["axes.labelsize"]
                    prop = mpl.font_manager.FontProperties(size=title_size)
                    leg._legend_title_box._text.set_font_properties(prop)

    def draw_single_line(self, ax, data, center):
        """Draw a single line through the middle of the violin"""
        kws = dict(color=self.gray, edgecolor=self.gray)
        upper = np.max(data)
        lower = np.min(data)

        ax.plot([center, center], [lower, upper],
                linewidth=self.linewidth,
                color=self.gray)

    def _plot_single_line(self, ax, center, y, density, split=None, color=None):
        """Plot a single line on a violin plot"""
        width = self.dwidth * np.max(density) * 1.1
        color = self.gray if color is None else color

        if split == "left":
            ax.plot([center - width, center], [y, y],
                    linewidth=self.linewidth,
                    color=color)
        elif split == "right":
            ax.plot([center, center + width], [y, y],
                    linewidth=self.linewidth,
                    color=color)
        else:
            ax.plot([center - width, center + width], [y, y],
                    linewidth=self.linewidth,
                    color=color)

    def draw_external_range(self, ax, data, support, density,
                            center, split=None, weights=None):
        """Draw lines extending outside of the violin showing given range"""
        width = self.dwidth * np.max(density) * 1.1

        if isinstance(self.outer, dict):
            if "percentage" in list(self.outer.keys()):
                percent = float(self.outer["percentage"])
                if weights is None:
                    lower, upper = np.percentile(data, [100 - percent, percent])
                else:
                    from pesummary.utils.array import Array

                    _data = Array(data, weights=weights)
                    lower, upper = _data.confidence_interval(
                        [100 - percent, percent]
                    )
                h1 = np.min(data[data >= (upper)])
                h2 = np.max(data[data <= (lower)])

                self._plot_single_line(ax, center, h1, density, split=split)
                self._plot_single_line(ax, center, h2, density, split=split)
            if any("inject" in i for i in list(self.outer.keys())):
                key = [i for i in list(self.outer.keys()) if "inject" in i]
                if any("injection:" in i for i in list(self.outer.keys())):
                    split = key[0].split("injection:")[1]
                    split = self._flatten_string(split)

                injection = self.outer[key[0]]
                if isinstance(injection, list):
                    self._plot_single_line(
                        ax, center, injection[center], density, split=split,
                        color="r"
                    )
                else:
                    self._plot_single_line(
                        ax, center, injection, density, split=split, color="r",
                    )
        elif isinstance(self.outer, str):
            if "percent" in self.outer:
                percent = self.outer.split("percent:")[1]
                percent = float(self._flatten_string(percent))
                percent += (100 - percent) / 2.

                if weights is None:
                    lower, upper = np.percentile(data, [100 - percent, percent])
                else:
                    from pesummary.utils.array import Array

                    _data = Array(data, weights=weights)
                    lower, upper = _data.confidence_interval(
                        [100 - percent, percent]
                    )
                h1 = np.min(data[data >= (upper)])
                h2 = np.max(data[data <= (lower)])

                self._plot_single_line(ax, center, h1, density, split=split)
                self._plot_single_line(ax, center, h2, density, split=split)
            if "inject" in self.outer:
                if "injection:" in self.outer:
                    split = self.outer.split("injection:")[1]
                    split = self._flatten_string(split)

                injection = self.outer.split("injection:")[1]

                self._plot_single_line(
                    ax, center, injection, density, split=split, color="r"
                )

    def draw_injected_line(self, ax, inj, data, support, density,
                           center, split=None):
        """Mark the injected value on the violin"""
        width = self.dwidth * np.max(density) * 1.1
        if math.isnan(inj):
            return
        self._plot_single_line(ax, center, inj, density, split=split, color='r')

    def plot(self, ax):
        """Make the violin plot."""
        self.draw_violins(ax)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()


def violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
               bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
               width=.8, inner="box", split=False, dodge=True, orient=None,
               linewidth=None, color=None, palette=None, saturation=.75,
               ax=None, outer=None, inj=None, kde=gaussian_kde, kde_kwargs={},
               weights=None, **kwargs):

    plotter = ViolinPlotter(x, y, hue, data, order, hue_order,
                            bw, cut, scale, scale_hue, gridsize,
                            width, inner, split, dodge, orient, linewidth,
                            color, palette, saturation, outer=outer,
                            inj=inj, kde=kde, kde_kwargs=kde_kwargs, weights=weights)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax


def split_dataframe(
    left, right, labels, left_label="left", right_label="right",
    weights_left=None, weights_right=None
):
    """Generate a pandas DataFrame containing two sets of distributions -- one
    set for the left hand side of the violins, and one set for the right hand
    side of the violins

    Parameters
    ----------
    left: np.ndarray
        array of samples representing the left hand side of the violins
    right: np.ndarray
        array of samples representing the right hand side of the violins
    labels: np.array
        array containing the label associated with each violin
    """
    import pandas

    nviolin = len(left)
    if len(left) != len(right) != len(labels):
        raise ValueError("Please ensure that 'left' == 'right' == 'labels'")
    _left_label = np.array([[left_label] * len(sample) for sample in left])
    _right_label = np.array([[right_label] * len(sample) for sample in right])
    _labels = [
        [label] * (len(left[num]) + len(right[num])) for num, label in
        enumerate(labels)
    ]
    labels = [x for y in _labels for x in y]
    dataframe = [
        x for y in [[i, j] for i, j in zip(left, right)] for x in y
    ]
    dataframe = [x for y in dataframe for x in y]
    sides = [
        x for y in [[i, j] for i, j in zip(_left_label, _right_label)] for x in
        y
    ]
    sides = [x for y in sides for x in y]
    df = pandas.DataFrame(
        data={"data": dataframe, "side": sides, "label": labels}
    )
    if all(kwarg is None for kwarg in [weights_left, weights_right]):
        return df

    left_inds = df["side"][df["side"] == left_label].index
    right_inds = df["side"][df["side"] == right_label].index
    if weights_left is not None and weights_right is None:
        weights_right = [np.ones(len(right[num])) for num in range(nviolin)]
    elif weights_left is None and weights_right is not None:
        weights_left = [np.ones(len(left[num])) for num in range(nviolin)]
    if any(len(kwarg) != nviolin for kwarg in [weights_left, weights_right]):
        raise ValueError("help")

    weights = [
        x for y in [[i, j] for i, j in zip(weights_left, weights_right)] for x in y
    ]
    weights = [x for y in weights for x in y]
    df["weights"] = weights
    return df
