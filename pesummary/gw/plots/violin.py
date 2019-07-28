# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
#                     Seaborn authors
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

from __future__ import division
from seaborn.categorical import _ViolinPlotter
import matplotlib as mpl
from textwrap import dedent
import colorsys
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings

from seaborn.external.six import string_types
from seaborn.external.six.moves import range

from seaborn import utils
from seaborn.utils import iqr, categorical_order, remove_na
from seaborn.algorithms import bootstrap
from seaborn.palettes import color_palette, husl_palette, light_palette, dark_palette
from seaborn.axisgrid import FacetGrid, _facet_docs


class ViolinPlotter(_ViolinPlotter):
    """A class to extend the _ViolinPlotter class provided by Seaborn
    """
    def __init__(self, x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                 bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
                 width=.8, inner="box", split=False, dodge=True, orient=None,
                 linewidth=None, color=None, palette=None, saturation=.75,
                 ax=None, outer=None, **kwargs):
        self.multi_color = False
        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)
        self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge

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
                                                     ["left", "right"][j])

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
                            center, split=None):
        """Draw lines extending outside of the violin showing given range"""
        width = self.dwidth * np.max(density) * 1.1

        if isinstance(self.outer, dict):
            if "percentage" in list(self.outer.keys()):
                percent = float(self.outer["percentage"])
                lower, upper = np.percentile(data, [100 - percent, percent])
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

                lower, upper = np.percentile(data, [100 - percent, percent])
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
               ax=None, outer=None, **kwargs):

    plotter = ViolinPlotter(x, y, hue, data, order, hue_order,
                            bw, cut, scale, scale_hue, gridsize,
                            width, inner, split, dodge, orient, linewidth,
                            color, palette, saturation, outer=outer)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax
