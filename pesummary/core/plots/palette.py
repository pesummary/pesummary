# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.plots.seaborn import SEABORN
from ._seaborn_palette import SEABORN_PALETTES
from matplotlib import colormaps
from matplotlib import colors as mcolors
import numpy as np
from itertools import cycle

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class ColorList(list):
    """Class inherited from list to add extra functionality to convert
    a list of colors to different formats

    Methods
    -------
    as_rgb
        return a ColorList object with colors in rgb format
    as_hex
        return a ColorList object with colors in hex format
    """
    def as_rgb(self):
        return ColorList([mcolors.to_rgb(c) for c in self])

    def as_hex(self):
        return ColorList([mcolors.rgb2hex(rgb) for rgb in self])


AVAILABLE_PALETTES = colormaps() + list(SEABORN_PALETTES.keys())
# alternative for color palette from seaborn
if SEABORN:
    from seaborn import color_palette
else:
    def color_palette(palette, n_colors=1):
        # check to see if provided palette is in matplotlib
        if palette in colormaps():
            cmap = colormaps[palette]
            colors = ColorList(cmap(np.linspace(0, 1, n_colors)).tolist())
            return colors.as_rgb()
        # else check if provided in seaborn palettes
        elif palette in SEABORN_PALETTES.keys():
            palette = SEABORN_PALETTES[palette]
            palette_cycle = cycle(palette)
            colors = ColorList([next(palette_cycle) for _ in range(n_colors)])
            return colors.as_rgb()
        else:
            raise ValueError(
                "Unknown color palette: {}. Available palettes are: {}".format(
                    palette, ", ".join(AVAILABLE_PALETTES)
                )
            )
