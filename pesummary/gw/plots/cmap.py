# Copyright (C) 2014-2016  Leo Singer, Charlie Hoy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from matplotlib import cm
from matplotlib import colors
import numpy as np
import pesummary


def cylon():
    # Read in color map RGB data.
    path = pesummary.__file__[:-12]
    with open(path + "/gw/plots/cylon.csv") as f:
        data = np.loadtxt(f, delimiter=',')

    # Create color map.
    cmap = colors.LinearSegmentedColormap.from_list("cylon", data)
    # Assign in module.
    locals().update({"cylon": cmap})
    # Register with Matplotlib.
    cm.register_cmap(cmap=cmap)
    # Create inverse color map
    cmap_r = colors.LinearSegmentedColormap.from_list("cylon_r", data[::-1])
    locals().update({"cylon_r": cmap_r})
    cm.register_cmap(cmap=cmap_r)


def colormap_with_fixed_hue(color, N=10):
    """Create a linear colormap with fixed hue

    Parameters
    ----------
    color: tuple
        color that determines the hue
    N: int, optional
        number of colors used in the palette
    """
    import seaborn
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, hex2color

    color_hsv = rgb_to_hsv(hex2color(color))
    base = seaborn.color_palette("Blues", 10)
    base_hsv = np.array(list(map(rgb_to_hsv, base)))
    h, s, v = base_hsv.T

    h_fixed = np.ones_like(h) * color_hsv[0]
    color_array = np.array(list(map(
        hsv_to_rgb, np.vstack([h_fixed, s * color_hsv[1], v]).T)))
    return LinearSegmentedColormap.from_list("mycmap", color_array)
