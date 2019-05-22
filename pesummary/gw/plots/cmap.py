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
