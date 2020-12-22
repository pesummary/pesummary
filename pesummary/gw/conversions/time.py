# Copyright (C) 2018 Charlie Hoy <charlie.hoy@ligo.org>
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

import numpy as np
from pesummary.utils.decorators import array_input

try:
    from lalsimulation import DetectorPrefixToLALDetector
    from lal import C_SI
    from astropy.time import Time
except ImportError:
    pass


@array_input
def time_in_each_ifo(detector, ra, dec, time_gps):
    """Return the event time in a given detector, given samples for ra, dec,
    time
    """
    gmst = Time(time_gps, format='gps', location=(0, 0))
    corrected_ra = gmst.sidereal_time('mean').rad - ra

    i = np.cos(dec) * np.cos(corrected_ra)
    j = np.cos(dec) * -1 * np.sin(corrected_ra)
    k = np.sin(dec)
    n = np.array([i, j, k])

    dx = [0, 0, 0] - DetectorPrefixToLALDetector(detector).location
    dt = dx.dot(n) / C_SI
    return time_gps + dt
