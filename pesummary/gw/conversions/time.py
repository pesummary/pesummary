# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from lalsimulation import DetectorPrefixToLALDetector
    from lal import C_SI
    from astropy.time import Time
except ImportError:
    pass


@array_input()
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
