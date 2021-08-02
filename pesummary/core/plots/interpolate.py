# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from scipy.interpolate import interp1d

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Bounded_interp1d(object):
    """Return a bounded 1-D interpolant. Interpolating outside of the bounded
    domain simply returns 0.

    Parameters
    ----------
    x: np.array
        A 1-D array of real values.
    y: np.array
        A N-D array of real values. The length of y along the interpolation axis
        must be equal to the length of x.
    xlow: float, optional
        the lower bound of the bounded domain
    xhigh: float, optional
        the upper bound of the bounded domain
    **kwargs: dict, optional
        all kwargs passed to scipy.interpolate.interp1d
    """
    def __init__(self, x, y, xlow=-np.inf, xhigh=np.inf, **kwargs):
        if xlow > np.min(x):
            self._xlow = xlow
        else:
            self._xlow = np.min(x)
        if xhigh < np.max(x):
            self._xhigh = xhigh
        else:
            self._xhigh = np.max(x)
        self._complex = np.iscomplexobj(y)
        self._interp_real = interp1d(x, np.real(y), **kwargs)
        if self._complex:
            self._interp_imaginary = interp1d(x, np.imag(y), **kwargs)

    @property
    def xlow(self):
        return self._xlow

    @property
    def xhigh(self):
        return self._xhigh

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        result = np.zeros_like(pts)
        within_bounds = np.ones_like(pts, dtype='bool')
        within_bounds[(pts < self.xlow) | (pts > self.xhigh)] = False
        result[within_bounds] = self._interp_real(pts[within_bounds])
        if self._complex:
            result[within_bounds] += 1j * self._interp_imaginary(
                pts[within_bounds]
            )
        return result
