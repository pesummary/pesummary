# Copyright (C) 2018  Charlie Hoy     <charlie.hoy@ligo.org>
#                     Michael Puerrer <michael.puerrer@ligo.org>
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
from scipy.stats import gaussian_kde as kde


class Bounded_1d_kde(kde):
    """Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain

    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    """
    def __init__(self, pts, xlow=None, xhigh=None, *args, **kwargs):
        pts = np.atleast_1d(pts)
        if pts.ndim == 2:
            raise TypeError("Bounded_1d_kde can only be one-dimensional")
        super(Bounded_1d_kde, self).__init__(pts.T, *args, **kwargs)
        self._xlow = xlow
        self._xhigh = xhigh

    @property
    def xlow(self):
        """The lower bound of the x domain
        """
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain
        """
        return self._xhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points
        """
        pts = np.atleast_1d(pts)
        assert pts.ndim == 2, 'points must be one-dimensional'

        x = pts.T
        pdf = super(Bounded_1d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_1d_kde, self).evaluate(2 * self.xlow - x)

        if self.xhigh is not None:
            pdf += super(Bounded_1d_kde, self).evaluate(2 * self.xhigh - x)

        return pdf

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
