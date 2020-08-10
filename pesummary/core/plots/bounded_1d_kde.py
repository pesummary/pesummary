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
from scipy.ndimage.filters import gaussian_filter1d
from pesummary.utils.decorators import deprecation
from pesummary.utils.utils import logger


def transform_logit(x, a=0., b=1.):
    """
    """
    return np.log((x - a) / (b - x))


def inverse_transform_logit(y, a=0., b=1.):
    """
    """
    return (a + b * np.exp(y)) / (1 + np.exp(y))


def dydx_logit(x, a=0., b=1.):
    """
    """
    return (-a + b) / ((a - x) * (-b + x))


def bounded_1d_kde(
    pts, method="Reflection", xlow=None, xhigh=None, *args, **kwargs
):
    """Return a bounded 1d KDE

    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    method: str, optional
        Method you wish to use to handle the boundaries
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    """
    try:
        return globals()["{}BoundedKDE".format(method)](
            pts, xlow=xlow, xhigh=xhigh, *args, **kwargs
        )
    except KeyError:

        raise ValueError("Unknown method: {}".format(method))


class BoundedKDE(kde):
    """Base class to handle the BoundedKDE

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
        if pts.ndim != 1:
            raise TypeError("Bounded_1d_kde can only be one-dimensional")
        super(BoundedKDE, self).__init__(pts.T, *args, **kwargs)
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


class TransformBoundedKDE(BoundedKDE):
    """Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain. The bounds are treated as reflections

    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    """
    allowed = ["logit"]

    def __init__(
        self, pts, xlow=None, xhigh=None, transform="logit", inv_transform=None,
        dydx=None, alpha=1.5, N=100, smooth=3, *args, **kwargs
    ):
        self.inv_transform = inv_transform
        self.dydx = dydx
        self.transform = transform
        _args = np.hstack(np.argwhere((pts > xlow) & (pts < xhigh)))
        pts = pts[_args]
        transformed_pts = self.transform(pts, xlow, xhigh)
        super(TransformBoundedKDE, self).__init__(
            transformed_pts, xlow=xlow, xhigh=xhigh, *args, **kwargs
        )
        self.alpha = alpha
        self.N = N
        self.smooth = smooth

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, str) and transform not in self.allowed:
            raise ValueError(
                "Please provide either a transform function or pick an "
                "allowed transform from the list: {}".format(
                    ", ".join(self.allowed)
                )
            )
        elif isinstance(transform, str):
            self.inv_transform = globals()["inverse_transform_{}".format(transform)]
            self.dydx = globals()["dydx_{}".format(transform)]
            transform = globals()["transform_{}".format(transform)]
        if not isinstance(transform, str):
            if any(param is None for param in [self.inv_transform, self.dydx]):
                raise ValueError(
                    "Please provide an inverse transformation and the "
                    "derivative of the transform"
                )
        self._transform = transform

    def __call__(self, pts):
        _args = np.argwhere((pts > self.xlow) & (pts < self.xhigh))
        if len(_args) != len(np.atleast_1d(pts)):
            logger.info(
                "Removing {} samples as they are outside of the allowed "
                "domain".format(len(np.atleast_1d(pts)) - len(_args))
            )
        if not len(_args):
            return np.zeros_like(pts)
        pts = np.hstack(pts[_args])
        pts = self.transform(np.atleast_1d(pts), self.xlow, self.xhigh)
        ymin = self.alpha * np.min(pts)
        ymax = self.alpha * np.max(pts)
        y = np.linspace(ymin, ymax, self.N)
        x = self.inv_transform(y, self.xlow, self.xhigh)
        Y = self.evaluate(y) * np.abs(self.dydx(x, self.xlow, self.xhigh))
        ysmoothed = gaussian_filter1d(Y, sigma=self.smooth)
        return x, ysmoothed


class ReflectionBoundedKDE(BoundedKDE):
    """Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain. The bounds are treated as reflections

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
        super(ReflectionBoundedKDE, self).__init__(
            pts, xlow=xlow, xhigh=xhigh, *args, **kwargs
        )

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points
        """
        x = pts.T
        pdf = super(ReflectionBoundedKDE, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(ReflectionBoundedKDE, self).evaluate(2 * self.xlow - x)
        if self.xhigh is not None:
            pdf += super(ReflectionBoundedKDE, self).evaluate(2 * self.xhigh - x)
        return pdf

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts > self.xhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results


class Bounded_1d_kde(ReflectionBoundedKDE):
    @deprecation(
        "The Bounded_1d_kde class has changed its name to ReflectionBoundedKDE. "
        "Bounded_1d_kde may not be supported in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(Bounded_1d_kde, self).__init__(*args, **kwargs)
