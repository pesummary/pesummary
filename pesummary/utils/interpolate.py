# Licensed under an MIT style license -- see LICENSE.md

from scipy.interpolate import interp1d, RectBivariateSpline as _RectBivariateSpline
import numpy as np

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


class BoundedInterp1d(object):
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


class RectBivariateSpline(_RectBivariateSpline):
    """A modified version of scipy.interpolant.RectBivariateSpline to add
    kwargs that were previously available in the deprecated
    scipy.interpolant.interp2d.

    Parameters
    ----------
    x: np.ndarray
        data points in the x direction
    y: np.ndarray
        data points in the y direction
    z: np.ndarray
        2d array of data points with shape (x.size, y.size)
    bounds_error: bool, optional
        If True, a ValueError is raised when interpolated values outside of
        the range [min(x), max(x)] and [min(y), max(y)] are requested. If False,
        fill_value is used when fill_value is not None. Default False
    fill_value: float, optional
        the value to use when interpolated values outside of the range
        [min(x), max(x)] and [min(y), max(y)] are requested. Default None
    """
    def __init__(self, x, y, z, bounds_error=False, fill_value=None, **kwargs):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        super(RectBivariateSpline, self).__init__(x=x, y=y, z=z, **kwargs)

    def __call__(self, x, y, *args, **kwargs):
        result = super(RectBivariateSpline, self).__call__(x=x, y=y, *args, **kwargs)
        if self.bounds_error or self.fill_value is not None:
            out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
            out_of_bounds_y = (y < self.y_min) | (y > self.y_max)
            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)
        if self.bounds_error and (any_out_of_bounds_x or any_out_of_bounds_y):
            raise ValueError(
                "x and y are out of range. Please ensure that x is in range "
                "[{}, {}] and y is in range [{}, {}]".format(
                    self.x_min, self.x_max, self.y_min, self.y_max
                )
            )
        if self.fill_value is not None:
            if any_out_of_bounds_x:
                result[out_of_bounds_x, :] = self.fill_value
            if any_out_of_bounds_y:
                result[:, out_of_bounds_y] = self.fill_value
        return result
