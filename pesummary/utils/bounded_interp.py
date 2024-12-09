# Licensed under an MIT style license -- see LICENSE.md

from scipy.interpolate import RectBivariateSpline as _RectBivariateSpline
import numpy as np

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


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
