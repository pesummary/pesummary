# Licensed under an MIT style license -- see LICENSE.md

import copy
import numpy as np
from scipy.stats import gaussian_kde as kde
from scipy.ndimage.filters import gaussian_filter1d
from pesummary.utils.decorators import deprecation
from pesummary.utils.utils import logger

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Michael Puerrer <michael.puerrer@ligo.org>"
]


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
        return _kdes["{}BoundedKDE".format(method)](
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
    domain. The bounds are handled by transforming to a new parameter
    space which is "unbounded" and then generating a KDE (including a Jacobian)
    in bounded space

    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    transform: str/func, optional
        The transform you wish to use. Default logit
    inv_transform: func, optional
        Inverse function of transform
    dydx: func, optional
        Derivateive of transform
    N: int, optional
        Number of points to use generating the KDE
    smooth: float, optional
        level of smoothing you wish to apply. Default 3
    apply_smoothing: Bool, optional
        Whether or not to apply smoothing. Default False
    """
    allowed = ["logit"]

    def __init__(
        self, pts, xlow=None, xhigh=None, transform="logit", inv_transform=None,
        dydx=None, alpha=1.5, N=100, smooth=3, apply_smoothing=False,
        weights=None, same_input=True, *args, **kwargs
    ):
        import pandas

        self.inv_transform = inv_transform
        self.dydx = dydx
        self.transform = transform
        self.same_input = same_input
        if isinstance(pts, pandas.core.series.Series):
            pts = np.array(pts)
        _args = np.hstack(np.argwhere((pts > xlow) & (pts < xhigh)))
        pts = pts[_args]
        if weights is not None:
            if isinstance(weights, pandas.core.series.Series):
                weights = np.array(weights)
            weights = weights[_args]
        transformed_pts = self.transform(pts, xlow, xhigh)
        super(TransformBoundedKDE, self).__init__(
            transformed_pts, xlow=xlow, xhigh=xhigh, *args, **kwargs
        )
        self.alpha = alpha
        self.N = N
        self.smooth = smooth
        self.apply_smoothing = apply_smoothing

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
            self.inv_transform = _default_methods[
                "inverse_transform_{}".format(transform)
            ]
            self.dydx = _default_methods["dydx_{}".format(transform)]
            transform = _default_methods["transform_{}".format(transform)]
        if not isinstance(transform, str):
            if any(param is None for param in [self.inv_transform, self.dydx]):
                raise ValueError(
                    "Please provide an inverse transformation and the "
                    "derivative of the transform"
                )
        self._transform = transform

    def __call__(self, pts):
        _original = copy.deepcopy(pts)
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
        delta = np.max(pts) - np.min(pts)
        ymin = np.min(pts) - ((self.alpha - 1.) / 2) * delta
        ymax = np.max(pts) + ((self.alpha - 1.) / 2) * delta
        y = np.linspace(ymin, ymax, self.N)
        x = self.inv_transform(y, self.xlow, self.xhigh)
        Y = self.evaluate(y) * np.abs(self.dydx(x, self.xlow, self.xhigh))
        if self.apply_smoothing:
            Y = gaussian_filter1d(Y, sigma=self.smooth)
        if self.same_input:
            from scipy.interpolate import interp1d

            f = interp1d(x, Y)
            _args = np.argwhere(
                (_original > np.amin(x)) & (_original < np.amax(x))
            )
            _Y = f(_original[_args])
            Y = np.zeros(len(_original))
            Y[_args] = _Y
            return Y
        return x, Y


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


_kdes = {
    "TransformBoundedKDE": TransformBoundedKDE,
    "ReflectionBoundedKDE": ReflectionBoundedKDE,
    "Bounded_1d_kde": Bounded_1d_kde
}

_default_methods = {
    "transform_logit": transform_logit,
    "inverse_transform_logit": inverse_transform_logit,
    "dydx_logit": dydx_logit
}
