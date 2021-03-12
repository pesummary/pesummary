import numpy as np
from scipy.stats import gaussian_kde as kde

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Bounded_2d_kde(kde):
    """Class to generate a two-dimensional KDE for a probability distribution
    functon that exists on a bounded domain
    """
    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None,
                 transform=None, *args, **kwargs):
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'
        self._transform = transform
        if transform is not None:
            pts = transform(pts)
        super(Bounded_2d_kde, self).__init__(pts, *args, **kwargs)
        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

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

    @property
    def ylow(self):
        """The lower bound of the y domain
        """
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain
        """
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        _pts = np.atleast_2d(pts)
        assert _pts.ndim == 2, 'points must be two-dimensional'
        if _pts.shape[0] != 2 and _pts.shape[1] == 2:
            _pts = _pts.T

        x, y = _pts
        pdf = super(Bounded_2d_kde, self).evaluate(_pts)
        if self.xlow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2 * self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2 * self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2 * self.ylow - y])

        if self.yhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2 * self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate(
                    [2 * self.xlow - x, 2 * self.ylow - y])

            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate(
                    [2 * self.xlow - x, 2 * self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate(
                    [2 * self.xhigh - x, 2 * self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate(
                    [2 * self.xhigh - x, 2 * self.yhigh - y])
        return pdf

    def __call__(self, pts):
        _pts = np.atleast_2d(pts)
        if _pts.shape[0] != 2 and _pts.shape[1] == 2:
            _pts = _pts.T
        if self._transform is not None:
            _pts = self._transform(_pts)
        transpose = _pts.T
        out_of_bounds = np.zeros(transpose.shape[0], dtype='bool')
        if self.xlow is not None:
            out_of_bounds[transpose[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[transpose[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[transpose[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[transpose[:, 1] > self.yhigh] = True
        results = self.evaluate(_pts)
        results[out_of_bounds] = 0.
        return results
