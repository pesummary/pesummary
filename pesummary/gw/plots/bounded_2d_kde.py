import numpy as np
from scipy.stats import gaussian_kde as kde


default_bounds = {"luminosity_distance": {"low": 0.},
                  "geocent_time": {"low": 0.},
                  "dec": {"low": -np.pi / 2, "high": np.pi / 2},
                  "ra": {"low": 0., "high": 2 * np.pi},
                  "a_1": {"low": 0., "high": 1.},
                  "a_2": {"low": 0., "high": 1.},
                  "phi_jl": {"low": 0., "high": 2 * np.pi},
                  "phase": {"low": 0., "high": 2 * np.pi},
                  "psi": {"low": 0., "high": 2 * np.pi},
                  "iota": {"low": 0., "high": np.pi},
                  "tilt_1": {"low": 0., "high": np.pi},
                  "tilt_2": {"low": 0., "high": np.pi},
                  "phi_12": {"low": 0., "high": 2 * np.pi},
                  "mass_2": {"low": 0., "high": "mass_1"},
                  "mass_1": {"low": 0},
                  "total_mass": {"low": 0.},
                  "chirp_mass": {"low": 0.},
                  "H1_time": {"low": 0.},
                  "L1_time": {"low": 0.},
                  "V1_time": {"low": 0.},
                  "E1_time": {"low": 0.},
                  "spin_1x": {"low": 0., "high": 1.},
                  "spin_1y": {"low": 0., "high": 1.},
                  "spin_1z": {"low": -1., "high": 1.},
                  "spin_2x": {"low": 0., "high": 1.},
                  "spin_2y": {"low": 0., "high": 1.},
                  "spin_2z": {"low": -1., "high": 1.},
                  "chi_p": {"low": 0., "high": 1.},
                  "chi_eff": {"low": -1., "high": 1.},
                  "mass_ratio": {"low": 0., "high": 1.},
                  "symmetric_mass_ratio": {"low": 0., "high": 0.25},
                  "phi_1": {"low": 0., "high": 2 * np.pi},
                  "phi_2": {"low": 0., "high": 2 * np.pi},
                  "cos_tilt_1": {"low": -1., "high": 1.},
                  "cos_tilt_2": {"low": -1., "high": 1.},
                  "redshift": {"low": 0.},
                  "comoving_distance": {"low": 0.},
                  "mass_1_source": {"low": 0.},
                  "mass_2_source": {"low": 0., "high": "mass_1_source"},
                  "chirp_mass_source": {"low": 0.},
                  "total_mass_source": {"low": 0.},
                  "cos_iota": {"low": -1., "high": 1.},
                  "theta_jn": {"low": 0., "high": np.pi},
                  "lambda_1": {"low": 0.},
                  "lambda_2": {"low": 0.},
                  "lambda_tilde": {"low": 0.},
                  "delta_lambda": {}}


class Bounded_2d_kde(kde):
    """Class to generate a two-dimensional KDE for a probability distribution
    functon that exists on a bounded domain
    """
    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None,
                 *args, **kwargs):
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'
        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)
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
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
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
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
