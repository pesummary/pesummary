# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde, bounded_1d_kde
from pesummary.core.plots.bounded_2d_kde import Bounded_2d_kde
from scipy.stats import gaussian_kde
import numpy as np
import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestBounded_kde(object):
    """Test the Bounded_1d_kde function
    """
    def test_bounded_1d_kde(self):
        samples = np.array(np.random.uniform(10, 5, 1000))
        x_low = 9.5
        x_high = 10.5
        scipy = gaussian_kde(samples)
        bounded = Bounded_1d_kde(samples, xlow=x_low, xhigh=x_high)
        assert scipy(9.45) != 0.
        assert bounded(9.45) == 0.
        assert scipy(10.55) != 0.
        assert bounded(10.55) == 0.
        bounded = bounded_1d_kde(samples, xlow=x_low, xhigh=x_high, method="Transform")
        assert bounded(10.55) == 0.
        assert bounded(9.45) == 0

    def test_bounded_2d_kde(self):
        samples = np.array([
            np.random.uniform(10, 5, 1000),
            np.random.uniform(5, 2, 1000)
        ])
        x_low = 9.5
        x_high = 10.5
        y_low = 4.5
        y_high = 5.5
        scipy = gaussian_kde(samples)
        bounded = Bounded_2d_kde(
            samples, xlow=x_low, xhigh=x_high, ylow=y_low, yhigh=y_high
        )
        assert scipy([9.45, 4.45]) != 0.
        assert bounded([9.45, 4.45]) == 0.
        assert scipy([9.45, 5.55]) != 0.
        assert bounded([9.45, 5.55]) == 0.

        assert scipy([10.55, 4.45]) != 0.
        assert bounded([10.55, 4.45]) == 0.
        assert scipy([10.55, 5.55]) != 0.
        assert bounded([10.55, 5.55]) == 0.

        with pytest.raises(AssertionError):
            np.testing.assert_almost_equal(scipy([[9.45, 10.55], [5., 5.]]),  [0., 0.])
        np.testing.assert_almost_equal(bounded([[9.45, 10.55], [5., 5.]]), [0., 0.])
