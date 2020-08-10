# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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

from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde, bounded_1d_kde
from pesummary.gw.plots.bounded_2d_kde import Bounded_2d_kde
from scipy.stats import gaussian_kde
import numpy as np


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
            samples.T, xlow=x_low, xhigh=x_high, ylow=y_low, yhigh=y_high
        )
        assert scipy([9.45, 4.45]) != 0.
        assert bounded([9.45, 4.45]) == 0.
        assert scipy([9.45, 5.55]) != 0.
        assert bounded([9.45, 5.55]) == 0.

        assert scipy([10.55, 4.45]) != 0.
        assert bounded([10.55, 4.45]) == 0.
        assert scipy([10.55, 5.55]) != 0.
        assert bounded([10.55, 5.55]) == 0.
