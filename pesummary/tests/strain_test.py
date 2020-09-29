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

from pesummary.gw.file.strain import StrainData
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.figure


class TestStrainData(object):
    """Class to test the pesummary.gw.file.strain.StrainData class
    """
    def test_fetch_open_data(self):
        """Test that the gwpy methods are still the same
        """
        args = ["L1", 1126259446, 1126259478]
        pesummary_data = StrainData.fetch_open_data(*args)
        gwpy_data = TimeSeries.fetch_open_data(*args)
        np.testing.assert_almost_equal(pesummary_data.value, gwpy_data.value)
        np.testing.assert_almost_equal(
            pesummary_data.times.value, gwpy_data.times.value
        )
        assert isinstance(pesummary_data.gwpy, TimeSeries)
        np.testing.assert_almost_equal(
            pesummary_data.gwpy.value, gwpy_data.value
        )
        np.testing.assert_almost_equal(
            pesummary_data.gwpy.times.value, gwpy_data.times.value
        )
        assert pesummary_data.IFO == "L1"
        assert list(pesummary_data.strain_dict.keys()) == ["L1"]
        np.testing.assert_almost_equal(
            pesummary_data.strain_dict["L1"].value, gwpy_data.value
        )
        np.testing.assert_almost_equal(
            pesummary_data.strain_dict["L1"].times.value, gwpy_data.times.value
        )

    def test_plots(self):
        """Test that the plotting methods work as expected
        """
        args = ["L1", 1126259446, 1126259478]
        pesummary_data = StrainData.fetch_open_data(*args)
        fig = pesummary_data.plot(type="td")
        assert isinstance(fig, matplotlib.figure.Figure)
        fig = pesummary_data.plot(type="fd")
        assert isinstance(fig, matplotlib.figure.Figure)
        fig = pesummary_data.plot(1126259446 + 20., type="omegascan")
        assert isinstance(fig, matplotlib.figure.Figure)
        fig = pesummary_data.plot(type="spectrogram")
        assert isinstance(fig, matplotlib.figure.Figure)
