# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.strain import StrainData
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.figure

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
