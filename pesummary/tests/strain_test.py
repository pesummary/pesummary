# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.strain import StrainData
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.figure

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestStrainData(object):
    """Class to test the pesummary.gw.file.strain.StrainData class
    """
    def test_fetch_open_frame(self):
        """test that the StrainData.fetch_open_frame works as expected
        """
        import requests
        pesummary_data = StrainData.fetch_open_frame(
            "GW190412", IFO="L1", duration=32, sampling_rate=4096.,
            channel="L1:GWOSC-4KHZ_R1_STRAIN", format="hdf5"
        )
        N = len(pesummary_data)
        np.testing.assert_almost_equal(N * pesummary_data.dt.value, 32.)
        np.testing.assert_almost_equal(1. / pesummary_data.dt.value, 4096.)
        assert pesummary_data.IFO == "L1"
        _data = requests.get(
            "https://www.gw-openscience.org/eventapi/html/GWTC-2/GW190412/v3/"
            "L-L1_GWOSC_4KHZ_R1-1239082247-32.gwf"
        )
        with open("L-L1_GWOSC_4KHZ_R1-1239082247-32.gwf", "wb") as f:
            f.write(_data.content)
        data2 = TimeSeries.read(
            "L-L1_GWOSC_4KHZ_R1-1239082247-32.gwf",
            channel="L1:GWOSC-4KHZ_R1_STRAIN"
        )
        np.testing.assert_almost_equal(pesummary_data.value, data2.value)
        np.testing.assert_almost_equal(
            pesummary_data.times.value, data2.times.value
        )

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
