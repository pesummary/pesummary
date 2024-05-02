# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.psd import PSDDict, PSD
import numpy as np
import os
import shutil
import tempfile

tmpdir = tempfile.TemporaryDirectory(prefix=".", dir=".").name

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestPSDDict(object):
    """Test that the PSDDict works as expected
    """
    def setup_method(self):
        """Setup the testing class
        """
        self.psd_data = {
            "H1": [[0.00000e+00, 2.50000e-01],
                   [1.25000e-01, 2.50000e-01],
                   [2.50000e-01, 2.50000e-01]],
            "V1": [[0.00000e+00, 2.50000e-01],
                   [1.25000e-01, 2.50000e-01],
                   [2.50000e-01, 2.50000e-01]]
        }
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)

    def teardown_method(self):
        """Remove all files and directories created from this class
        """
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
        
    def test_initiate(self):
        """Test that the PSDDict class can be initalized correctly
        """
        psd_dict = PSDDict(self.psd_data.keys(), self.psd_data.values())
        assert sorted(list(psd_dict.detectors)) == ["H1", "V1"]
        assert isinstance(psd_dict["H1"], PSD)
        np.testing.assert_almost_equal(
            psd_dict["H1"].frequencies, [0, 0.125, 0.25]
        )
        np.testing.assert_almost_equal(
            psd_dict["V1"].strains, [0.25, 0.25, 0.25]
        )

        psd_dict = PSDDict(self.psd_data)
        assert sorted(list(psd_dict.detectors)) == ["H1", "V1"]
        assert isinstance(psd_dict["H1"], PSD)
        np.testing.assert_almost_equal(
            psd_dict["H1"].frequencies, [0, 0.125, 0.25]
        )
        np.testing.assert_almost_equal(
            psd_dict["V1"].strains, [0.25, 0.25, 0.25]
        )

    def test_plot(self):
        """Test the plotting function works correctly
        """
        import matplotlib

        psd_dict = PSDDict(self.psd_data)
        assert isinstance(psd_dict.plot(), matplotlib.figure.Figure)

    def test_read(self):
        """Test that the PSDDict class can be initialized correctly with the
        read classmethod
        """
        f = PSDDict(self.psd_data)
        for ifo, psd in f.items():
            psd.save_to_file("{}/{}_test.dat".format(tmpdir, ifo))
        g = PSDDict.read(
            files=[
                "{}/H1_test.dat".format(tmpdir), "{}/V1_test.dat".format(tmpdir)
            ], detectors=["H1", "V1"]
        )
        for ifo, psd in g.items():
            np.testing.assert_almost_equal(psd.frequencies, f[ifo].frequencies)
            np.testing.assert_almost_equal(psd.strains, f[ifo].strains)
        g = PSDDict.read(
            common_string="%s/{}_test.dat" % (tmpdir), detectors=["H1", "V1"]
        )
        for ifo, psd in g.items():
            np.testing.assert_almost_equal(psd.frequencies, f[ifo].frequencies)
            np.testing.assert_almost_equal(psd.strains, f[ifo].strains)

    def test_interpolate(self):
        """Test the interpolate method
        """
        f = PSDDict(self.psd_data)
        g = f.interpolate(
            f["H1"].low_frequency, f["H1"].delta_f / 2
        )
        for ifo, psd in f.items():
            np.testing.assert_almost_equal(g[ifo].delta_f, psd.delta_f / 2)
            np.testing.assert_almost_equal(g[ifo].low_frequency, psd.low_frequency)


class TestPSD(object):
    """Test the PSD class
    """
    def setup_method(self):
        """Setup the testing class
        """
        self.obj = PSD([[10, 20], [10.25, 20], [10.5, 20]])
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)

    def teardown_method(self):
        """Remove all files and directories created from this class
        """
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_to_file(self):
        """Test the save to file method
        """
        self.obj.save_to_file("{}/test.dat".format(tmpdir))
        data = np.genfromtxt("{}/test.dat".format(tmpdir))
        np.testing.assert_almost_equal(data.T[0], [10, 10.25, 10.5])
        np.testing.assert_almost_equal(data.T[1], [20, 20, 20])

    def test_invalid_input(self):
        """Test that the appropiate error is raised if the input is wrong
        """
        import pytest

        with pytest.raises(IndexError):
            obj = PSD([10, 10])

    def test_interpolate(self):
        """Test the interpolate method
        """
        g = self.obj.interpolate(
            self.obj.low_frequency,
            self.obj.delta_f / 2
        )
        np.testing.assert_almost_equal(g.delta_f, self.obj.delta_f / 2)
        np.testing.assert_almost_equal(g.low_frequency, self.obj.low_frequency)
