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

from pesummary.gw.file.psd import PSDDict, PSD
import numpy as np
import os
import shutil


class TestPSDDict(object):
    """Test that the PSDDict works as expected
    """
    def setup(self):
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


class TestPSD(object):
    """Test the PSD class
    """
    def setup(self):
        """Setup the testing class
        """
        self.obj = PSD([[10, 20], [10, 20]])
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove all files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_save_to_file(self):
        """Test the save to file method
        """
        self.obj.save_to_file(".outdir/test.dat")
        data = np.genfromtxt(".outdir/test.dat")
        np.testing.assert_almost_equal(data.T[0], [10, 10])
        np.testing.assert_almost_equal(data.T[1], [20, 20])

    def test_invalid_input(self):
        """Test that the appropiate error is raised if the input is wrong
        """
        import pytest

        with pytest.raises(IndexError):
            obj = PSD([10, 10])
