# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil
import numpy as np
from .base import make_injection_file, testing_dir
from pesummary.gw.file.injection import GWInjection

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestInjection(object):
    """Class to test the Injection class for both the core and gw package
    """
    def setup(self):
        """Setup the TestInjection class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def check(self, extension):
        """
        """
        if extension == "xml":
            ff = testing_dir + "/main_injection.xml"
            data = {
                'dec': [1.949725], 'geocent_time': [1186741861],
                'spin_2x': [0.0], 'spin_2y': [0.0], 'spin_2z': [0.0],
                'luminosity_distance': [139.7643], 'ra': [-1.261573],
                'spin_1y': [0.0], 'spin_1x': [0.0], 'spin_1z': [0.0],
                'psi': [1.75], 'phase': [0.0], 'iota': [1.0471976],
                'mass_1': [53.333332], 'mass_2': [26.666668],
                'symmetric_mass_ratio': [0.22222222], 'chirp_mass': [32.446098],
                'phase': [0.0]
            }
        else:
            ff, data = make_injection_file(
                extension=extension, return_filename=True,
                return_injection_dict=True
            )
        inj = GWInjection.read(ff, conversion=False)
        assert all(param in data.keys() for param in inj.samples_dict.keys())
        for param, value in inj.samples_dict.items():
            np.testing.assert_almost_equal(value, data[param], 5)

    def test_read_json(self):
        """Test that the Injection class can read in json formats
        """
        self.check("json")

    def test_read_dat(self):
        """Test that the Injection class can read in dat format
        """
        self.check("dat")

    def test_read_hdf5(self):
        """Test that the Injection class can read in hdf5 format
        """
        self.check("hdf5")
        self.check("h5")

    def test_read_xml(self):
        """Test that the Injection class can read in xml format
        """
        self.check("xml")
