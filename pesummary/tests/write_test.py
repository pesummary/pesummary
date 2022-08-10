# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil
import numpy as np
import pytest

from pesummary.io import write, read
import tempfile

tmpdir = tempfile.TemporaryDirectory(prefix=".", dir=".").name

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Base(object):
    """Base class containing useful functions
    """
    def write(self, file_format, filename, **kwargs):
        """Write the samples to file
        """
        self.parameters = ["a", "b"]
        self.samples = np.array([
            np.random.uniform(10, 5, 100),
            np.random.uniform(100, 2, 100)
        ]).T
        write(
            self.parameters, self.samples, file_format=file_format, filename=filename,
            outdir=tmpdir, **kwargs
        )
        return self.parameters, self.samples

    def check_samples(self, filename, parameters, samples, pesummary=False):
        """Check the saved posterior samples
        """
        f = read(filename)
        posterior_samples = f.samples_dict
        if pesummary:
            posterior_samples = posterior_samples["label"]
        for num, param in enumerate(parameters):
            np.testing.assert_almost_equal(
                samples[num], posterior_samples[param]
            )


class TestWrite(Base):
    """Class to test the pesummary.io.write method
    """
    def setup(self):
        """Setup the Write class
        """
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir) 

    def test_dat(self):
        """Test that the user can write to a dat file
        """
        parameters, samples = self.write("dat", "pesummary.dat")
        self.check_samples("{}/pesummary.dat".format(tmpdir), parameters, samples.T)

    def test_json(self):
        """Test that the user can write to a json file
        """
        parameters, samples = self.write("json", "pesummary.json")
        self.check_samples("{}/pesummary.json".format(tmpdir), parameters, samples.T)

    def test_hdf5(self):
        """Test that the user can write to a hdf5 file
        """
        parameters, samples = self.write("h5", "pesummary.h5")
        self.check_samples("{}/pesummary.h5".format(tmpdir), parameters, samples.T)

    def test_bilby(self):
        """Test that the user can write to a bilby file
        """
        parameters, samples = self.write("bilby", "bilby.json")
        self.check_samples("{}/bilby.json".format(tmpdir), parameters, samples.T)
        parameters, samples = self.write("bilby", "bilby.h5", extension="hdf5")
        self.check_samples("{}/bilby.h5".format(tmpdir), parameters, samples.T)

    def test_lalinference(self):
        """Test that the user can write to a lalinference file
        """
        parameters, samples = self.write("lalinference", "lalinference.hdf5")
        self.check_samples("{}/lalinference.hdf5".format(tmpdir), parameters, samples.T)

    def test_sql(self):
        """Test that the user can write to an sql database
        """
        parameters, samples = self.write("sql", "sql.db")
        self.check_samples("{}/sql.db".format(tmpdir), parameters, samples.T)

    def test_numpy(self):
        """Test that the user can write to a npy file
        """
        parameters, samples = self.write("numpy", "numpy.npy")
        self.check_samples("{}/numpy.npy".format(tmpdir), parameters, samples.T)

    def test_pesummary(self):
        """Test that the user can write to a pesummary file
        """
        parameters, samples = self.write("pesummary", "pesummary.hdf5", label="label")
        self.check_samples(
            "{}/pesummary.hdf5".format(tmpdir), parameters, samples.T, pesummary=True
        )


class TestWritePESummary(object):
    """Test the `.write` function as part of the
    `pesummary.gw.file.formats.pesummary.PESummary class
    """
    @pytest.fixture(scope='class', autouse=True)
    def setup(self):
        """Setup the TestWritePESummary class
        """
        import os

        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        os.system(
           "curl https://dcc.ligo.org/public/0163/P190412/009/posterior_samples.h5 "
           "-o {}/GW190412_posterior_samples.h5".format(tmpdir)
        )
        type(self).result = read("{}/GW190412_posterior_samples.h5".format(tmpdir))
        type(self).posterior = type(self).result.samples_dict

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)

    def _write(self, file_format, extension, pesummary=False, **kwargs):
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        filename = {
            "IMRPhenomHM": "test.{}".format(extension),
            "IMRPhenomPv3HM": "test2.{}".format(extension)
        }
        self.result.write(
            labels=["IMRPhenomHM", "IMRPhenomPv3HM"], file_format=file_format,
            filenames=filename, outdir=tmpdir, **kwargs
        )
        if not pesummary:
            assert os.path.isfile("{}/test.{}".format(tmpdir, extension))
            assert os.path.isfile("{}/test2.{}".format(tmpdir, extension))
            one = read("{}/test.{}".format(tmpdir, extension))
            two = read("{}/test2.{}".format(tmpdir, extension))
            np.testing.assert_almost_equal(
                one.samples_dict["mass_1"], self.posterior["IMRPhenomHM"]["mass_1"]
            )
            np.testing.assert_almost_equal(
                two.samples_dict["mass_1"], self.posterior["IMRPhenomPv3HM"]["mass_1"]
            )
            os.system("rm {}/test.{}".format(tmpdir, extension))
            os.system("rm {}/test2.{}".format(tmpdir, extension))
        else:
            assert os.path.isfile("{}/test.h5".format(tmpdir))
            one = read("{}/test.h5".format(tmpdir))
            assert sorted(one.labels) == sorted(["IMRPhenomHM"])
            np.testing.assert_almost_equal(
                one.samples_dict["IMRPhenomHM"]["mass_1"],
                self.posterior["IMRPhenomHM"]["mass_1"]
            )
            np.testing.assert_almost_equal(
                one.psd["IMRPhenomHM"]["H1"], self.result.psd["IMRPhenomHM"]["H1"]
            )

    def test_write_dat(self):
        """Test write to dat
        """
        self._write("dat", "dat")

    def test_write_numpy(self):
        """Test write to numpy
        """
        self._write("numpy", "npy")

    def test_write_json(self):
        """Test write to dat
        """
        self._write("json", "json")

    def test_write_hdf5(self):
        """Test write to dat
        """
        self._write("hdf5", "h5")

    def test_write_bilby(self):
        """Test write to dat
        """
        self._write("bilby", "json")

    def test_write_pesummary(self):
        """Test write to dat
        """
        self._write("pesummary", "h5", pesummary=True)

    def test_write_lalinference(self):
        """Test write to dat
        """
        self._write("lalinference", "h5")
        self._write("lalinference", "dat", dat=True)
