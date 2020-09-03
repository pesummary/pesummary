import os
import shutil
import glob
import subprocess
import numpy as np

from .base import make_result_file, get_list_of_plots, get_list_of_files, data_dir
import pytest
from pesummary.utils.exceptions import InputError
import importlib


class Base(object):
    """Base class for testing the executables
    """
    def launch(self, command_line):
        """
        """
        args = command_line.split(" ")
        executable = args[0]
        cla = args[1:]
        module = importlib.import_module("pesummary.cli.{}".format(executable))
        print(cla)
        module.main(args=[i for i in cla if i != " " and i != ""])


class TestSummaryPages(Base):
    """Test the `summarypages` executable with trivial examples
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        self.dirs = [".outdir", ".outdir1", ".outdir2"]
        for dd in self.dirs:
            if not os.path.isdir(dd):
                os.mkdir(dd)
        make_result_file(gw=False, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_result_file(gw=False, extension="hdf5")
        os.rename(".outdir/test.h5", ".outdir/example2.h5")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        for dd in self.dirs:
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    def check_output(self, number=1, mcmc=False):
        """Check the output from the summarypages executable
        """
        assert os.path.isfile(".outdir/home.html")
        plots = get_list_of_plots(gw=False, number=number, mcmc=mcmc)
        assert all(
            i == j for i, j in zip(
                sorted(plots), sorted(glob.glob("./.outdir/plots/*.png"))
            )
        )
        files = get_list_of_files(gw=False, number=number)
        assert all(
            i == j for i, j in zip(
                sorted(files), sorted(glob.glob("./.outdir/html/*.html"))
            )
        )


    def test_prior_input(self):
        """Check that `summarypages` works when a prior file is passed from
        the command line
        """
        import importlib
        import pkg_resources

        path = pkg_resources.resource_filename("bilby", "gw")
        bilby_prior_file = os.path.join(
            path, "prior_files", "GW150914.prior"
        )

        for package in ["core", "gw"]:
            gw = True if package == "gw" else False
            module = importlib.import_module(
                "pesummary.{}.file.read".format(package)
            )
            make_result_file(gw=gw, extension="json")
            os.rename(".outdir/test.json", ".outdir/prior.json")
            for _file in [".outdir/prior.json", bilby_prior_file]:
                command_line = (
                    "summarypages --webdir .outdir --samples .outdir/example.json "
                    "--labels test --prior_file {}".format(_file)
                )
                command_line += " --gw" if gw else ""
                self.launch(command_line)
                f = module.read(".outdir/samples/posterior_samples.h5")
                if _file != bilby_prior_file:
                    stored = f.priors["samples"]["test"]
                    f = module.read(_file)
                    original = f.samples_dict
                    for param in original.keys():
                        np.testing.assert_almost_equal(
                            original[param], stored[param]
                        )
                else:
                    from bilby.core.prior import PriorDict

                    analytic = f.priors["analytic"]["test"]
                    bilby_prior = PriorDict(filename=bilby_prior_file)
                    for param, value in bilby_prior.items():
                        assert analytic[param] == str(value)

    def test_calibration_and_psd(self):
        """Test that the calibration and psd files are passed appropiately
        """
        from pesummary.gw.file.read import read
        from .base import make_psd, make_calibration

        make_psd()
        make_calibration()
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--psd H1:.outdir/psd.dat --calibration L1:.outdir/calibration.dat "
            "--labels test --posterior_samples_filename example.h5"
        )
        self.launch(command_line)
        f = read(".outdir/samples/example.h5")
        psd = np.genfromtxt(".outdir/psd.dat")
        calibration = np.genfromtxt(".outdir/calibration.dat")
        np.testing.assert_almost_equal(f.psd["test"]["H1"], psd)
        np.testing.assert_almost_equal(
            f.priors["calibration"]["test"]["L1"], calibration
        )

    def test_gracedb(self):
        """Test that when the gracedb ID is passed from the command line it is
        correctly stored in the meta data
        """
        from pesummary.gw.file.read import read

        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--gracedb G17864 --gw --labels test"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        assert "gracedb" in f.extra_kwargs[0]["meta_data"]
        assert "G17864" == f.extra_kwargs[0]["meta_data"]["gracedb"]["id"]

    def test_single(self):
        """Test on a single input
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json --label core0"
        )
        self.launch(command_line)
        self.check_output(number=1)

    def test_summarycombine_output(self):
        """Test on a summarycombine output
        """
        from .base import make_psd, make_calibration

        make_psd()
        make_calibration()
        command_line = (
            "summarycombine --webdir .outdir1 --samples "
            ".outdir/example.json --label gw0 "
            "--calibration L1:.outdir/calibration.dat --gw"
        )
        self.launch(command_line)
        command_line = (
            "summarycombine --webdir .outdir2 --samples "
            ".outdir/example.json --label gw1 "
            "--psd H1:.outdir/psd.dat --gw"
        )
        self.launch(command_line)
        command_line = (
            "summarycombine --webdir .outdir --gw --samples "
            ".outdir1/samples/posterior_samples.h5 "
            ".outdir2/samples/posterior_samples.h5 "
        )
        self.launch(command_line)
        command_line = (
            "summarypages --webdir .outdir --gw --samples "
            ".outdir/samples/posterior_samples.h5"
        )
        self.launch(command_line)
        

    def test_mcmc(self):
        """Test the `--mcmc_samples` command line argument
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            "--label core0 --mcmc_samples"
        )
        self.launch(command_line)
        self.check_output(number=1, mcmc=True)

    def test_mcmc_more_than_label(self):
        """Test that the code fails with the `--mcmc_samples` command line
        argument when multiple labels are passed.
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            ".outdir/example.json .outdir/example2.h5 "
            "--label core0 core1 --mcmc_samples"
        )
        with pytest.raises(InputError): 
            self.launch(command_line)

    def test_file_format_wrong_number(self):
        """Test that the code fails with the `--file_format` command line
        argument when the number of file formats does not match the number of
        samples
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            "--file_format hdf5 json dat"
        )
        with pytest.raises(InputError):
            self.launch(command_line)


class TestSummaryClassification(Base):
    """Test the `summaryclassification` executable
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True, gw=True, pesummary_label="test")
        os.rename(".outdir/test.json", ".outdir/pesummary.json")
        make_result_file(bilby=True, gw=True)
        os.rename(".outdir/test.json", ".outdir/bilby.json")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def check_output(self):
        """Check the output from the `summaryclassification` executable
        """
        import glob
        import json

        files = glob.glob(".outdir/*")
        assert ".outdir/test_default_prior_pe_classification.json" in files
        assert ".outdir/test_default_pepredicates_bar.png" in files
        with open(".outdir/test_default_prior_pe_classification.json", "r") as f:
            data = json.load(f)
        assert all(
            i in data.keys() for i in [
                "BNS", "NSBH", "BBH", "MassGap", "HasNS", "HasRemnant"
            ]
        )

    def test_result_file(self):
        """Test the `summaryclassification` executable for a random result file
        """
        command_line = (
            "summaryclassification --webdir .outdir --samples "
            ".outdir/bilby.json --prior default --label test"
        )
        self.launch(command_line)
        self.check_output()

    def test_pesummary_file(self):
        """Test the `summaryclassification` executable for a pesummary metafile
        """
        command_line = (
            "summaryclassification --webdir .outdir --samples "
            ".outdir/pesummary.json --prior default"
        )
        self.launch(command_line)
        self.check_output()


class TestSummaryClean(Base):
    """Test the `summaryclean` executable
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_clean(self):
        """Test the `summaryclean` executable
        """
        import h5py

        parameters = ["mass_ratio"]
        data = [[0.5], [0.5], [-1.5]]
        h5py_data = np.array(
            [tuple(i) for i in data], dtype=[tuple([i, 'float64']) for i in
            parameters]
        )
        f = h5py.File(".outdir/test.hdf5", "w")
        lalinference = f.create_group("lalinference")
        nest = lalinference.create_group("lalinference_nest")
        samples = nest.create_dataset("posterior_samples", data=h5py_data)
        f.close()
        command_line = (
            "summaryclean --webdir .outdir --samples .outdir/test.hdf5 "
            "--file_format dat --labels test"
        )
        self.launch(command_line)
        self.check_output()

    def check_output(self):
        """Check the output from the `summaryclean` executable
        """
        from pesummary.gw.file.read import read

        f = read(".outdir/pesummary_test.dat")
        print(f.samples_dict["mass_ratio"])
        assert len(f.samples_dict["mass_ratio"]) == 2
        assert all(i == 0.5 for i in f.samples_dict["mass_ratio"])


class _SummaryCombine_Metafiles(Base):
    """Test the `summarycombine_metafile` executable
    """
    def test_combine(self, gw=False):
        """Test the executable for 2 metafiles
        """
        make_result_file(pesummary=True, pesummary_label="label2")
        os.rename(".outdir/test.json", ".outdir/test2.json")
        make_result_file(pesummary=True)
        command_line = (
            "summarycombine --webdir .outdir "
            "--samples .outdir/test.json .outdir/test2.json "
            "--save_to_json"
        )
        if gw:
            command_line += " --gw"
        self.launch(command_line)

    def check_output(self, gw=False):
        if gw:
            from pesummary.gw.file.read import read
        else:
            from pesummary.core.file.read import read

        assert os.path.isfile(".outdir/samples/posterior_samples.json")
        combined = read(".outdir/samples/posterior_samples.json")
        for f in [".outdir/test.json", ".outdir/test2.json"]:
            data = read(f)
            labels = data.labels
            assert all(i in combined.labels for i in labels)
            assert all(
                all(
                    data.samples_dict[j][num] == combined.samples_dict[i][j][num]
                    for num in range(data.samples_dict[j])
                ) for j in data.samples_dict.keys()
            )


class TestCoreSummaryCombine_Metafiles(_SummaryCombine_Metafiles):
    """Test the `summarycombine_metafile` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_combine(self):
        """Test the executable for 2 metafiles
        """
        super(TestCoreSummaryCombine_Metafiles, self).test_combine(gw=False)

    def check_output(self):
        super(TestCoreSummaryCombine_Metafiles, self).check_output(gw=False)


class TestGWSummaryCombine_Metafiles(_SummaryCombine_Metafiles):
    """Test the `summarycombine_metafile` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_combine(self):
        """Test the executable for 2 metafiles
        """
        super(TestGWSummaryCombine_Metafiles, self).test_combine(gw=True)

    def check_output(self, gw=True):
        super(TestGWSummaryCombine_Metafiles, self).check_output(gw=True)


class TestSummaryCombine(Base):
    """Test the `summarycombine` executable
    """
    def setup(self):
        """Setup the SummaryCombine class
        """
        self.dirs = [".outdir"]
        for dd in self.dirs:
            if not os.path.isdir(dd):
                os.mkdir(dd)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        for dd in self.dirs:
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    def test_disable_prior_sampling(self):
        """Test that the code skips prior sampling when the appropiate flag
        is provided to the `summarypages` executable
        """
        from pesummary.io import read

        make_result_file(bilby=True, gw=False)
        os.rename(".outdir/test.json", ".outdir/bilby.json")
        command_line = (
            "summarycombine --webdir .outdir --samples .outdir/bilby.json "
            "--labels core0"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        assert len(f.priors["samples"]["core0"])

        command_line = (
            "summarycombine --webdir .outdir --samples .outdir/bilby.json "
            "--disable_prior_sampling --labels core0"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        assert not len(f.priors["samples"]["core0"])

    def test_external_hdf5_links(self):
        """Test that seperate hdf5 files are made when the
        `--external_hdf5_links` command line is passed
        """
        from pesummary.gw.file.read import read
        from .base import make_psd, make_calibration

        make_result_file(gw=True, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_psd()
        make_calibration()
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --external_hdf5_links --gw "
            "--psd H1:.outdir/psd.dat --calibration L1:.outdir/calibration.dat "
            "--no_conversion"
        )
        self.launch(command_line)
        assert os.path.isfile(
            os.path.join(".outdir", "samples", "posterior_samples.h5")
        )
        assert os.path.isfile(
            os.path.join(".outdir", "samples", "_gw0.h5")
        )
        f = read(".outdir/samples/posterior_samples.h5")
        g = read(".outdir/example.json")
        h = read(".outdir/samples/_gw0.h5")
        np.testing.assert_almost_equal(f.samples[0], g.samples)
        np.testing.assert_almost_equal(f.samples[0], h.samples[0])
        np.testing.assert_almost_equal(f.psd["gw0"]["H1"], h.psd["gw0"]["H1"])
        np.testing.assert_almost_equal(
            f.priors["calibration"]["gw0"]["L1"],
            h.priors["calibration"]["gw0"]["L1"]
        )

    def test_compression(self):
        """Test that the metafile is reduced in size when the datasets are
        compressed with maximum compression level
        """
        from pesummary.gw.file.read import read
        from .base import make_psd, make_calibration

        make_result_file(gw=True, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_psd()
        make_calibration()
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--psd H1:.outdir/psd.dat --calibration L1:.outdir/calibration.dat "
        )
        self.launch(command_line)
        original_size = os.stat("./.outdir/samples/posterior_samples.h5").st_size 
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--psd H1:.outdir/psd.dat --calibration L1:.outdir/calibration.dat "
            "--hdf5_compression 9 --posterior_samples_filename posterior_samples2.h5"
        )
        self.launch(command_line)
        compressed_size = os.stat("./.outdir/samples/posterior_samples2.h5").st_size
        assert compressed_size < original_size

        f = read("./.outdir/samples/posterior_samples.h5")
        g = read("./.outdir/samples/posterior_samples2.h5")
        posterior_samples = f.samples[0]
        posterior_samples2 = g.samples[0]
        np.testing.assert_almost_equal(posterior_samples, posterior_samples2)

    def test_seed(self):
        """Test that the samples stored in the metafile are identical for two
        runs if the random seed is the same
        """
        from pesummary.gw.file.read import read

        make_result_file(gw=True, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--nsamples 10 --seed 1000"
        )
        self.launch(command_line)
        original = read(".outdir/samples/posterior_samples.h5")
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--nsamples 10 --seed 2000"
        )
        self.launch(command_line)
        new = read(".outdir/samples/posterior_samples.h5")
        try:
            np.testing.assert_almost_equal(
                original.samples[0], new.samples[0]
            )
            raise AssertionError("Failed")
        except AssertionError:
            pass

        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--nsamples 10 --seed 1000"
        )
        self.launch(command_line)
        original = read(".outdir/samples/posterior_samples.h5")
        command_line = (
            "summarycombine --webdir .outdir --samples "
            ".outdir/example.json --label gw0 --no_conversion --gw "
            "--nsamples 10 --seed 1000"
        )
        self.launch(command_line)
        new = read(".outdir/samples/posterior_samples.h5")
        np.testing.assert_almost_equal(
            original.samples[0], new.samples[0]
        )


class TestSummaryReview(Base):
    """Test the `summaryreview` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(lalinference=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_review(self):
        """Test the `summaryreview` script for a `lalinference` result file
        """
        command_line = (
            "summaryreview --webdir .outdir --samples .outdir/test.hdf5"
        )
        self.launch(command_line)


class TestSummaryModify(Base):
    """Test the `summarymodify` executable
    """
    def setup(self):
        """Setup the SummaryModify class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(
            pesummary=True, pesummary_label="replace", extension="hdf5"
        )

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_modify_kwargs_replace(self):
        """Test that kwargs are correctly replaced in the meta file
        """
        import h5py

        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 "
            "--delimiter / --kwargs replace/log_evidence:1000"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        original_data = h5py.File(".outdir/test.h5", "r")
        data = h5py.File(".outdir/test.h5", "r")
        assert original_data["replace"]["meta_data"]["sampler"]["log_evidence"][0] != b'1000'
        assert modified_data["replace"]["meta_data"]["sampler"]["log_evidence"][0] == b'1000'
        modified_data.close()
        original_data.close()

    def test_modify_kwargs_append(self):
        """Test that kwargs are correctly added to the result file
        """
        import h5py

        original_data = h5py.File(".outdir/test.h5", "r")
        assert "other" not in original_data["replace"]["meta_data"].keys()
        original_data.close()
        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 "
            "--delimiter / --kwargs replace/test:10 "
            "--overwrite"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/test.h5", "r")
        assert modified_data["replace"]["meta_data"]["other"]["test"][0] == b'10'
        modified_data.close()

    def test_modify_posterior(self):
        """Test that a posterior distribution is correctly modified
        """
        import h5py

        new_posterior = np.random.uniform(10, 0.5, 1000)
        np.savetxt(".outdir/different_posterior.dat", new_posterior)
        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 --delimiter ; "
            "--replace_posterior replace;mass_1:.outdir/different_posterior.dat"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        np.testing.assert_almost_equal(
            modified_data["replace"]["posterior_samples"]["mass_1"], new_posterior
        )
        modified_data.close()
        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 --delimiter ; "
            "--replace_posterior replace;abc:.outdir/different_posterior.dat"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        np.testing.assert_almost_equal(
            modified_data["replace"]["posterior_samples"]["abc"], new_posterior
        )
        modified_data.close()

    def test_remove_posterior(self):
        """Test that a posterior is correctly removed
        """
        import h5py

        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 --delimiter ; "
            "--remove_posterior replace;mass_1"
        )
        self.launch(command_line)
        original_data = h5py.File(".outdir/test.h5", "r")
        params = list(original_data["replace"]["posterior_samples"]["parameter_names"])
        if isinstance(params[0], bytes):
            params = [param.decode("utf-8") for param in params]
        assert "mass_1" in params
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        assert "mass_1" not in modified_data["replace"]["posterior_samples"].dtype.names
        original_data.close()
        modified_data.close()

    def test_remove_multiple_posteriors(self):
        """Test that multiple posteriors are correctly removed
        """
        import h5py

        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 --delimiter ; "
            "--remove_posterior replace;mass_1 replace;mass_2"
        )
        self.launch(command_line)
        original_data = h5py.File(".outdir/test.h5", "r")
        params = list(original_data["replace"]["posterior_samples"]["parameter_names"])
        if isinstance(params[0], bytes):
            params = [param.decode("utf-8") for param in params]
        assert "mass_1" in params
        assert "mass_2" in params
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        assert "mass_1" not in modified_data["replace"]["posterior_samples"].dtype.names
        assert "mass_2" not in modified_data["replace"]["posterior_samples"].dtype.names
        original_data.close()
        modified_data.close()

    def test_store_skymap(self):
        """Test that multiple skymaps are correctly stored
        """
        import astropy_healpix as ah
        from ligo.skymap.io.fits import write_sky_map
        import h5py

        nside = 128
        npix = ah.nside_to_npix(nside)
        prob = np.random.random(npix)
        prob /= sum(prob)

        write_sky_map(
            '.outdir/test.fits', prob,
            objid='FOOBAR 12345',
            gps_time=10494.3,
            creator="test",
            origin='LIGO Scientific Collaboration',
        )
        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 "
            "--store_skymap replace:.outdir/test.fits"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        assert "skymap" in modified_data["replace"].keys()
        np.testing.assert_almost_equal(
            modified_data["replace"]["skymap"]["data"], prob
        )
        np.testing.assert_almost_equal(
            modified_data["replace"]["skymap"]["meta_data"]["gps_time"][0], 10494.3
        )
        _creator = modified_data["replace"]["skymap"]["meta_data"]["creator"][0]
        if isinstance(_creator, bytes):
            _creator = _creator.decode("utf-8")
        assert _creator == "test"

        command_line = (
            "summarymodify --webdir .outdir "
            "--samples .outdir/modified_posterior_samples.h5 "
            "--store_skymap replace:.outdir/test.fits --force_replace"
        )
        self.launch(command_line)
        command_line = (
            "summarypages --webdir .outdir/webpage --gw --no_conversion "
            "--samples .outdir/modified_posterior_samples.h5"
        )
        self.launch(command_line)
        data = h5py.File(".outdir/webpage/samples/posterior_samples.h5", "r")
        np.testing.assert_almost_equal(data["replace"]["skymap"]["data"], prob)
        data.close()
        with pytest.raises(ValueError):
            command_line = (
                "summarymodify --webdir .outdir "
                "--samples .outdir/modified_posterior_samples.h5 "
                "--store_skymap replace:.outdir/test.fits"
            )
            self.launch(command_line)

    def test_modify(self):
        """Test the `summarymodify` script
        """
        import h5py

        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 "
            "--labels replace:new"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        data = h5py.File(".outdir/test.h5", "r")
        assert "replace" not in list(modified_data.keys())
        assert "new" in list(modified_data.keys())
        for key in data["replace"].keys():
            assert key in modified_data["new"].keys()
            for i, j in zip(data["replace"][key], modified_data["new"][key]):
                try:
                    if isinstance(data["replace"][key][i],h5py._hl.dataset.Dataset):
                        try:
                            assert all(k == l for k, l in zip(
                                data["replace"][key][i],
                                modified_data["new"][key][j]
                            ))
                        except ValueError:
                            assert all(
                                all(m == n for m, n in zip(k, l)) for k, l in zip(
                                    data["replace"][key][i],
                                    modified_data["new"][key][j]
                                )
                            )
                except TypeError:
                    pass
        data.close()
        modified_data.close()


class TestSummaryRecreate(Base):
    """Test the `summaryrecreate` executable
    """
    def setup(self):
        """Setup the SummaryRecreate class
        """
        import configparser

        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(data_dir + "/config_lalinference.ini")
        config_dictionary = dict(config._sections)
        config_dictionary["paths"]["webdir"] = (
            "./{}/webdir".format(os.environ["USER"])
        )
        make_result_file(
            pesummary=True, pesummary_label="recreate", extension="hdf5",
            config=config_dictionary
        )
        with open("GW150914.txt", "w") as f:
            f.writelines(["115"])

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_recreate(self):
        """Test the `summaryrecreate` script
        """
        import configparser

        command_line = (
            "summaryrecreate --rundir .outdir --samples .outdir/test.h5 "
        )
        self.launch(command_line)
        assert os.path.isdir(os.path.join(".outdir", "recreate"))
        assert os.path.isfile(os.path.join(".outdir", "recreate", "config.ini"))
        assert os.path.isdir(os.path.join(".outdir", "recreate", "outdir"))
        assert os.path.isdir(os.path.join(".outdir", "recreate", "outdir", "caches"))
        config = configparser.ConfigParser()
        config.read(os.path.join(".outdir", "recreate", "config.ini"))
        original_config = configparser.ConfigParser()
        original_config.read(data_dir + "/config_lalinference.ini")
        for a, b in zip(
          sorted(config.sections()), sorted(original_config.sections())
        ):
            assert a == b
            for key, item in config[a].items():
                assert config[b][key] == item
        command_line = (
            "summaryrecreate --rundir .outdir_modify --samples .outdir/test.h5 "
            "--config_override approx:IMRPhenomPv3HM srate:4096"
        )
        self.launch(command_line)
        config = configparser.ConfigParser()
        config.read(os.path.join(".outdir_modify", "recreate", "config.ini"))
        original_config = configparser.ConfigParser()
        original_config.read(data_dir + "/config_lalinference.ini")
        for a, b in zip(
          sorted(config.sections()), sorted(original_config.sections())
        ):
            assert a == b
            for key, item in config[a].items():
                if key == "approx":
                    assert original_config[b][key] != item
                    assert config[b][key] == "IMRPhenomPv3HM"
                elif key == "srate":
                    assert original_config[b][key] != item
                    assert config[b][key] == "4096"
                elif key == "webdir":
                    pass
                else:
                    assert original_config[b][key] == item


class TestSummaryCompare(Base):
    """Test the SummaryCompare executable
    """
    def setup(self):
        """Setup the SummaryCompare class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_example_in_docs(self):
        """Test that the code runs for the example in the docs
        """
        import numpy as np
        from pesummary.io import write

        parameters = ["a", "b", "c", "d"]
        data = np.random.random([100, 4])
        write(
            parameters, data, file_format="dat", outdir=".outdir",
            filename="example1.dat"
        )
        parameters2 = ["a", "b", "c", "d", "e"]
        data2 = np.random.random([100, 5])
        write(
            parameters2, data2, file_format="json", outdir=".outdir",
            filename="example2.json"
        )
        command_line = (
            "summarycompare --samples .outdir/example1.dat "
            ".outdir/example2.json --properties_to_compare posterior_samples -v"
        )
        self.launch(command_line)


class TestSummaryJSCompare(Base):
    """Test the `summaryjscompare` executable
    """
    def setup(self):
        """Setup the SummaryJSCompare class
        """
        self.dirs = [".outdir"]
        for dd in self.dirs:
            if not os.path.isdir(dd):
                os.mkdir(dd)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        for dd in self.dirs:
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    def test_runs_on_core_file(self):
        """Test that the code successfully generates a plot for 2 core result files
        """
        make_result_file(bilby=True, gw=False)
        os.rename(".outdir/test.json", ".outdir/bilby.json")
        make_result_file(bilby=True, gw=False)
        os.rename(".outdir/test.json", ".outdir/bilby2.json")
        command_line = (
            "summaryjscompare --event test-bilby1-bilby2 --main_keys  a b c d --webdir .outdir --samples .outdir/bilby.json "
            ".outdir/bilby2.json --labels bilby_1 bilby_2"
        )
        self.launch(command_line)

    def test_runs_on_gw_file(self):
        """Test that the code successfully generates a plot for 2 gw result files
        """
        make_result_file(bilby=True, gw=True)
        os.rename(".outdir/test.json", ".outdir/bilby.json")
        make_result_file(lalinference=True)
        os.rename(".outdir/test.hdf5", ".outdir/lalinference.hdf5")
        command_line = (
            "summaryjscompare --event test-bilby-lalinf --main_keys mass_1 mass_2 a_1 a_2 --webdir .outdir --samples .outdir/bilby.json "
            ".outdir/lalinference.hdf5 --labels bilby lalinf"
        )
        self.launch(command_line)
