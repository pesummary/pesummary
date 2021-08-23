# License under an MIT style license -- see LICENSE.md

import os
import shutil
import glob
import subprocess
import numpy as np

from .base import (
    make_result_file, get_list_of_plots, get_list_of_files, data_dir
)
import pytest
from pesummary.utils.exceptions import InputError
import importlib

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        return module.main(args=[i for i in cla if i != " " and i != ""])


class TestSummaryPublication(Base):
    """Test the `summarypublication` executable with trivial examples
    """
    def setup(self):
        """Setup the SummaryPublication class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(bilby=True, gw=True)
        os.rename(".outdir/test.json", ".outdir/bilby.json")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_2d_contour(self):
        """Test the 2d contour plot generation
        """
        command_line = (
            "summarypublication --webdir .outdir --samples .outdir/bilby.json "
            "--labels test --parameters mass_1 mass_2 --levels 0.9 0.5 "
            "--plot 2d_contour --palette colorblind"
        )
        self.launch(command_line)
        assert os.path.isfile(
            os.path.join(".outdir", "2d_contour_plot_mass_1_and_mass_2.png")
        )

    def test_violin(self):
        """Test the violin plot generation
        """
        command_line = (
            "summarypublication --webdir .outdir --samples .outdir/bilby.json "
            "--labels test --parameters mass_1 --plot violin "
            "--palette colorblind"
        )
        self.launch(command_line)
        assert os.path.isfile(
            os.path.join(".outdir", "violin_plot_mass_1.png")
        )

    def test_spin_disk(self):
        """Test the spin disk generation
        """
        command_line = (
            "summarypublication --webdir .outdir --samples .outdir/bilby.json "
            "--labels test --parameters mass_1 --plot spin_disk "
            "--palette colorblind"
        )
        self.launch(command_line)
        assert os.path.isfile(
            os.path.join(".outdir", "spin_disk_plot_test.png")
        )


class TestSummaryPipe(Base):
    """Test the `summarypipe` executable with trivial examples
    """
    def setup(self):
        """Setup the SummaryPipe class
        """
        self.dirs = [
            ".outdir", ".outdir/lalinference", ".outdir/bilby",
            ".outdir/lalinference/posterior_samples",
            ".outdir/lalinference/ROQdata", ".outdir/lalinference/engine",
            ".outdir/lalinference/caches", ".outdir/lalinference/log",
            ".outdir/bilby/data", ".outdir/bilby/result",
            ".outdir/bilby/submit", ".outdir/bilby/log_data_analysis"
        ]
        for dd in self.dirs:
            if not os.path.isdir(dd):
                os.mkdir(dd)
        make_result_file(
            gw=False, lalinference=True,
            outdir=".outdir/lalinference/posterior_samples/"
        )
        os.rename(
            ".outdir/lalinference/posterior_samples/test.hdf5",
            ".outdir/lalinference/posterior_samples/posterior_HL_result.hdf5"
        )
        make_result_file(
            gw=False, bilby=True, outdir=".outdir/bilby/result/"
        )
        os.rename(
            ".outdir/bilby/result/test.json",
            ".outdir/bilby/result/label_result.json"
        )

    def add_config_file(self):
        shutil.copyfile(
            os.path.join(data_dir, "config_lalinference.ini"),
            ".outdir/lalinference/config.ini"
        )
        shutil.copyfile(
            os.path.join(data_dir, "config_bilby.ini"),
            ".outdir/bilby/config.ini"
        )

    def teardown(self):
        """Remove the files and directories created from this class
        """
        for dd in self.dirs:
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    def test_no_config(self):
        """Test that the code fails if there is no config file in the
        directory
        """
        for _type in ["lalinference", "bilby"]:
            command_line = "summarypipe --rundir .outdir/{}".format(_type)
            with pytest.raises(FileNotFoundError):
                self.launch(command_line)

    def test_no_samples(self):
        """Test that the code fails if there are no posterior samples in the
        directory
        """
        self.add_config_file()
        for _type in ["lalinference", "bilby"]:
            if _type == "lalinference":
                os.remove(
                    ".outdir/{}/posterior_samples/posterior_HL_result.hdf5".format(
                        _type
                    )
                )
            else:
                os.remove(
                    ".outdir/{}/result/label_result.json".format(_type)
                )
            command_line = "summarypipe --rundir .outdir/{}".format(_type)
            with pytest.raises(FileNotFoundError):
                self.launch(command_line)

    def test_basic(self):
        """Test that the code runs for a trivial example
        """
        self.add_config_file()
        for _type in ["lalinference", "bilby"]:
            command_line = (
                "summarypipe --rundir .outdir/{} --return_string".format(_type)
            )
            output = self.launch(command_line)
            assert "--config" in output
            assert ".outdir/{}/config.ini".format(_type) in output
            assert "--samples" in output
            if _type == "lalinference":
                _f = (
                    ".outdir/{}/posterior_samples/posterior_HL_result.hdf5".format(
                        _type
                    )
                )
            else:
                _f = ".outdir/{}/result/label_result.json".format(_type)
            assert _f in output
            assert "--webdir" in output
            assert "--approximant" in output
            assert "--labels" in output

    def test_override(self):
        """Test that when you provide an option from the command line it
        overrides the one inferred from the rundir
        """
        self.add_config_file()
        command_line = (
            "summarypipe --rundir .outdir/lalinference --return_string"
        )
        output = self.launch(command_line)
        command_line += " --labels hello"
        output2 = self.launch(command_line)
        assert output != output2
        label = output.split(" ")[output.split(" ").index("--labels") + 1]
        label2 = output2.split(" ")[output2.split(" ").index("--labels") + 1]
        assert label != label2
        assert label2 == "hello"

    def test_add_to_summarypages_command(self):
        """Test that when you provide an option from the command line that
        is not already in the summarypages command line, it adds it to the one
        inferred from the rundir
        """
        self.add_config_file()
        command_line = (
            "summarypipe --rundir .outdir/lalinference --return_string"
        )
        output = self.launch(command_line)
        command_line += " --multi_process 10 --kde_plot --cosmology Planck15_lal"
        output2 = self.launch(command_line)
        assert output != output2
        assert "--multi_process 10" in output2
        assert "--cosmology Planck15_lal" in output2
        assert "--kde_plot" in output2
        assert "--multi_process 10" not in output
        assert "--cosmology Planck15_lal" not in output
        assert "--kde_plot" not in output


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

    def check_output(
        self, number=1, mcmc=False, existing_plot=False, expert=False,
        gw=False
    ):
        """Check the output from the summarypages executable
        """
        assert os.path.isfile(".outdir/home.html")
        plots = get_list_of_plots(
            gw=gw, number=number, mcmc=mcmc, existing_plot=existing_plot,
            expert=expert
        )
        assert all(
            i == j for i, j in zip(
                sorted(plots), sorted(glob.glob("./.outdir/plots/*.png"))
            )
        )
        files = get_list_of_files(
            gw=gw, number=number, existing_plot=existing_plot
        )
        assert all(
            i == j for i, j in zip(
                sorted(files), sorted(glob.glob("./.outdir/html/*.html"))
            )
        )

    def test_descriptions(self):
        """Check that summarypages stores the correct descriptions when the
        `--descriptions` flag is provided
        """
        import json
        from pesummary.io import read
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            ".outdir/example.json --labels core0 core1 --nsamples 100 "
            "--disable_expert --disable_corner --descriptions core0:Description"
        )
        self.launch(command_line)
        opened = read(".outdir/samples/posterior_samples.h5")
        assert opened.description["core0"] == "Description"
        assert opened.description["core1"] == "No description found"

        with open(".outdir/descriptions.json", "w") as f:
            json.dump({"core0": "Testing description", "core1": "Test"}, f)
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            ".outdir/example.json --labels core0 core1 --nsamples 100 "
            "--disable_expert --disable_corner --descriptions .outdir/descriptions.json"
        )
        self.launch(command_line)
        opened = read(".outdir/samples/posterior_samples.h5")
        assert opened.description["core0"] == "Testing description"
        assert opened.description["core1"] == "Test"

    def test_reweight(self):
        """Check that summarypages reweights the posterior samples if the
        `--reweight_samples` flag is provided
        """
        from pesummary.io import read
        make_result_file(gw=True, extension="json")
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/test.json --gw "
            "--labels gw0 --nsamples 100 --disable_expert --disable_corner "
            "--reweight_samples uniform_in_comoving_volume "
        )
        self.launch(command_line)
        self.check_output(number=1, expert=False, gw=True)
        original = read(".outdir/test.json").samples_dict
        _reweighted = read(".outdir/samples/posterior_samples.h5")
        reweighted = _reweighted.samples_dict
        assert original.number_of_samples >= reweighted["gw0"].number_of_samples
        inds = np.array([
            original.parameters.index(param) for param in
            reweighted["gw0"].parameters if param in original.parameters
        ])
        assert all(
            reweighted_sample[inds] in original.samples.T for reweighted_sample
            in reweighted["gw0"].samples.T
        )
        _kwargs = _reweighted.extra_kwargs[0]
        assert _kwargs["sampler"]["nsamples_before_reweighting"] == 100
        assert _kwargs["sampler"]["nsamples"] == reweighted["gw0"].number_of_samples
        assert _kwargs["meta_data"]["reweighting"] == "uniform_in_comoving_volume"

    def test_checkpoint(self):
        """Check that when restarting from checkpoint, the outputs are
        consistent
        """
        import time
        command_line = (
            "summarypages --webdir .outdir --samples  .outdir/example.json "
            "--labels core0 --nsamples 100 --disable_expert --restart_from_checkpoint"
        )
        t0 = time.time()
        self.launch(command_line)
        t1 = time.time()
        assert os.path.isfile(".outdir/checkpoint/pesummary_resume.pickle")
        self.check_output(number=1, expert=False)
        t2 = time.time()
        self.launch(command_line)
        t3 = time.time()
        assert t3 - t2 < t1 - t0
        self.check_output(number=1, expert=False)
        # get timestamp of plot
        made_time = os.path.getmtime(glob.glob(".outdir/plots/*.png")[0])
        assert made_time < t2

    def test_expert(self):
        """Check that summarypages produces the expected expert diagnostic
        plots
        """
        command_line = (
            "summarypages --webdir .outdir --samples  .outdir/example.json "
            "--labels core0 --nsamples 100 --disable_expert"
        )
        self.launch(command_line)
        self.check_output(number=1, expert=False)
        command_line = (
            "summarypages --webdir .outdir --samples  .outdir/example.json "
            "--labels core0 --nsamples 100"
        )
        self.launch(command_line)
        self.check_output(number=1, expert=True)

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
                    "--labels test --prior_file {} --nsamples_for_prior "
                    "10 --disable_expert".format(_file)
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
                        # Non-bilby prior file will have same number or prior
                        # samples as posterior samples
                        assert len(stored[param]) == 1000
                else:
                    from bilby.core.prior import PriorDict

                    analytic = f.priors["analytic"]["test"]
                    bilby_prior = PriorDict(filename=bilby_prior_file)
                    for param, value in bilby_prior.items():
                        assert analytic[param] == str(value)
                    params = list(f.priors["samples"]["test"].keys())
                    # A bilby prior file will have 10 prior samples
                    assert len(f.priors["samples"]["test"][params[0]]) == 10

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
            "--labels test --posterior_samples_filename example.h5 "
            "--disable_expert"
        )
        self.launch(command_line)
        f = read(".outdir/samples/example.h5")
        psd = np.genfromtxt(".outdir/psd.dat")
        calibration = np.genfromtxt(".outdir/calibration.dat")
        np.testing.assert_almost_equal(f.psd["test"]["H1"], psd)
        np.testing.assert_almost_equal(
            f.priors["calibration"]["test"]["L1"], calibration
        )

    def test_strain_data(self):
        """Test that the gravitational wave data is passed appropiately
        """
        from pesummary.io import read
        from gwpy.timeseries import TimeSeries

        H1_series = TimeSeries(
            np.random.uniform(-1, 1, 1000), t0=101, dt=0.1, name="H1:test"
        )
        H1_series.write(".outdir/H1.gwf", format="gwf")
        L1_series = TimeSeries(
            np.random.uniform(-1, 1, 1000), t0=201, dt=0.2, name="L1:test"
        )
        L1_series.write(".outdir/L1.hdf", format="hdf5")
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--gwdata H1:test:.outdir/H1.gwf L1:test:.outdir/L1.hdf "
            "--labels test --disable_expert --disable_corner --disable_interactive"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        gwdata = f.gwdata
        assert all(IFO in gwdata.detectors for IFO in ["H1", "L1"])
        strain = {"H1": H1_series, "L1": L1_series}
        for IFO in gwdata.detectors:
            np.testing.assert_almost_equal(gwdata[IFO].value, strain[IFO].value)
            assert gwdata[IFO].t0 == strain[IFO].t0
            assert gwdata[IFO].dt == strain[IFO].dt
            assert gwdata[IFO].unit == strain[IFO].unit

    def test_gracedb(self):
        """Test that when the gracedb ID is passed from the command line it is
        correctly stored in the meta data
        """
        from pesummary.gw.file.read import read

        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--gracedb G17864 --gw --labels test --disable_expert"
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
            ".outdir/example.json --label core0 --disable_expert"
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
            ".outdir/samples/posterior_samples.h5 --disable_expert"
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

    def test_kde_plot(self):
        """Test that the kde plots work on a single input and on MCMC inputs
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json --label core0 --kde_plot --disable_expert"
        )
        self.launch(command_line)
        self.check_output(number=1)
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            "--label core0 --mcmc_samples --kde_plot"
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

    def test_add_existing_plot(self):
        """Test that an Additional page is made if existing plots are provided
        to the summarypages executable
        """
        with open(".outdir/test.png", "w") as f:
            f.writelines("")
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json --label core0 --add_existing_plot "
            "core0:.outdir/test.png --disable_expert"
        )
        self.launch(command_line)
        self.check_output(number=1, existing_plot=True)
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example.json --label core0 core1 "
            "--add_existing_plot core0:.outdir/test.png core1:.outdir/test.png "
            "--disable_expert"
        )
        self.launch(command_line)
        self.check_output(number=2, existing_plot=True)


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


class TestSummaryTGR(Base):
    """Test the `summarytgr` executable
    """
    def setup(self):
        """Setup the SummaryTGR class
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

    def check_output(self, diagnostic=True):
        """Check the output from the `summarytgr` executable
        """
        import glob

        image_files = glob.glob(".outdir/plots/*")
        image_base_string = ".outdir/plots/primary_imrct_{}.png"
        file_strings = ["deviations_triangle_plot"]
        if diagnostic:
            file_strings += [
                "mass_1_mass_2", "a_1_a_2",
                "final_mass_non_evolved_final_spin_non_evolved"
            ]
        for file_string in file_strings:
            assert image_base_string.format(file_string) in image_files

    def test_result_file(self):
        """Test the `summarytgr` executable for a random result file
        """
        command_line = (
            "summarytgr --webdir .outdir "
            "--samples .outdir/bilby.json .outdir/bilby.json "
            "--test imrct "
            "--labels inspiral postinspiral "
            "--imrct_kwargs N_bins:11 "
            "--make_diagnostic_plots "
            "--disable_pe_page_generation"
        )
        self.launch(command_line)
        self.check_output()

    def test_pesummary_file(self):
        """Test the `summarytgr` executable for a pesummary metafile
        """
        command_line = (
            "summarytgr --webdir .outdir --samples "
            ".outdir/pesummary.json .outdir/pesummary.json --labels "
            "test:inspiral test:postinspiral --test imrct --imrct_kwargs "
            "N_bins:11 --disable_pe_page_generation"
        )
        self.launch(command_line)
        self.check_output(diagnostic=False)

    def test_pdfs_and_gr_quantile(self):
        """Test that the GR quantile and pdf matches the LAL implementation
        The LAL files were produced by the executable imrtgr_imr_consistency_test
        with N_bins=201 dMfbyMf_lim=3 dchifbychif_lim=3 and bbh_average_fits_precessing
        """
        from pesummary.io import read

        make_result_file(outdir="./", extension="dat", gw=True, random_seed=123456789)
        os.rename("./test.dat", ".outdir/inspiral.dat")
        make_result_file(outdir="./", extension="dat", gw=True, random_seed=987654321)
        os.rename("./test.dat", ".outdir/postinspiral.dat")
        command_line = (
                "summarytgr --webdir .outdir "
                "--samples .outdir/inspiral.dat .outdir/postinspiral.dat "
                "--test imrct "
                "--labels inspiral postinspiral "
                "--imrct_kwargs N_bins:201 final_mass_deviation_lim:3 final_spin_deviation_lim:3 "
                "--disable_pe_page_generation"
        )
        self.launch(command_line)
        f = read(".outdir/samples/tgr_samples.h5")
        pesummary_quantile = f.extra_kwargs["primary"]["GR Quantile (%)"]
        probdict = f.imrct_deviation["final_mass_final_spin_deviations"]
        lal_pdf = np.loadtxt(os.path.join(data_dir, "lal_pdf_for_summarytgr.dat.gz"))
        pesummary_pdf = probdict.probs / probdict.dx / probdict.dy

        np.testing.assert_almost_equal(pesummary_quantile, 3.276372814744687306)
        np.testing.assert_almost_equal(pesummary_pdf, lal_pdf)


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
            "summaryreview --webdir .outdir --samples .outdir/test.hdf5 "
            "--test core_plots"
        )
        self.launch(command_line)


class TestSummarySplit(Base):
    """Test the `summarysplit` executable
    """
    def setup(self):
        """Setup the SummarySplit class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(gw=False, extension="json")
        make_result_file(gw=False, extension="hdf5", n_samples=500)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_split_single_analysis(self):
        """Test that a file containing a single analysis is successfully split
        into N_samples result files
        """
        from pesummary.io import read
        command_line = (
            "summarysplit --samples .outdir/test.json --file_format json "
            "--outdir .outdir/split"
        )
        self.launch(command_line)
        original = read(".outdir/test.json").samples_dict
        files = glob.glob(".outdir/split/*.json")
        assert len(files) == original.number_of_samples
        for num, f in enumerate(files):
            g = read(f).samples_dict
            assert g.number_of_samples == 1
            idx = int(f.split("/")[-1].split("_")[-1].split(".")[0])
            for param in g.keys():
                assert g[param] == original[param][idx]
        command_line = (
            "summarycombine_posteriors --use_all --samples {} "
            "--outdir .outdir --filename combined_split.dat "
            "--file_format dat --labels {}"
        ).format(
            " ".join(files), " ".join(
                np.arange(original.number_of_samples).astype(str)
            )
        )
        self.launch(command_line)
        combined = read(".outdir/combined_split.dat").samples_dict
        assert all(param in original.keys() for param in combined.keys())
        for param in original.keys():
            assert all(sample in combined[param] for sample in original[param])
            assert all(sample in original[param] for sample in combined[param])

    def test_split_single_analysis_specific_N_files(self):
        """Test that a file containing a single analysis is successfully split
        into 10 result files
        """
        from pesummary.io import read
        command_line = (
            "summarysplit --samples .outdir/test.json --file_format json "
            "--outdir .outdir/split --N_files 10"
        )
        self.launch(command_line)
        original = read(".outdir/test.json").samples_dict
        files = glob.glob(".outdir/split/*.json")
        assert len(files) == 10
        for num, f in enumerate(files):
            g = read(f).samples_dict
            for param in g.keys():
                assert all(sample in original[param] for sample in g[param])

    def test_split_multi_analysis(self):
        """Test that a file containing multiple analyses is successfully split
        into N_samples result files
        """
        from pesummary.io import read
        command_line = (
            "summarycombine --webdir .outdir --samples .outdir/test.json "
            ".outdir/test.h5 --labels one two"
        )
        self.launch(command_line)
        command_line = (
            "summarysplit --samples .outdir/samples/posterior_samples.h5 "
            "--file_format hdf5 --outdir .outdir/split"
        )
        self.launch(command_line)
        assert os.path.isdir(".outdir/split/one")
        assert os.path.isdir(".outdir/split/two")
        zipped = zip(["one", "two"], [".outdir/test.json", ".outdir/test.h5"])
        for analysis, f in zipped:
            original = read(f).samples_dict
            files = glob.glob(".outdir/split/{}/*.hdf5".format(analysis))
            assert len(files) == original.number_of_samples
            for num, g in enumerate(files):
                h = read(g).samples_dict
                assert h.number_of_samples == 1
                idx = int(g.split("/")[-1].split("_")[-1].split(".")[0])
                for param in h.keys():
                    assert h[param] == original[param][idx]

class TestSummaryExtract(Base):
    """Test the `summaryextract` executable
    """
    def setup(self):
        """Setup the SummaryExtract class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(gw=False, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_result_file(gw=False, extension="hdf5")
        os.rename(".outdir/test.h5", ".outdir/example2.h5")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_extract(self):
        """Test that a set if posterior samples are correctly extracted
        """
        from pesummary.io import read
        command_line = (
            "summarycombine --samples .outdir/example.json .outdir/example2.h5 "
            "--labels one two --webdir .outdir"
        )
        self.launch(command_line)
        command_line = (
            "summaryextract --outdir .outdir --filename one.dat --file_format dat "
            "--samples .outdir/samples/posterior_samples.h5 --label one"
        )
        self.launch(command_line)
        assert os.path.isfile(".outdir/one.dat")
        extracted = read(".outdir/one.dat").samples_dict
        original = read(".outdir/example.json").samples_dict
        assert all(param in extracted.keys() for param in original.keys())
        np.testing.assert_almost_equal(extracted.samples, original.samples)


class TestSummaryCombine_Posteriors(Base):
    """Test the `summarycombine_posteriors` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Posteriors class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(gw=True, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_result_file(gw=True, extension="hdf5")
        os.rename(".outdir/test.h5", ".outdir/example2.h5")
        make_result_file(gw=True, extension="dat")
        os.rename(".outdir/test.dat", ".outdir/example3.dat")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_combine(self):
        """Test that the two posteriors are combined
        """
        from pesummary.io import read
        command_line = (
            "summarycombine_posteriors --outdir .outdir --filename test.dat "
            "--file_format dat --samples .outdir/example.json .outdir/example2.h5 "
            "--labels one two --weights 0.5 0.5 --seed 12345"
        )
        self.launch(command_line)
        assert os.path.isfile(".outdir/test.dat")
        combined = read(".outdir/test.dat").samples_dict
        one = read(".outdir/example.json").samples_dict
        two = read(".outdir/example2.h5").samples_dict
        nsamples = combined.number_of_samples
        half = int(nsamples / 2.)
        for param in combined.keys():
            assert all(ss in one[param] for ss in combined[param][:half])
            assert all(ss in two[param] for ss in combined[param][half:])

    def test_combine_metafile_failures(self):
        """Test that errors are raised when incorrect labels are passed when "
        trying to combine posteriors from a single metafile and when trying
        to combine posteriors from multiple metafiles
        """
        command_line = (
            "summarycombine --samples .outdir/example.json .outdir/example2.h5 "
            ".outdir/example3.dat --labels one two three --webdir .outdir "
            "--no_conversion"
        )
        self.launch(command_line)
        with pytest.raises(Exception):
            command_line = (
                "summarycombine_posteriors --outdir .outdir --filename test.dat "
                "--file_format dat --samples .outdir/samples/posterior_samples.h5 "
                "--labels one four --weights 0.5 0.5 --seed 12345"
            )
            self.launch(command_line)
        with pytest.raises(Exception):
            command_line = (
                "summarycombine_posteriors --outdir .outdir --filename test.dat "
                "--file_format dat --samples .outdir/samples/posterior_samples.h5 "
                ".outdir/samples/posterior_samples.h5 --labels one two "
                "--weights 0.5 0.5 --seed 12345"
            )
            self.launch(command_line)
        with pytest.raises(Exception):
            command_line = (
                "summarycombine_posteriors --outdir .outdir --filename test.dat "
                "--file_format dat --samples .outdir/samples/posterior_samples.h5 "
                ".outdir/example3.dat --labels one two --weights 0.5 0.5 --seed 12345"
            )
            self.launch(command_line)

    def test_combine_metafile(self):
        """Test that the two posteriors are combined when a single metafile
        is provided
        """
        from pesummary.io import read
        command_line = (
            "summarycombine --samples .outdir/example.json .outdir/example2.h5 "
            ".outdir/example3.dat --labels one two three --webdir .outdir "
            "--no_conversion"
        )
        self.launch(command_line)
        command_line = (
            "summarycombine_posteriors --outdir .outdir --filename test.dat "
            "--file_format dat --samples .outdir/samples/posterior_samples.h5 "
            "--labels one two --weights 0.5 0.5 --seed 12345"
        )
        self.launch(command_line)
        assert os.path.isfile(".outdir/test.dat")
        combined = read(".outdir/test.dat").samples_dict
        one = read(".outdir/example.json").samples_dict
        two = read(".outdir/example2.h5").samples_dict
        nsamples = combined.number_of_samples
        half = int(nsamples / 2.)
        for param in combined.keys():
            assert all(ss in one[param] for ss in combined[param][:half])
            assert all(ss in two[param] for ss in combined[param][half:])

        # test that you add the samples to the original file
        command_line = (
            "summarycombine_posteriors --outdir .outdir --filename test.dat "
            "--file_format dat --samples .outdir/samples/posterior_samples.h5 "
            "--labels one two --weights 0.5 0.5 --seed 12345 --add_to_existing"
        )
        self.launch(command_line)
        assert os.path.isfile(".outdir/posterior_samples_combined.h5")
        combined = read(".outdir/posterior_samples_combined.h5")
        combined_samples = combined.samples_dict
        assert "one_two_combined" in combined.labels
        assert "one_two_combined" in combined_samples.keys()
        combined_samples = combined_samples["one_two_combined"]
        for param in combined_samples.keys():
            assert all(ss in one[param] for ss in combined_samples[param][:half])
            assert all(ss in two[param] for ss in combined_samples[param][half:])
        # check that summarypages works fine on output
        command_line = (
            "summarypages --webdir .outdir/combined --disable_expert "
            " --no_conversion --samples .outdir/posterior_samples_combined.h5 "
            "--disable_corner --disable_interactive --gw"
        )
        self.launch(command_line)
        assert os.path.isfile(".outdir/combined/samples/posterior_samples.h5")
        output = read(".outdir/combined/samples/posterior_samples.h5")
        assert "one_two_combined" in output.labels


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

    def test_descriptions(self):
        """Test that the descriptions are correctly replaced in the meta file
        """
        import json
        import h5py

        command_line = (
            'summarymodify --webdir .outdir --samples .outdir/test.h5 '
            '--descriptions replace:TestingSummarymodify'
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        original_data = h5py.File(".outdir/test.h5", "r")
        data = h5py.File(".outdir/test.h5", "r")
        if "description" in original_data["replace"].keys():
            assert original_data["replace"]["description"][0] != b'TestingSummarymodify'
        assert modified_data["replace"]["description"][0] == b'TestingSummarymodify'
        modified_data.close()
        original_data.close()

        with open(".outdir/descriptions.json", "w") as f:
            json.dump({"replace": "NewDescription"}, f)

        command_line = (
            'summarymodify --webdir .outdir --samples .outdir/test.h5 '
            '--descriptions .outdir/descriptions.json'
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        assert modified_data["replace"]["description"][0] == b'NewDescription'
        modified_data.close()

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
            "--samples .outdir/modified_posterior_samples.h5 --disable_expert"
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
