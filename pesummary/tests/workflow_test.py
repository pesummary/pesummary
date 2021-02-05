# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil
import glob
import numpy as np

from .base import make_argparse, get_list_of_plots, get_list_of_files
from .base import read_result_file
from pesummary.utils.utils import functions
from pesummary.cli.summarypages import WebpageGeneration
from pesummary.cli.summaryplots import PlotGeneration

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Base(object):
    """Base class to test the full workflow
    """
    def test_single_run(self, extension, bilby=False):
        """Test the full workflow for a single result file case

        Parameters
        ----------
        result_file: str
            path to result file you wish to run with
        """
        opts, inputs = make_argparse(
            gw=False, extension=extension, bilby=bilby)
        func = functions(opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))
        assert all(i == j for i, j in zip(plots, get_list_of_plots(gw=False)))
        assert all(i in plots for i in get_list_of_plots(gw=False))
        assert all(i in get_list_of_plots(gw=False) for i in plots)
        assert all(i == j for i, j in zip(files, get_list_of_files(gw=False)))
        assert all(i in files for i in get_list_of_files(gw=False))
        assert all(i in get_list_of_files(gw=False) for i in files)
        self.check_samples(extension, bilby=bilby)

    def check_samples(self, extension, bilby=False):
        """Check that the samples in the result file are consistent with the
        inputs
        """
        from pesummary.core.file.read import read

        initial_samples = read_result_file(extension=extension, bilby=bilby)
        data = read("./.outdir/samples/posterior_samples.h5")
        samples = data.samples_dict
        label = data.labels[0]
        for param in initial_samples.keys():
            for i, j in zip(initial_samples[param], samples[label][param]):
                assert np.round(i, 8) == np.round(j, 8)
        

class GWBase(Base):
    """Base class to test the full workflow including gw specific options
    """
    def test_single_run(self, extension, bilby=False, lalinference=False):
        """Test the full workflow for a single result file case

        Parameters
        ----------
        result_file: str
            path to result file you wish to run with
        """
        opts, inputs = make_argparse(
            gw=True, extension=extension, bilby=bilby, lalinference=lalinference)
        print(opts)
        func = functions(opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))
        assert all(i == j for i, j in zip(plots, get_list_of_plots(gw=True)))
        assert all(i in plots for i in get_list_of_plots(gw=True))
        assert all(i in get_list_of_plots(gw=True) for i in plots)
        for i, j in zip(files, get_list_of_files(gw=True)):
            print(i, j)
        assert all(i == j for i, j in zip(files, get_list_of_files(gw=True)))
        assert all(i in files for i in get_list_of_files(gw=True))
        assert all(i in get_list_of_files(gw=True) for i in files)
        self.check_samples(extension, bilby=bilby, lalinference=lalinference)

    def check_samples(self, extension, bilby=False, lalinference=False):
        """Check that the samples in the result file are consistent with the
        inputs
        """
        from pesummary.core.file.read import read

        initial_samples = read_result_file(
            extension=extension, bilby=bilby, lalinference=lalinference)
        data = read("./.outdir/samples/posterior_samples.h5")
        samples = data.samples_dict
        label = data.labels[0]
        for param in initial_samples.keys():
            for i, j in zip(initial_samples[param], samples[label][param]):
                assert np.round(i, 8) == np.round(j, 8)


class TestCoreDat(Base):
    """Test the full workflow with a core dat file
    """
    def setup(self):
        """Setup the TestCoreDat class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a core dat result file
        """
        extension = "dat"
        super(TestCoreDat, self).test_single_run(extension)


class TestCoreJson(Base):
    """Test the full workflow with a core json file
    """
    def setup(self):
        """Setup the TestCoreJson class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a core json result file
        """
        extension = "json"
        super(TestCoreJson, self).test_single_run(extension)


class TestCoreHDF5(Base):
    """Test the full workflow with a core hdf5 file
    """
    def setup(self):
        """Setup the TestCoreHDF5 class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a core hdf5 result file
        """
        extension = "h5"
        super(TestCoreHDF5, self).test_single_run(extension)


class TestCoreBilbyJson(Base):
    """Test the full workflow with a core json bilby file
    """
    def setup(self):
        """Setup the TestCoreBilby class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")
        if os.path.isdir(".outdir_pesummary"):
            shutil.rmtree(".outdir_pesummary")

    def test_single_run(self):
        """Test the full workflow with a core bilby result file
        """
        extension = "json"
        super(TestCoreBilbyJson, self).test_single_run(extension, bilby=True)

    def test_double_run(self):
        """Test the full workflow for 2 lalinference result files
        """
        opts, inputs = make_argparse(
            gw=False, extension="json", bilby=True, number=2)
        func = functions(opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))
        assert all(i == j for i, j in zip(plots, get_list_of_plots(
                       gw=False, number=2)))
        assert all(i in plots for i in get_list_of_plots(gw=False, number=2))
        assert all(i in get_list_of_plots(gw=False, number=2) for i in plots)
        assert all(i == j for i, j in zip(files, get_list_of_files(
                       gw=False, number=2)))
        assert all(i in files for i in get_list_of_files(gw=False, number=2))
        assert all(i in get_list_of_files(gw=False, number=2) for i in files)

    def test_existing_run(self):
        """Test the fill workflow for when you add to an existing webpage
        """
        opts, inputs = make_argparse(
            gw=False, extension="json", bilby=True)
        func = functions(opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        opts, inputs = make_argparse(
            gw=False, extension="json", bilby=True, existing=True)
        func = functions(opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))

        assert all(i == j for i, j in zip(plots, get_list_of_plots(
                       gw=False, number=2)))
        assert all(i in plots for i in get_list_of_plots(gw=False, number=2))
        assert all(i in get_list_of_plots(gw=False, number=2) for i in plots)
        assert all(i == j for i, j in zip(files, get_list_of_files(
                       gw=False, number=2)))
        assert all(i in files for i in get_list_of_files(gw=False, number=2))
        assert all(i in get_list_of_files(gw=False, number=2) for i in files)

    def test_pesummary_input(self):
        """Test the full workflow for a pesummary input file
        """
        opts, inputs = make_argparse(
            gw=False, extension="json", bilby=True, number=2)
        func = functions(opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))

        from pesummary.core.command_line import command_line

        parser = command_line()
        default_args = ["--webdir", ".outdir_pesummary",
                        "--samples", ".outdir/samples/posterior_samples.h5",
                        "--disable_expert"]
        opts = parser.parse_args(default_args)
        func = functions(opts)
        inputs = func["input"](opts)
        PlotGeneration(inputs)
        WebpageGeneration(inputs)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots_pesummary = sorted(glob.glob("./.outdir_pesummary/plots/*.png"))
        files_pesummary = sorted(glob.glob("./.outdir_pesummary/html/*.html"))

        assert all(i.split("/")[-1] == j.split("/")[-1] for i, j in zip(
                   plots, plots_pesummary))
        assert all(i.split("/")[-1] == j.split("/")[-1] for i, j in zip(
                   files, files_pesummary))

class TestCoreBilbyHDF5(Base):
    """Test the full workflow with a core hdf5 bilby file
    """
    def setup(self):
        """Setup the TestCoreBilby class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a core bilby result file
        """
        extension = "h5"
        super(TestCoreBilbyHDF5, self).test_single_run(extension, bilby=True)


class TestGWDat(GWBase):
    """Test the full workflow with a gw dat file
    """
    def setup(self):
        """Setup the TestCoreDat class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a gw dat result file
        """
        extension = "dat"
        super(TestGWDat, self).test_single_run(extension)


class TestGWJson(GWBase):
    """Test the full workflow with a json dat file
    """
    def setup(self):
        """Setup the TestGWJson class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a gw json result file
        """
        extension = "json"
        super(TestGWJson, self).test_single_run(extension)


class TestGWBilbyJson(GWBase):
    """Test the full workflow with a gw bilby json file
    """
    def setup(self):
        """Setup the TestGWJson class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a gw bilby json result file
        """
        extension = "json"
        super(TestGWBilbyJson, self).test_single_run(extension, bilby=True)


class TestGWBilbyHDF5(GWBase):
    """Test the full workflow with a gw bilby HDF5 file
    """
    def setup(self):
        """Setup the TestGWJson class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_single_run(self):
        """Test the full workflow with a gw bilby HDF5 result file
        """
        extension = "h5"
        super(TestGWBilbyHDF5, self).test_single_run(extension, bilby=True)


class TestGWLALInference(GWBase):
    """Test the full workflow with a lalinference file
    """
    def setup(self):
        """Setup the TestGWJson class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")
        if os.path.isdir(".outdir_pesummary"):
            shutil.rmtree(".outdir_pesummary")

    def test_single_run(self):
        """Test the full workflow with a lalinference result file
        """
        extension = "hdf5"
        super(TestGWLALInference, self).test_single_run(extension, lalinference=True)

    def test_double_run(self):
        """Test the full workflow for 2 lalinference result files
        """
        opts, inputs = make_argparse(
            gw=True, extension="hdf5", lalinference=True, number=2)
        func = functions(opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))
        assert all(i == j for i, j in zip(plots, get_list_of_plots(
                       gw=True, number=2)))
        assert all(i in plots for i in get_list_of_plots(gw=True, number=2))
        assert all(i in get_list_of_plots(gw=True, number=2) for i in plots)
        for i, j in zip(files, get_list_of_files(
                       gw=True, number=2)):
            print(i, j)
        assert all(i == j for i, j in zip(files, get_list_of_files(
                       gw=True, number=2)))
        assert all(i in files for i in get_list_of_files(gw=True, number=2))
        assert all(i in get_list_of_files(gw=True, number=2) for i in files)

    def test_existing_run(self):
        """Test the fill workflow for when you add to an existing webpage
        """
        opts, inputs = make_argparse(
            gw=True, extension="hdf5", lalinference=True)
        func = functions(opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        opts, inputs = make_argparse(
            gw=True, extension="hdf5", lalinference=True, existing=True)
        func = functions(opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))
        assert all(i == j for i, j in zip(plots, get_list_of_plots(
                       gw=True, number=2)))
        assert all(i in plots for i in get_list_of_plots(gw=True, number=2))
        assert all(i in get_list_of_plots(gw=True, number=2) for i in plots)
        assert all(i == j for i, j in zip(files, get_list_of_files(
                       gw=True, number=2)))
        assert all(i in files for i in get_list_of_files(gw=True, number=2))
        assert all(i in get_list_of_files(gw=True, number=2) for i in files)

    def test_pesummary_input(self):
        """Test the full workflow for a pesummary input file
        """
        opts, inputs = make_argparse(
            gw=True, extension="hdf5", lalinference=True, number=2)
        func = functions(opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots = sorted(glob.glob("./.outdir/plots/*.png"))
        files = sorted(glob.glob("./.outdir/html/*.html"))

        from pesummary.core.command_line import command_line
        from pesummary.gw.command_line import insert_gwspecific_option_group

        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_args = ["--webdir", ".outdir_pesummary",
                        "--samples", ".outdir/samples/posterior_samples.h5",
                        "--gw", "--disable_expert"]
        from pesummary.gw.file.read import read
        f = read(".outdir/samples/posterior_samples.h5")
        opts = parser.parse_args(default_args)
        inputs = func["input"](opts)
        PlotGeneration(inputs, gw=True)
        WebpageGeneration(inputs, gw=True)
        func["MetaFile"](inputs)
        func["FinishingTouches"](inputs)

        plots_pesummary = sorted(glob.glob("./.outdir_pesummary/plots/*.png"))
        files_pesummary = sorted(glob.glob("./.outdir_pesummary/html/*.html"))

        assert all(i.split("/")[-1] == j.split("/")[-1] for i, j in zip(
                   plots, plots_pesummary))
        assert all(i.split("/")[-1] == j.split("/")[-1] for i, j in zip(
                   files, files_pesummary))
