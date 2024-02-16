# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil
from glob import glob

from pesummary.gw.cli.parser import ArgumentParser
from pesummary.gw.cli.inputs import PlottingInput, WebpagePlusPlottingPlusMetaFileInput
from pesummary.cli.summaryplots import _GWPlotGeneration as GWPlotGeneration
from pesummary.gw.file.meta_file import GWMetaFile
from pesummary.cli.summarypages import _GWWebpageGeneration as GWWebpageGeneration
from .base import make_result_file, get_list_of_plots, data_dir

import pytest
import tempfile
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestPlotGeneration(object):

    def setup_method(self):
        tmpdir = Path(tempfile.TemporaryDirectory(prefix=".", dir=".").name).name
        os.makedirs(tmpdir)
        self.dir = tmpdir

    def teardown_method(self):
        """Remove the files created from this class
        """
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)

    def test_plot_generation_for_bilby_structure(self):
        with open(f"{self.dir}/psd.dat", "w") as f:
            f.writelines(["1.00 3.44\n"])
            f.writelines(["100.00 4.00\n"])
            f.writelines(["1000.00 5.00\n"])
            f.writelines(["2000.00 6.00\n"])
        with open(f"{self.dir}/calibration.dat", "w") as f:
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"])
            f.writelines(["2000.0 2.0 3.0 4.0 5.0 6.0 7.0"])
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", bilby=True, outdir=self.dir,
            n_samples=10
        )
        os.rename(f"{self.dir}/test.h5", f"{self.dir}/bilby_example.h5")
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/bilby_example.h5",
            "--config", data_dir + "/config_bilby.ini",
            "--psd", f"{self.dir}/psd.dat",
            "--calibration", f"{self.dir}/calibration.dat",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", outdir=self.dir, psd=True,
            calibration=False
        )
        for i, j in zip(expected_plots, plots):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_lalinference_structure(self):
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.hdf5", f"{self.dir}/lalinference_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/lalinference_example.h5",
            "--config", data_dir + "/config_lalinference.ini",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", outdir=self.dir,
        )
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_comparison(self):
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.hdf5", f"{self.dir}/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True, outdir=self.dir,
            n_samples=10
        )
        os.rename(
            f"{self.dir}/test.h5", f"{self.dir}/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/bilby_example.h5",
            f"{self.dir}/lalinference_example.h5",
            "--labels", "H10", "H11", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=self.dir
        )
        for i,j in zip(sorted(plots), sorted(expected_plots)):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_add_to_existing(self):
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.hdf5", f"{self.dir}/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.h5", f"{self.dir}/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/bilby_example.h5",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = WebpagePlusPlottingPlusMetaFileInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        webpage = GWWebpageGeneration(inputs)
        webpage.generate_webpages()
        meta_file = GWMetaFile(inputs)
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", self.dir,
            "--samples", f"{self.dir}/lalinference_example.h5",
            "--labels", "H11", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs) 
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=self.dir
        )
        assert all(i == j for i, j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_multiple_without_comparison(self):
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.hdf5", f"{self.dir}/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.h5", f"{self.dir}/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/bilby_example.h5",
            f"{self.dir}/lalinference_example.h5",
            "--labels", "H10", "H11", "--no_ligo_skymap",
            "--disable_comparison", "--disable_expert"
        ]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=self.dir,
            comparison=False
        )
        for i,j in zip(sorted(plots), sorted(expected_plots)):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_add_to_existing_without_comparison(self):
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.hdf5", f"{self.dir}/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir=self.dir, n_samples=10
        )
        os.rename(
            f"{self.dir}/test.h5", f"{self.dir}/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", self.dir,
            "--samples", f"{self.dir}/bilby_example.h5",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = WebpagePlusPlottingPlusMetaFileInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        webpage = GWWebpageGeneration(inputs)
        webpage.generate_webpages()
        meta_file = GWMetaFile(inputs)
        parser = ArgumentParser()
        parser.add_all_known_options_to_parser()
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", self.dir,
            "--samples", f"{self.dir}/lalinference_example.h5",
            "--labels", "H11", "--no_ligo_skymap",
            "--disable_comparison", "--disable_expert"
        ]
        opts = parser.parse_args(default_arguments)
        inputs = PlottingInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob(f"{self.dir}/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=self.dir,
            comparison=False
        )
        assert all(i == j for i, j in zip(sorted(plots), sorted(expected_plots)))
