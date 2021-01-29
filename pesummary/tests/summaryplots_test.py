# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil
from glob import glob

from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from pesummary.cli.summaryplots import _GWPlotGeneration as GWPlotGeneration
from pesummary.gw.file.meta_file import GWMetaFile
from pesummary.cli.summarypages import _GWWebpageGeneration as GWWebpageGeneration
from .base import make_result_file, get_list_of_plots, data_dir

import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestPlotGeneration(object):

    def setup(self):
        directories = ["./.outdir_bilby", "./.outdir_lalinference",
                       "./.outdir_comparison", "./.outdir_add_to_existing2",
                       ".outdir_comparison_no_comparison",
                       ".outdir_add_to_existing_no_comparison"]
        for i in directories:
            if os.path.isdir(i):
                shutil.rmtree(i)
            os.makedirs(i)

    def test_plot_generation_for_bilby_structure(self):
        with open("./.outdir_bilby/psd.dat", "w") as f:
            f.writelines(["1.00 3.44\n"])
            f.writelines(["100.00 4.00\n"])
            f.writelines(["1000.00 5.00\n"])
            f.writelines(["2000.00 6.00\n"])
        with open("./.outdir_bilby/calibration.dat", "w") as f:
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"])
            f.writelines(["2000.0 2.0 3.0 4.0 5.0 6.0 7.0"])
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", bilby=True, outdir="./.outdir_bilby/"
        )
        os.rename("./.outdir_bilby/test.h5", "./.outdir_bilby/bilby_example.h5")
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_bilby",
            "--samples", "./.outdir_bilby/bilby_example.h5",
            "--config", data_dir + "/config_bilby.ini",
            "--psd", "./.outdir_bilby/psd.dat",
            "--calibration", "./.outdir_bilby/calibration.dat",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_bilby/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", outdir=".outdir_bilby", psd=True,
            calibration=False
        )
        for i, j in zip(expected_plots, plots):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_lalinference_structure(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir="./.outdir_lalinference/"
        )
        os.rename(
            "./.outdir_lalinference/test.hdf5",
            "./.outdir_lalinference/lalinference_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_lalinference",
            "--samples", "./.outdir_lalinference/lalinference_example.h5",
            "--config", data_dir + "/config_lalinference.ini",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_lalinference/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", outdir=".outdir_lalinference"
        )
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_comparison(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir="./.outdir_comparison/"
        )
        os.rename(
            "./.outdir_comparison/test.hdf5",
            "./.outdir_comparison/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True, outdir="./.outdir_comparison/"
        )
        os.rename(
            "./.outdir_comparison/test.h5",
            "./.outdir_comparison/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir_comparison",
            "--samples", "./.outdir_comparison/bilby_example.h5",
            "./.outdir_comparison/lalinference_example.h5",
            "--labels", "H10", "H11", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_comparison/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=".outdir_comparison"
        )
        for i,j in zip(sorted(plots), sorted(expected_plots)):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_add_to_existing(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir="./.outdir_add_to_existing2/"
        )
        os.rename(
            "./.outdir_add_to_existing2/test.hdf5",
            "./.outdir_add_to_existing2/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir="./.outdir_add_to_existing2/"
        )
        os.rename(
            "./.outdir_add_to_existing2/test.h5",
            "./.outdir_add_to_existing2/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_add_to_existing2",
            "--samples", "./.outdir_add_to_existing2/bilby_example.h5",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        webpage = GWWebpageGeneration(inputs)
        webpage.generate_webpages()
        meta_file = GWMetaFile(inputs)
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", "./.outdir_add_to_existing2",
            "--samples", "./.outdir_add_to_existing2/lalinference_example.h5",
            "--labels", "H11", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs) 
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_add_to_existing2/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=".outdir_add_to_existing2"
        )
        assert all(i == j for i, j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_multiple_without_comparison(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir="./.outdir_comparison_no_comparison/"
        )
        os.rename(
            "./.outdir_comparison_no_comparison/test.hdf5",
            "./.outdir_comparison_no_comparison/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir="./.outdir_comparison_no_comparison/"
        )
        os.rename(
            "./.outdir_comparison_no_comparison/test.h5",
            "./.outdir_comparison_no_comparison/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir_comparison_no_comparison",
            "--samples", "./.outdir_comparison_no_comparison/bilby_example.h5",
            "./.outdir_comparison_no_comparison/lalinference_example.h5",
            "--labels", "H10", "H11", "--no_ligo_skymap",
            "--disable_comparison", "--disable_expert"
        ]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_comparison/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=".outdir_comparison_no_comparison",
            comparison=False
        )
        for i,j in zip(sorted(plots), sorted(expected_plots)):
            print(i, j)
        assert all(i == j for i,j in zip(sorted(plots), sorted(expected_plots)))

    def test_plot_generation_for_add_to_existing_without_comparison(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        make_result_file(
            gw=True, extension="hdf5", lalinference=True,
            outdir="./.outdir_add_to_existing_no_comparison/"
        )
        os.rename(
            "./.outdir_add_to_existing_no_comparison/test.hdf5",
            "./.outdir_add_to_existing_no_comparison/lalinference_example.h5"
        )
        make_result_file(
            gw=True, extension="hdf5", bilby=True,
            outdir="./.outdir_add_to_existing_no_comparison/"
        )
        os.rename(
            "./.outdir_add_to_existing_no_comparison/test.h5",
            "./.outdir_add_to_existing_no_comparison/bilby_example.h5"
        )
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_add_to_existing_no_comparison",
            "--samples", "./.outdir_add_to_existing_no_comparison/bilby_example.h5",
            "--labels", "H10", "--no_ligo_skymap", "--disable_expert"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        webpage = GWWebpageGeneration(inputs)
        webpage.generate_webpages()
        meta_file = GWMetaFile(inputs)
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", "./.outdir_add_to_existing_no_comparison",
            "--samples", "./.outdir_add_to_existing_no_comparison/lalinference_example.h5",
            "--labels", "H11", "--no_ligo_skymap",
            "--disable_comparison", "--disable_expert"
        ]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWPlotGeneration(inputs)
        webpage.generate_plots()
        plots = sorted(glob("./.outdir_add_to_existing_no_comparison/plots/*.png"))
        expected_plots = get_list_of_plots(
            gw=True, label="H1", number=2, outdir=".outdir_add_to_existing_no_comparison",
            comparison=False
        )
        assert all(i == j for i, j in zip(sorted(plots), sorted(expected_plots)))
