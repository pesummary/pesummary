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

import os
import shutil
from glob import glob

from pesummary.command_line import command_line
from pesummary.inputs import Input
from pesummary.summaryplots import PlotGeneration

import pytest


class TestPlotGeneration(object):

    def setup(self):
        directories = ["./.outdir_bilby", "./.outdir_lalinference",
                       "./.outdir_comparison"]
        for i in directories:
            if os.path.isdir(i):
                shutil.rmtree(i)
            os.makedirs(i)

    def test_plot_generation_for_bilby_structure(self):
        with open("./.outdir_bilby/psd.dat", "w") as f:
            f.writelines(["1.00 3.44\n"])
            f.writelines(["100.00 4.00"])
        with open("./.outdir_bilby/calibration.dat", "w") as f:
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"])
            f.writelines(["2000.0 2.0 3.0 4.0 5.0 6.0 7.0"])
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_bilby",
            "--samples", "./tests/files/bilby_example.h5",
            "--config", "./tests/files/config_bilby.ini",
            "--psd", "./.outdir_bilby/psd.dat",
            "--calibration", "./.outdir_bilby/calibration.dat"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        webpage = PlotGeneration(inputs)
        plots = sorted(glob("./.outdir_bilby/plots/*"))
        expected_plots = [
            './.outdir_bilby/plots/H1_1d_posterior_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_bilby/plots/H1_1d_posterior_IMRPhenomPv2_log_likelihood.png',
            './.outdir_bilby/plots/H1_1d_posterior_IMRPhenomPv2_mass_1.png',
            './.outdir_bilby/plots/H1_autocorrelation_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_bilby/plots/H1_autocorrelation_IMRPhenomPv2_log_likelihood.png',
            './.outdir_bilby/plots/H1_autocorrelation_IMRPhenomPv2_mass_1.png',
            './.outdir_bilby/plots/H1_cdf_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_bilby/plots/H1_cdf_IMRPhenomPv2_log_likelihood.png',
            './.outdir_bilby/plots/H1_cdf_IMRPhenomPv2_mass_1.png',
            './.outdir_bilby/plots/H1_sample_evolution_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_bilby/plots/H1_sample_evolution_IMRPhenomPv2_log_likelihood.png',
            './.outdir_bilby/plots/H1_sample_evolution_IMRPhenomPv2_mass_1.png',
            './.outdir_bilby/plots/corner',
            './.outdir_bilby/plots/psd_plot.png',
            './.outdir_bilby/plots/calibration_plot.png']
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_lalinference_structure(self):
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_lalinference",
            "--samples", "./tests/files/lalinference_example.h5",
            "--config", "./tests/files/config_lalinference.ini"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        webpage = PlotGeneration(inputs)
        plots = sorted(glob("./.outdir_lalinference/plots/*"))
        expected_plots = [
            './.outdir_lalinference/plots/H1_1d_posterior_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_lalinference/plots/H1_1d_posterior_IMRPhenomPv2_log_likelihood.png',
            './.outdir_lalinference/plots/H1_1d_posterior_IMRPhenomPv2_mass_1.png',
            './.outdir_lalinference/plots/H1_autocorrelation_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_lalinference/plots/H1_autocorrelation_IMRPhenomPv2_log_likelihood.png',
            './.outdir_lalinference/plots/H1_autocorrelation_IMRPhenomPv2_mass_1.png',
            './.outdir_lalinference/plots/H1_cdf_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_lalinference/plots/H1_cdf_IMRPhenomPv2_log_likelihood.png',
            './.outdir_lalinference/plots/H1_cdf_IMRPhenomPv2_mass_1.png',
            './.outdir_lalinference/plots/H1_sample_evolution_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_lalinference/plots/H1_sample_evolution_IMRPhenomPv2_log_likelihood.png',
            './.outdir_lalinference/plots/H1_sample_evolution_IMRPhenomPv2_mass_1.png',
            './.outdir_lalinference/plots/corner']
        assert all(i == j for i,j in zip(sorted(expected_plots), sorted(plots)))

    def test_plot_generation_for_comparison(self):
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir_comparison",
            "--samples", "./tests/files/bilby_example.h5",
            "./tests/files/lalinference_example.h5"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        webpage = PlotGeneration(inputs)
        plots = sorted(glob("./.outdir_comparison/plots/*"))
        print(plots)
        expected_plots = [
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomP_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomP_log_likelihood.png',
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomP_mass_1.png',
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomPv2_log_likelihood.png',
            './.outdir_comparison/plots/H1_1d_posterior_IMRPhenomPv2_mass_1.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomP_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomP_log_likelihood.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomP_mass_1.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomPv2_log_likelihood.png',
            './.outdir_comparison/plots/H1_autocorrelation_IMRPhenomPv2_mass_1.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomP_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomP_log_likelihood.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomP_mass_1.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomPv2_log_likelihood.png',
            './.outdir_comparison/plots/H1_cdf_IMRPhenomPv2_mass_1.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomP_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomP_log_likelihood.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomP_mass_1.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomPv2_H1_optimal_snr.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomPv2_log_likelihood.png',
            './.outdir_comparison/plots/H1_sample_evolution_IMRPhenomPv2_mass_1.png',
            './.outdir_comparison/plots/combined_1d_posterior_H1_optimal_snr.png',
            './.outdir_comparison/plots/combined_1d_posterior_log_likelihood.png',
            './.outdir_comparison/plots/combined_1d_posterior_mass_1.png',
            './.outdir_comparison/plots/combined_cdf_H1_optimal_snr.png',
            './.outdir_comparison/plots/combined_cdf_log_likelihood.png',
            './.outdir_comparison/plots/combined_cdf_mass_1.png',
            './.outdir_comparison/plots/corner']
        assert all(i == j for i,j in zip(sorted(plots), sorted(expected_plots)))
