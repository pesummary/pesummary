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

from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from cli.summarypages import GWWebpageGeneration
from pesummary.gw.file.meta_file import GWMetaFile
from cli.summaryplots import GWPlotGeneration

import pytest


class TestWebpageGeneration(object):

    def setup(self):
        directories = ["./.outdir_cbc", "./.outdir_bilby",
                       "./.outdir_lalinference", "./.outdir_comparison",
                       "./.outdir_addition"]
        for i in directories:
            if os.path.isdir(i):
                shutil.rmtree(i)
            os.makedirs(i)

    def test_webpage_generation_for_bilby_structure(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_bilby",
            "--samples", "./tests/files/bilby_example.h5",
            "--config", "./tests/files/config_bilby.ini"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_bilby/html/*"))
        expected_html = ['./.outdir_bilby/html/H1_bilby_example.h5_temp.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_config.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_corner.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_log_likelihood.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_mass_1.html',
                         './.outdir_bilby/html/H1_bilby_example.h5_temp_multiple.html',
                         './.outdir_bilby/html/error.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html)))

    def test_webpage_generation_for_lalinference_structure(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_lalinference",
            "--samples", "./tests/files/lalinference_example.h5",
            "--config", "./tests/files/config_lalinference.ini"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_lalinference/html/*"))
        expected_html = ['./.outdir_lalinference/html/H1_lalinference_example.h5_temp.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_config.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_corner.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_log_likelihood.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_mass_1.html',
                         './.outdir_lalinference/html/H1_lalinference_example.h5_temp_multiple.html',
                         './.outdir_lalinference/html/error.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html)))

    def test_webpage_generation_for_comparison(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir_comparison",
            "--samples", "./tests/files/bilby_example.h5",
            "./tests/files/lalinference_example.h5"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_comparison/html/*"))
        expected_html = ['./.outdir_comparison/html/Comparison.html',
                         './.outdir_comparison/html/Comparison_H1_optimal_snr.html',
                         './.outdir_comparison/html/Comparison_log_likelihood.html',
                         './.outdir_comparison/html/Comparison_mass_1.html',
                         './.outdir_comparison/html/Comparison_multiple.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_config.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_corner.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_log_likelihood.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_mass_1.html',
                         './.outdir_comparison/html/H1_0_bilby_example.h5_temp_multiple.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_config.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_corner.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_log_likelihood.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_mass_1.html',
                         './.outdir_comparison/html/H1_1_lalinference_example.h5_temp_multiple.html',
                         './.outdir_comparison/html/error.html']
        assert all(i == j for i,j in zip(sorted(html), sorted(expected_html)))

    def test_webpage_generation_for_add_to_existing(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_addition",
            "--samples", "./tests/files/bilby_example.h5"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        GWMetaFile(inputs)
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", "./.outdir_addition",
            "--samples", "./tests/files/lalinference_example.h5"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        GWPlotGeneration(inputs)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_addition/html/*"))
        expected_html = ['./.outdir_addition/html/Comparison.html',
                         './.outdir_addition/html/Comparison_H1_optimal_snr.html',
                         './.outdir_addition/html/Comparison_log_likelihood.html',
                         './.outdir_addition/html/Comparison_mass_1.html',
                         './.outdir_addition/html/Comparison_multiple.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_config.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_corner.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_log_likelihood.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_mass_1.html',
                         './.outdir_addition/html/H1_0_lalinference_example.h5_temp_multiple.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_H1_optimal_snr.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_config.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_corner.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_log_likelihood.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_mass_1.html',
                         './.outdir_addition/html/H1_bilby_example.h5_temp_multiple.html',
                         './.outdir_addition/html/H1_posterior_samples.json.html',
                         './.outdir_addition/html/H1_posterior_samples.json_H1_optimal_snr.html',
                         './.outdir_addition/html/H1_posterior_samples.json_config.html',
                         './.outdir_addition/html/H1_posterior_samples.json_corner.html',
                         './.outdir_addition/html/H1_posterior_samples.json_log_likelihood.html',
                         './.outdir_addition/html/H1_posterior_samples.json_mass_1.html',
                         './.outdir_addition/html/H1_posterior_samples.json_multiple.html',
                         './.outdir_addition/html/error.html']
        assert all(i == j for i,j in zip(sorted(html), sorted(expected_html)))

    def test_webpage_generation_for_full_cbc(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_cbc",
            "--samples", "./tests/files/GW150914_result.h5"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = glob("./.outdir_cbc/html/*")
        print(html)
        expected_html = ['./.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_geocent_time.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_tilt_1.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_log_likelihood.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_symmetric_mass_ratio.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_luminosity_distance.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_dec.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_a_1.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_comoving_distance.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_ra.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_mass_1.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_mass_ratio.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_config.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_phi_jl.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_multiple.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_phase.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_cos_tilt_2.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_total_mass.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_cos_tilt_1.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_total_mass_source.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_mass_2_source.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_mass_2.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_a_2.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_iota.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_tilt_2.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_psi.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_chirp_mass.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_corner.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_mass_1_source.html',
                         './.outdir_cbc/html/error.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_phi_12.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_redshift.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp_chirp_mass_source.html',
                         './.outdir_cbc/html/1556813622_GW150914_result_GW150914_result.h5_temp.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html))) 
