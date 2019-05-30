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
                       "./.outdir_addition", "./.outdir_cbc_copy"]
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
            "--config", "./tests/files/config_bilby.ini",
            "--label", "test"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_bilby/html/*"))
        expected_html = ['./.outdir_bilby/html/error.html',
                         './.outdir_bilby/html/test_test.html',
                         './.outdir_bilby/html/test_test_H1_optimal_snr.html',
                         './.outdir_bilby/html/test_test_config.html',
                         './.outdir_bilby/html/test_test_corner.html',
                         './.outdir_bilby/html/test_test_log_likelihood.html',
                         './.outdir_bilby/html/test_test_mass_1.html',
                         './.outdir_bilby/html/test_test_multiple.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html)))

    def test_webpage_generation_for_lalinference_structure(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_lalinference",
            "--samples", "./tests/files/lalinference_example.h5",
            "--config", "./tests/files/config_lalinference.ini",
            "--label", "test"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_lalinference/html/*"))
        expected_html = ['./.outdir_lalinference/html/test_test.html',
                         './.outdir_lalinference/html/test_test_H1_optimal_snr.html',
                         './.outdir_lalinference/html/test_test_config.html',
                         './.outdir_lalinference/html/test_test_corner.html',
                         './.outdir_lalinference/html/test_test_log_likelihood.html',
                         './.outdir_lalinference/html/test_test_mass_1.html',
                         './.outdir_lalinference/html/test_test_multiple.html',
                         './.outdir_lalinference/html/error.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html)))

    def test_webpage_generation_for_comparison(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir_comparison",
            "--samples", "./tests/files/bilby_example.h5",
            "./tests/files/lalinference_example.h5",
            "--label", "test1", "test2"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html = sorted(glob("./.outdir_comparison/html/*"))
        print(html)
        expected_html = ['./.outdir_comparison/html/Comparison.html',
                         './.outdir_comparison/html/Comparison_H1_optimal_snr.html',
                         './.outdir_comparison/html/Comparison_log_likelihood.html',
                         './.outdir_comparison/html/Comparison_mass_1.html',
                         './.outdir_comparison/html/Comparison_multiple.html',
                         './.outdir_comparison/html/error.html',
                         './.outdir_comparison/html/test1_test1.html',
                         './.outdir_comparison/html/test1_test1_H1_optimal_snr.html',
                         './.outdir_comparison/html/test1_test1_config.html',
                         './.outdir_comparison/html/test1_test1_corner.html',
                         './.outdir_comparison/html/test1_test1_log_likelihood.html',
                         './.outdir_comparison/html/test1_test1_mass_1.html',
                         './.outdir_comparison/html/test1_test1_multiple.html',
                         './.outdir_comparison/html/test2_test2.html',
                         './.outdir_comparison/html/test2_test2_H1_optimal_snr.html',
                         './.outdir_comparison/html/test2_test2_config.html',
                         './.outdir_comparison/html/test2_test2_corner.html',
                         './.outdir_comparison/html/test2_test2_log_likelihood.html',
                         './.outdir_comparison/html/test2_test2_mass_1.html',
                         './.outdir_comparison/html/test2_test2_multiple.html']
        assert all(i == j for i,j in zip(sorted(html), sorted(expected_html)))

    def test_webpage_generation_for_add_to_existing(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_addition",
            "--samples", "./tests/files/bilby_example.h5",
            "--labels", "test"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        GWMetaFile(inputs)
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomP",
            "--existing_webdir", "./.outdir_addition",
            "--samples", "./tests/files/lalinference_example.h5",
            "--labels", "test2"]
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
                         './.outdir_addition/html/error.html',
                         './.outdir_addition/html/test2_test2.html',
                         './.outdir_addition/html/test2_test2_H1_optimal_snr.html',
                         './.outdir_addition/html/test2_test2_config.html',
                         './.outdir_addition/html/test2_test2_corner.html',
                         './.outdir_addition/html/test2_test2_log_likelihood.html',
                         './.outdir_addition/html/test2_test2_mass_1.html',
                         './.outdir_addition/html/test2_test2_multiple.html',
                         './.outdir_addition/html/test_test.html',
                         './.outdir_addition/html/test_test_H1_optimal_snr.html',
                         './.outdir_addition/html/test_test_config.html',
                         './.outdir_addition/html/test_test_corner.html',
                         './.outdir_addition/html/test_test_log_likelihood.html',
                         './.outdir_addition/html/test_test_mass_1.html',
                         './.outdir_addition/html/test_test_multiple.html']
        assert all(i == j for i,j in zip(sorted(html), sorted(expected_html)))

    def test_webpage_generation_for_full_cbc(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir_cbc",
            "--samples", "./tests/files/GW150914_result.h5",
            "--labels", "test"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        metafile = GWMetaFile(inputs)
        html = glob("./.outdir_cbc/html/*")
        expected_html = ['./.outdir_cbc/html/test_test_mass_2.html',
                         './.outdir_cbc/html/test_test_ra.html',
                         './.outdir_cbc/html/test_test_total_mass_source.html',
                         './.outdir_cbc/html/test_test.html',
                         './.outdir_cbc/html/test_test_a_2.html',
                         './.outdir_cbc/html/test_test_psi.html',
                         './.outdir_cbc/html/test_test_total_mass.html',
                         './.outdir_cbc/html/test_test_corner.html',
                         './.outdir_cbc/html/test_test_cos_tilt_2.html',
                         './.outdir_cbc/html/test_test_log_likelihood.html',
                         './.outdir_cbc/html/test_test_phi_12.html',
                         './.outdir_cbc/html/test_test_redshift.html',
                         './.outdir_cbc/html/test_test_iota.html',
                         './.outdir_cbc/html/test_test_mass_ratio.html',
                         './.outdir_cbc/html/test_test_chirp_mass_source.html',
                         './.outdir_cbc/html/test_test_tilt_2.html',
                         './.outdir_cbc/html/test_test_geocent_time.html',
                         './.outdir_cbc/html/test_test_tilt_1.html',
                         './.outdir_cbc/html/test_test_mass_2_source.html',
                         './.outdir_cbc/html/test_test_cos_tilt_1.html',
                         './.outdir_cbc/html/test_test_symmetric_mass_ratio.html',
                         './.outdir_cbc/html/test_test_luminosity_distance.html',
                         './.outdir_cbc/html/test_test_phi_jl.html',
                         './.outdir_cbc/html/test_test_config.html',
                         './.outdir_cbc/html/test_test_multiple.html',
                         './.outdir_cbc/html/test_test_dec.html',
                         './.outdir_cbc/html/test_test_mass_1_source.html',
                         './.outdir_cbc/html/test_test_a_1.html',
                         './.outdir_cbc/html/test_test_comoving_distance.html',
                         './.outdir_cbc/html/error.html',
                         './.outdir_cbc/html/test_test_chirp_mass.html',
                         './.outdir_cbc/html/test_test_phase.html',
                         './.outdir_cbc/html/test_test_mass_1.html']
        assert all(i == j for i,j in zip(sorted(expected_html), sorted(html)))
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./.outdir_cbc/samples/posterior_samples.json",
            "--labels", "test"]
        opts = parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        webpage = GWWebpageGeneration(inputs)
        html_copy = glob("./.outdir_cbc_copy/html/*")
        assert all(i == j for i,j in zip(sorted(html_copy), sorted(html)))
