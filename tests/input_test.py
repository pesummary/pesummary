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
import glob

import argparse

from pesummary.inputs import Input, PostProcessing
from pesummary.command_line import command_line

import numpy as np
import h5py

import pytest


class TestCommandLine(object):

    def setup(self):
        self.parser = command_line()

    def test_webdir(self):
        assert self.parser.get_default("webdir") == None
        opts = self.parser.parse_args(["--webdir", "test"])
        assert opts.webdir == "test"

    def test_baseurl(self):
        assert self.parser.get_default("baseurl") == None
        opts = self.parser.parse_args(["--baseurl", "test"])
        assert opts.baseurl == "test"

    def test_add_to_existing(self):
        assert self.parser.get_default("add_to_existing") == False
        opts = self.parser.parse_args(["--add_to_existing"])
        assert opts.add_to_existing == True

    def test_approximant(self):
        assert self.parser.get_default("approximant") == None
        opts = self.parser.parse_args(["--approximant", "test"])
        assert opts.approximant == ["test"]

    def test_config(self):
        assert self.parser.get_default("config") == None
        opts = self.parser.parse_args(["--config", "test"])
        assert opts.config == ["test"]

    def test_dump(self):
        assert self.parser.get_default("dump") == False
        opts = self.parser.parse_args(["--dump"])
        assert opts.dump == True

    def test_email(self):
        assert self.parser.get_default("email") == None
        opts = self.parser.parse_args(["--email", "test"])
        assert opts.email == "test"

    def test_existing(self):
        assert self.parser.get_default("existing") == None
        opts = self.parser.parse_args(["--existing_webdir", "test"])
        assert opts.existing == "test"

    def test_gracedb(self):
        assert self.parser.get_default("gracedb") == None
        opts = self.parser.parse_args(["--gracedb", "test"])
        assert opts.gracedb == "test"

    def test_inj_file(self):
        assert self.parser.get_default("inj_file") == None
        opts = self.parser.parse_args(["--inj_file", "test"])
        assert opts.inj_file == ["test"]

    def test_samples(self):
        assert self.parser.get_default("samples") == None
        opts = self.parser.parse_args(["--samples", "test"])
        assert opts.samples == ["test"]

    def test_sensitivity(self):
        assert self.parser.get_default("sensitivity") == False
        opts = self.parser.parse_args(["--sensitivity"])
        assert opts.sensitivity == True

    def test_user(self):
        assert self.parser.get_default("user") == "albert.einstein"
        opts = self.parser.parse_args(["--user", "test"])
        assert opts.user == "test"

    def test_verbose(self):
        opts = self.parser.parse_args(["-v"])
        assert opts.verbose == True


class TestInputExceptions(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
            shutil.rmtree("./.outdir")
        os.mkdir('./.outdir')
        self.parser = command_line()

    def test_no_webdir(self):
        with pytest.raises(Exception) as info:
            opts = self.parser.parse_args(["--webdir", None])
            x = Input(opts)
        assert "Please provide a web directory" in str(info.value)

    def test_make_webdir_if_it_does_not_exist(self):
        assert os.path.isdir("./.outdir/path") == False
        opts = self.parser.parse_args(['--webdir', './.outdir/path',
                                       '--approximant', 'IMRPhenomPv2',
                                       '--samples', "./tests/files/bilby_example.h5"])
        x = Input(opts)
        assert os.path.isdir("./.outdir/path") == True

    def test_invalid_existing_directory(self):
        if os.path.isdir("./.existing"):
            shutil.rmtree("./.existing")
        with pytest.raises(Exception) as info:
            opts = self.parser.parse_args(['--existing_webdir', './.existing'])
            x = Input(opts)
        assert "The directory ./.existing does not exist" in str(info.value)

    def test_not_base_of_existing_directory(self):
        if os.path.isdir("./.existing2"):
            shutil.rmtree("./.existing2")
        if os.path.isdir("./.existing2/samples"):
            shutil.rmtree("./.existing2/samples")
        os.mkdir("./.existing2")
        os.mkdir("./.existing2/samples")
        opts = self.parser.parse_args(['--existing_webdir', './.existing2/samples'])
        with pytest.raises(Exception) as info:
            x = Input(opts)
        assert "Please give the base directory" in str(info.value)

    def test_add_to_existing_and_no_existing_flag(self):
        opts = self.parser.parse_args(["--add_to_existing"])
        with pytest.raises(Exception) as info:
            x = Input(opts)
        assert "pass this path under the --existing_dir" in str(info.value)

    def test_no_samples(self):
        opts = self.parser.parse_args(["--webdir", "./.outdir"])
        with pytest.raises(Exception) as info:
            x = Input(opts)
        assert "Please provide a results file" in str(info.value)

    def test_non_existance_samples(self):
        opts = self.parser.parse_args(["--webdir", "./.outdir",
                                       "--samples", "./.outdir/no_existance"])
        with pytest.raises(Exception) as info:
            x = Input(opts)
        assert "File ./.outdir/no_existance does not exist" in str(info.value)

    def test_no_approximant_in_results_file(self):
        data = np.array([(1.0, 2.0)], dtype=[("m1", "f"), ("m2", "f")])
        f = h5py.File("./.outdir/noapprox.h5", "w")
        posterior_samples = f.create_group("lalinference/lalinference_nest")
        posterior_samples.create_dataset("posterior_samples", data=data)
        opts = self.parser.parse_args(["--webdir", "./.outdir",
                                       "--samples", "./.outdir/noapprox.h5"])
        with pytest.raises(Exception) as info:
            x = Input(opts)

    def test_napproximant_not_equal_to_nsamples(self):
        opts = self.parser.parse_args(["--webdir", "./.outdir",
                                       "--samples", "./tests/files/bilby_example.h5",
                                       "./tests/files/bilby_example.h5",
                                       "--approximant", "IMRPhenomPv2"])
        with pytest.raises(Exception) as info:
            x = Input(opts)
        assert "does not match the number of approximants" in str(info.value)


class TestInput(object):

    def setup(self):
        self.parser = command_line()
        self.default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        self.make_input_object()

    @staticmethod
    def make_existing_file(path):
        parameters = np.array(["mass_1", "mass_2", "luminosity_distance"],
                              dtype="S")
        samples = np.array([[10, 5, 400], [40, 20, 800], [50, 10, 200]])
        injected_samples = np.array([float("nan"), float("nan"), float("nan")])

        f = h5py.File(path + "/posterior_samples.h5", "w")
        posterior_samples = f.create_group("posterior_samples")
        label = posterior_samples.create_group("H1_L1")
        approx = label.create_group("IMRPhenomPv2")
        approx.create_dataset("parameter_names", data=parameters)
        approx.create_dataset("samples", data=samples)
        approx.create_dataset("injected_parameters", data=parameters)
        approx.create_dataset("injected_samples", data=injected_samples)
        f.close()
        return path + "/posterior_samples.h5"

    def add_argument(self, argument):
        if isinstance(argument, list):
            for i in argument:
                self.default_arguments.append(i)
        else:
            self.default_arguments.append(argument)
        self.opts = self.parser.parse_args(self.default_arguments)
        self.inputs = Input(self.opts)

    def replace_existing_argument(self, argument, new_value):
        if argument in self.default_arguments:
            index = self.default_arguments.index(argument)
            arguments = self.default_arguments
            arguments[index+1] = new_value
            self.default_arguments = arguments
        self.make_input_object()

    def make_input_object(self):
        self.opts = self.parser.parse_args(self.default_arguments) 
        self.inputs = Input(self.opts)

    def test_webdir(self):
        assert self.inputs.webdir == "./.outdir"

    def test_samples(self):
        assert self.inputs.result_files == ["./tests/files/bilby_example.h5_temp"]

    def test_approximant(self):
        assert self.inputs.approximant == ["IMRPhenomPv2"]

    def test_existing(self):
        assert self.inputs.existing == None

    def test_baseurl(self):
        assert self.inputs.baseurl == "https://./.outdir"

    def test_inj_file(self):
        assert self.inputs.inj_file == [None]

    def test_config(self):
        assert self.inputs.config == None

    def test_email(self):
        assert self.inputs.email == "albert.einstein@ligo.org"

    def test_add_to_existing(self):
        assert self.inputs.add_to_existing == False

    def test_sensitivity(self):
        assert self.inputs.sensitivity == False

    def test_dump(self):
        assert self.inputs.dump == False
        self.add_argument(["--dump"])
        assert self.inputs.dump == True

    def test_gracedb(self):
        assert self.inputs.gracedb == "grace"

    def test_detectors(self):
        assert self.inputs.detectors == ["H1"]

    def test_labels(self):
        assert self.inputs.labels == ["grace_H1"]
        self.add_argument(["--label", "test"])
        assert self.inputs.labels == ["test"]

    def test_existing_labels(self):
        assert self.inputs.existing_labels == None
        path = self.make_existing_file("./.outdir/samples")
        with open("./.outdir/home.html", "w") as f:
            f.writelines("test")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        assert inputs.existing_labels == ["H1_L1"]

    def test_existing_approximant(self):
        assert self.inputs.existing_approximant == None
        path = self.make_existing_file("./.outdir/samples")
        with open("./.outdir/home.html", "w") as f:
            f.writelines("test")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        assert inputs.existing_approximant == ["IMRPhenomPv2"]

    def test_existing_samples(self):
        assert self.inputs.existing_samples == None
        path = self.make_existing_file("./.outdir/samples")
        with open("./.outdir/home.html", "w") as f:
            f.writelines("test")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        assert all(
            i == j for i,j in zip(inputs.existing_samples[0][0], [10, 5, 400]))
        assert all(
            i == j for i,j in zip(inputs.existing_samples[0][1], [40, 20, 800]))
        assert all(
            i == j for i,j in zip(inputs.existing_samples[0][2], [50, 10, 200]))

    def test_existing_names(self):
        assert self.inputs.existing_names == None
        path = self.make_existing_file("./.outdir/samples")
        with open("./.outdir/home.html", "w") as f:
            f.writelines("test")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        assert inputs.existing_names == ["H1_L1_IMRPhenomPv2"]

    def test_psd(self):
        with open("./.outdir/psd.dat", "w") as f:
            f.writelines(["1.00 3.44"])
        assert self.inputs.psds == None
        self.add_argument(["--psd", "./.outdir/psd.dat"])
        assert self.inputs.psds == ["./.outdir/psd.dat"]

    def test_calibration(self):
        with open("./.outdir/calibration.dat", "w") as f:
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"])
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0"])
        assert self.inputs.calibration == None
        self.add_argument(["--calibration", "./.outdir/calibration.dat"])
        assert all(i == j for i, j in zip(
            self.inputs.calibration[0][0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

    def test_calibration_labels(self):
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5",
            "--calibration", "./.outdir/calibration.dat"])
        inputs = Input(opts)
        postprocessing = PostProcessing(inputs)
        assert postprocessing.calibration_labels == ['calibration.dat']

    def test_IFO_from_file_name(self):
        file_name = "IFO0.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "H1"
        file_name = "IFO1.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "L1"
        file_name = "IFO2.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "V1"

        file_name = "IFO_H.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "H1"
        file_name = "IFO_L.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "L1"
        file_name = "IFO_V.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "V1"
        
        file_name = "example.dat"
        assert PostProcessing._IFO_from_file_name(file_name) == "example.dat"

    def test_make_directories(self):
        assert os.path.isdir("./.outdir/samples/samples") == False
        self.replace_existing_argument("--webdir", "./.outdir/samples")
        self.inputs.make_directories()
        assert os.path.isdir("./.outdir/samples/samples") == True

    def test_copy_files(self):
        if os.path.isdir("./.outdir/samples"):
            shutil.rmtree("./.outdir/samples")
        assert os.path.isfile(
            "./.outdir/samples/js/combine_corner.js") == False
        self.replace_existing_argument("--webdir", "./.outdir/samples")
        self.add_argument(["--config", "tests/files/config_lalinference.ini"])
        self.inputs.copy_files()
        assert os.path.isfile(
            "./.outdir/samples/js/combine_corner.js") == True
        assert os.path.isfile(
            "./.outdir/samples/config/IMRPhenomPv2_config_lalinference.ini") == True

    def test_default_labels(self):
        assert self.inputs._default_labels() == ['grace_H1']
        self.replace_existing_argument("--gracedb", "grace_test")
        assert self.inputs._default_labels() == ['grace_test_H1']


class TestPostProcessing(object):

    def setup(self):
        self.parser = command_line()
        self.opts = self.parser.parse_args(["--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir", "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org", "--gracedb", "grace"])
        self.inputs = Input(self.opts)
        self.postprocessing = PostProcessing(self.inputs)

    def test_coherence_test(self):
        assert self.postprocessing.coherence_test == False
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5"])
        inputs = Input(opts)
        postprocessing = PostProcessing(inputs)
        assert postprocessing.coherence_test == True 

    def test_colors(self):
        assert self.postprocessing.colors == [
            "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CA9161", "#FBAFE4",
            "#949494", "#ECE133", "#56B4E9"]
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5"])
        inputs = Input(opts)
        with pytest.raises(Exception) as info:
            postprocessing = PostProcessing(inputs, colors=["b"])

    def test_injection_data(self):
        assert sorted(list(self.postprocessing.injection_data[0].keys())) == [
            'H1_optimal_snr', 'log_likelihood', 'mass_1']

    def test_maxL_samples(self):
        assert self.postprocessing.maxL_samples[0]["mass_1"] == 20.0
        assert self.postprocessing.maxL_samples[0]["H1_optimal_snr"] == 30.0
        assert self.postprocessing.maxL_samples[0]["log_likelihood"] == 3.0
        assert self.postprocessing.maxL_samples[0]["approximant"] == "IMRPhenomPv2"

    def test_same_parameters(self):
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5"])
        inputs = Input(opts)
        postprocessing = PostProcessing(inputs)
        assert sorted(postprocessing.same_parameters) == [
            'H1_optimal_snr', 'log_likelihood', 'mass_1']

    def test_label_to_prepend_approximant(self):
        assert self.postprocessing.label_to_prepend_approximant == [None]
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5"])
        inputs = Input(opts)
        postprocessing = PostProcessing(inputs)
        assert postprocessing.label_to_prepend_approximant == ['H1_0', 'H1_1']

    def test_psd_labels(self):
        with pytest.raises(Exception) as info:
            self.postprocessing.psd_labels
        parser = command_line()
        opts = parser.parse_args(["--approximant", "IMRPhenomPv2",
            "IMRPhenomPv2", "--webdir", "./.outdir", "--samples",
            "./tests/files/bilby_example.h5", "./tests/files/lalinference_example.h5",
            "--psd", "./.outdir/psd.dat"])
        inputs = Input(opts)
        postprocessing = PostProcessing(inputs)
        assert postprocessing.psd_labels == ['psd.dat']

    def test_grab_frequencies_from_psd_data_file(self):
        assert(self.postprocessing._grab_frequencies_from_psd_data_file(
            "./.outdir/psd.dat")) == [1.0]
        with open("./.outdir/psd_2.dat", "w") as f:
            f.writelines(["1.0 2.0\n", "3.0 4.0"])
        assert(self.postprocessing._grab_frequencies_from_psd_data_file(
            "./.outdir/psd_2.dat")) == [1.0, 3.0]

    def test_grab_strains_from_psd_data_file(self):
        assert(self.postprocessing._grab_strains_from_psd_data_file(
            "./.outdir/psd.dat")) == [3.44]
        with open("./.outdir/psd_2.dat", "w") as f:
            f.writelines(["1.0 2.0\n", "3.0 4.0"])
        assert(self.postprocessing._grab_strains_from_psd_data_file(
            "./.outdir/psd_2.dat")) == [2.0, 4.0]        
