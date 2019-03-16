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

from pesummary.file import meta_file
from pesummary.command_line import command_line
from pesummary.inputs import Input

import h5py
import numpy as np
import math

import pytest


def test_make_group_in_hf5_file():
    if os.path.isdir("./.outdir"):
        shutil.rmtree("./.outdir")
    os.makedirs("./.outdir")
    f = h5py.File("./.outdir/test.h5", "w")
    f.create_group("test")
    f.close()
    f = h5py.File("./.outdir/test.h5", "r")
    assert list(f.keys()) == ["test"]
    f.close()
    meta_file.make_group_in_hf5_file("./.outdir/test.h5", "new_group")
    f = h5py.File("./.outdir/test.h5", "r")
    assert sorted(list(f.keys())) == ["new_group", "test"]
    f.close()
    meta_file.make_group_in_hf5_file("./.outdir/test.h5", "test/new_subgroup")
    f = h5py.File("./.outdir/test.h5", "r")
    assert sorted(list(f["test"].keys())) == ["new_subgroup"]
    f.close()


def test_combine_hdf_files():
    f = h5py.File("./.outdir/combine_hdf_files.h5", "w")
    posterior_samples = f.create_group("posterior_samples")
    label = posterior_samples.create_group("label")
    group = label.create_group("approx1")
    parameters = np.array(["m1"], dtype="S")
    samples = np.array([[1], [2]])
    approximant = np.array(["approx1"], dtype="S")
    injection_data = np.array([float("nan")])
    group.create_dataset("parameter_names", data=parameters)
    group.create_dataset("samples", data=samples)
    group.create_dataset("injection_parameters", data=parameters)
    group.create_dataset("injection_data", data=injection_data)

    g = h5py.File("./.outdir/combine_hdf_files_new.h5", "w")
    parameters = np.array(["m1"], dtype="S")
    samples = np.array([[1], [2]])
    approximant = np.array(["approx2"], dtype="S")
    injection_data = np.array([float("nan")])
    posterior_samples = g.create_group("posterior_samples")
    label = posterior_samples.create_group("label")
    group = label.create_group("approx2")
    group.create_dataset("parameter_names", data=parameters)
    group.create_dataset("samples", data=samples)
    group.create_dataset("injection_parameters", data=parameters)
    group.create_dataset("injection_data", data=injection_data)

    meta_file.combine_hdf_files("./.outdir/combine_hdf_files.h5",
                                "./.outdir/combine_hdf_files_new.h5")
    f = h5py.File("./.outdir/combine_hdf_files.h5")
    path = "posterior_samples/label"
    assert sorted(list(f[path].keys())) == ["approx1", "approx2"]
    assert sorted(list(f[path+"/approx1"].keys())) == ["injection_data", 
        "injection_parameters", "parameter_names", "samples"]
    assert [i for i in f[path+"/approx1"]["samples"]] == [[1], [2]]
    assert [i for i in f[path+"/approx1"]["parameter_names"]] == [b"m1"]
    assert [i for i in f[path+"/approx1"]["injection_parameters"]] == [b"m1"]
    assert math.isnan(f[path+"/approx1"]["injection_data"][0])
    assert sorted(list(f[path+"/approx1"].keys())) == ["injection_data",
        "injection_parameters", "parameter_names", "samples"]
    assert [i for i in f[path+"/approx2"]["samples"]] == [[1], [2]]
    assert [i for i in f[path+"/approx2"]["parameter_names"]] == [b"m1"]
    assert [i for i in f[path+"/approx2"]["injection_parameters"]] == [b"m1"]
    assert math.isnan(f["posterior_samples/label/approx2"]["injection_data"][0])


class TestMetaFile(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
            shutil.rmtree("./.outdir")
        os.makedirs("./.outdir")
        self.parser = command_line()
        self.default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        self.opts = self.parser.parse_args(self.default_arguments)
        self.inputs = Input(self.opts)
        self.metafile = meta_file.MetaFile(self.inputs)

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

    def test_meta_file(self):
        assert self.metafile.meta_file == "./.outdir/samples/posterior_samples.h5"

    def test_get_keys_from_hdf5_file(self):
        f = h5py.File("./.outdir/testing.h5", "w")
        test_level = f.create_group("test")
        test_level.create_group("sublevel")
        f.close()
        assert self.metafile.get_keys_from_hdf5_file(
            "./.outdir/testing.h5") == ["test"]
        assert self.metafile.get_keys_from_hdf5_file(
            "./.outdir/testing.h5", level="test") == ["sublevel"]
        with pytest.raises(Exception) as info:
            self.metafile.get_keys_from_hdf5_file(
                "./.outdir/testing.h5", level="non_existant")

    def test_labels_and_approximants_to_include(self):
        assert self.metafile.labels_and_approximants_to_include() == (
            ['grace_H1'], ['IMRPhenomPv2'])
        path = self.make_existing_file("./.outdir/samples")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = self.parser.parse_args(self.default_arguments)
        inputs = Input(self.opts)
        metafile = meta_file.MetaFile(self.inputs)
        assert metafile.labels_and_approximants_to_include() == (
            ['grace_H1'], ['IMRPhenomPv2'])

    def test_generate_meta_file_from_scratch(self):
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        self.metafile._generate_meta_file_from_scratch()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2", "IMRPhenomP",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "./tests/files/lalinference_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        metafile._generate_meta_file_from_scratch()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        f = h5py.File("./.outdir/samples/posterior_samples.h5", "r")
        assert list(f["posterior_samples"].keys()) == ["grace_H1"]
        assert sorted(
            list(f["posterior_samples/grace_H1"].keys())) == sorted([
                "IMRPhenomPv2", "IMRPhenomP"])
        f.close()

    def test_add_to_existing_meta_file(self):
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        with open("./.outdir/home.html", "w") as f:
            f.writelines(["test"])
        opts = parser.parse_args(default_arguments)
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        path = self.make_existing_file("./.outdir/samples")
        metafile._add_to_existing_meta_file()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        f = h5py.File("./.outdir/samples/posterior_samples.h5", "r")
        assert sorted(
            list(f["posterior_samples"].keys())) == sorted(
                ["H1_L1", "grace_H1_0"])
        assert list(f["posterior_samples/grace_H1_0"].keys()) == ["IMRPhenomPv2"]
        assert list(f["posterior_samples/H1_L1"].keys()) == ["IMRPhenomPv2"]
        f.close()

    def test_add_psds_to_meta_file(self):
        with open("./.outdir/psd.dat", "w") as f:
            f.writelines(["1.00 3.44"])
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--psd", "./.outdir/psd.dat"]
        opts = parser.parse_args(default_arguments) 
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        metafile._add_psds_to_meta_file()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        f = h5py.File("./.outdir/samples/posterior_samples.h5", "r")
        assert list(f.keys()) == ["psds"]
        assert list(f["psds"].keys()) == ["H1"]
        assert list(f["psds/H1"].keys()) == ["IMRPhenomPv2"]
        x = f["psds/H1/IMRPhenomPv2/psd.dat"][0]
        assert np.round(float(x[0]), 2) == 1.00
        assert np.round(float(x[1]), 2) == 3.44
        f.close()

    def test_add_calibration_to_meta_file(self):
        with open("./.outdir/calibration.dat", "w") as f:
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"])
            f.writelines(["1.0 2.0 3.0 4.0 5.0 6.0 7.0"])
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--calibration", "./.outdir/calibration.dat"]
        opts = parser.parse_args(default_arguments) 
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        metafile._add_calibration_to_meta_file()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        f = h5py.File("./.outdir/samples/posterior_samples.h5", "r")
        assert list(f.keys()) == ["calibration"]
        assert list(f["calibration"].keys()) == ["H1"]
        assert list(f["calibration/H1"].keys()) == ["IMRPhenomPv2"]
        x = f["calibration/H1/IMRPhenomPv2/calibration.dat"][0]
        assert np.round(float(x[0]), 2) == 1.00
        assert np.round(float(x[1]), 2) == 2.00
        f.close()

    def test_add_config_to_meta_file(self):
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--config", "tests/files/config_lalinference.ini"]
        opts = parser.parse_args(default_arguments) 
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isfile("./.outdir/samples/posterior_samples.h5"):
            os.remove("./.outdir/samples/posterior_samples.h5")
        metafile._add_config_to_meta_file()
        assert os.path.isfile("./.outdir/samples/posterior_samples.h5")
        f = h5py.File("./.outdir/samples/posterior_samples.h5", "r")
        assert list(f.keys()) == ["config"]
        assert list(f["config"].keys()) == ["H1"]
        assert list(f["config/H1"].keys()) == ["IMRPhenomPv2"]
        expected_dirs = sorted([
            'analysis', 'bayeswave', 'condor', 'data', 'datafind', 'engine',
            'input', 'lalinference', 'ligo-skymap-from-samples', 'ligo-skymap-plot',
            'mpi', 'paths', 'ppanalysis', 'resultspage', 'segfind', 'segments',
            'singularity', 'skyarea'])
        assert all(
            i in list(f["config/H1/IMRPhenomPv2"].keys()) for i in expected_dirs)
        assert f["config/H1/IMRPhenomPv2/engine/approx"][0].decode(
            "utf-8") == "IMRPhenomPv2pseudoFourPN"
        assert f["config/H1/IMRPhenomPv2/engine/no-detector-frame"][0].decode(
            "utf-8") == ""

    def test_generate_dat_file(self):
        parser = command_line()
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--config", "tests/files/config_lalinference.ini"]
        opts = parser.parse_args(default_arguments) 
        inputs = Input(opts)
        metafile = meta_file.MetaFile(inputs)
        if os.path.isdir("./.outdir/samples/dat"):
            shutil.rmtree("./.outdir/samples/dat")
        metafile.generate_dat_file()
        assert os.path.isdir("./.outdir/samples/dat")
        assert os.path.isfile(
            "./.outdir/samples/dat/H1_IMRPhenomPv2/H1_IMRPhenomPv2_mass_1_samples.dat")
        f = open("./.outdir/samples/dat/H1_IMRPhenomPv2/H1_IMRPhenomPv2_mass_1_samples.dat")
        f = f.readlines()
        f = sorted([i.strip().split() for i in f])
        assert f[0] == ["0.0"]
        assert f[1] == ["10.0"]
