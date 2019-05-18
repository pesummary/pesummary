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

from pesummary.gw.file import meta_file
from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from cli.summaryplots import GWPlotGeneration

import h5py
import numpy as np
import math

import pytest


def test_recursively_save_dictionary_to_hdf5_file():
    if os.path.isdir("./.outdir"):
        shutil.rmtree("./.outdir")
    os.makedirs("./.outdir")

    data = {
               "posterior_samples": {
                   "H1_L1_IMRPhenomPv2": {
                       "parameters": ["mass_1", "mass_2"],
                       "samples": [[10, 2], [50, 5], [100, 90]]
                       },
                   "H1_L1_IMRPhenomP": {
                       "parameters": ["ra", "dec"],
                       "samples": [[0.5, 0.8], [1.2, 0.4], [0.9, 1.5]]
                       },
                   "H1_SEOBNRv4": {
                       "parameters": ["psi", "phi"],
                       "samples": [[1.2, 0.2], [3.14, 0.1], [0.5, 0.3]]
                       }
                   }
               }

    with h5py.File("./.outdir/test.h5") as f:
        meta_file._recursively_save_dictionary_to_hdf5_file(f, data)

    f = h5py.File("./.outdir/test.h5", "r")
    assert sorted(list(f.keys())) == sorted(["posterior_samples"])
    assert sorted(list(f["posterior_samples"].keys())) == sorted(
        ["H1_L1_IMRPhenomPv2", "H1_L1_IMRPhenomP", "H1_SEOBNRv4"]
    )
    assert sorted(
        list(f["posterior_samples/H1_L1_IMRPhenomPv2"].keys())) == sorted(
            ["parameters", "samples"]
    )
    assert f["posterior_samples/H1_L1_IMRPhenomPv2/parameters"][0].decode("utf-8") == "mass_1"
    assert f["posterior_samples/H1_L1_IMRPhenomPv2/parameters"][1].decode("utf-8") == "mass_2"
    assert f["posterior_samples/H1_L1_IMRPhenomP/parameters"][0].decode("utf-8") == "ra"
    assert f["posterior_samples/H1_L1_IMRPhenomP/parameters"][1].decode("utf-8") == "dec"
    assert f["posterior_samples/H1_SEOBNRv4/parameters"][0].decode("utf-8") == "psi"
    assert f["posterior_samples/H1_SEOBNRv4/parameters"][1].decode("utf-8") == "phi"

    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][0],
            [10, 2]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][1],
            [50, 5]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][2],
            [100, 90]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][0],
            [0.5, 0.8]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][1],
            [1.2, 0.4]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][2],
            [0.9, 1.5]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][0],
            [1.2, 0.2]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][1],
            [3.14, 0.1]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][2],
            [0.5, 0.3]
        )
    )
    


class TestMetaFile(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
            shutil.rmtree("./.outdir")
        os.makedirs("./.outdir")
        self.parser = command_line()
        insert_gwspecific_option_group(self.parser)
        self.default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace"]
        self.opts = self.parser.parse_args(self.default_arguments)
        self.inputs = GWInput(self.opts)
        self.metafile = meta_file.GWMetaFile(self.inputs)

    @staticmethod
    def make_existing_file(path):
        parameters = np.array(["mass_1", "mass_2", "luminosity_distance"],
                              dtype="S")
        samples = np.array([[10, 5, 400], [40, 20, 800], [50, 10, 200]])
        injected_samples = np.array([float("nan"), float("nan"), float("nan")])

        f = h5py.File(path + "/posterior_samples.h5", "w")
        posterior_samples = f.create_group("posterior_samples")
        label = posterior_samples.create_group("H1_L1")
        label.create_dataset("parameter_names", data=parameters)
        label.create_dataset("samples", data=samples)
        injected_data = f.create_group("injection_data")
        label = injected_data.create_group("H1_L1")
        label.create_dataset("injection_values", data=injected_samples)
        f.close()
        return path + "/posterior_samples.h5"

    def test_meta_file(self):
        assert self.metafile.meta_file == "./.outdir/samples/posterior_samples.json"
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace",
            "--save_to_hdf5"]
        opts = self.parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        metafile = meta_file.GWMetaFile(inputs)
        assert metafile.meta_file == "./.outdir/samples/posterior_samples.h5"

    def test_convert_to_list(self):
         array = [np.array([1,2,3]), np.array([4,5,6])]
         test_list = meta_file.GWMetaFile.convert_to_list(array)
         assert isinstance(test_list, list)
         assert isinstance(test_list[0], list)
         assert isinstance(test_list[1], list)
         print(test_list[0][0])
         assert all(i == j for i,j in zip(sorted(test_list[0]),sorted([1,2,3])))
         assert all(i == j for i,j in zip(sorted(test_list[1]),sorted([4,5,6])))

    def test_add_to_existing(self):
        if os.path.isdir("./.outdir_addition"):
            shutil.rmtree("./.outdir_addition")
        os.makedirs("./.outdir_addition/samples")
        with open("./.outdir_addition/home.html", "w") as f:
            f.writelines(["test"])
        self.make_existing_file("./.outdir_addition/samples")
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--existing_webdir", "./.outdir_addition",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--gracedb", "grace",
            "--save_to_hdf5"]
        opts = self.parser.parse_args(default_arguments)
        inputs = GWInput(opts)
        metafile = meta_file.GWMetaFile(inputs)
        f = h5py.File(metafile.meta_file, "r")
        assert sorted(list(f.keys())) == ["approximant", "injection_data",
                                          "posterior_samples"]
        for i, j in zip(sorted(["grace_H1", "H1_L1"]),
                        sorted(list(f["posterior_samples"].keys()))):
            assert i in j
        assert list(f["posterior_samples/H1_L1"].keys()) == ["parameter_names", "samples"]

    def test_generate_dat_file(self):
        parser = command_line()
        insert_gwspecific_option_group(parser)
        default_arguments = [
            "--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org",
            "--config", "tests/files/config_lalinference.ini"]
        opts = parser.parse_args(default_arguments) 
        inputs = GWInput(opts)
        metafile = meta_file.GWMetaFile(inputs)
        if os.path.isdir("./.outdir/samples/dat"):
            shutil.rmtree("./.outdir/samples/dat")
        metafile.generate_dat_file()
        assert os.path.isdir("./.outdir/samples/dat")
        assert os.path.isfile(
            "./.outdir/samples/dat/H1/H1_bilby_example.h5_mass_1_samples.dat")
        f = open("./.outdir/samples/dat/H1/H1_bilby_example.h5_mass_1_samples.dat")
        f = f.readlines()
        f = sorted([i.strip().split() for i in f])
        assert f[0] == ["10.0"]
        assert f[1] == ["10.0"]
