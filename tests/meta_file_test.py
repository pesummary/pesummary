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
import sys
import socket
import shutil

from pesummary.file import meta_file
from pesummary.command_line import command_line

import h5py
import numpy as np
import math

import pytest
from testfixtures import LogCapture

class TestUtils(object):

    def setup(self):
        directory = './.outdir'
        try:
            os.mkdir(directory) 
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)
        f = h5py.File("./.outdir/add_content.h5", "w")
        group = f.create_group("test")
        group.create_dataset("example", np.array([5]))
        f.close()

    def test_add_new_dataset_to_hdf_file(self):
        content = np.array([10])
        meta_file.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content)
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["new_dataset", "test"]
        assert list(f["test"].keys()) == ["example"]
        assert len(f["new_dataset"]) == 1
        assert f["new_dataset"][0] == 10

    def test_add_new_group_to_hdf_file(self):
        content = np.array([10])
        meta_file.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content, group="test")
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["test"]
        assert sorted(list(f["test"].keys())) == ["example", "new_dataset"]
        assert len(f["test/new_dataset"]) == 1
        assert f["test/new_dataset"][0] == 10

    def test_replace_dataset_to_hdf_file(self):
        content = np.array([0])
        meta_file.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content, group="test")
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["test"]
        assert sorted(list(f["test"].keys())) == ["example", "new_dataset"]
        assert len(f["test/new_dataset"]) == 1
        assert f["test/new_dataset"][0] == 0

    def test_combine_hdf_files(self):
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
