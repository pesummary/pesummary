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
import socket
import shutil

import numpy as np
import h5py
import json

import deepdish

from pesummary.file import one_format

import pytest

dictionary = {
    "level1": {
        "level2": {
            "level3": [1,2,3],
            "level3a": [1,2,3]
        },
        "level2a": {
            "level3b": [1,2,3],
            "level3c": [1,2,3]
        }
    },
    "level1a": {
        "level2b": [1,2,3]
    }
}  


def test_paths_to_key():
    path, = one_format.paths_to_key("level2", dictionary)
    assert path == ["level1", "level2"]
    path, = one_format.paths_to_key("level3c", dictionary)
    assert path == ["level1", "level2a", "level3c"]
    path, = one_format.paths_to_key("level3", dictionary)
    assert path == ["level1", "level2", "level3"]

def test_load_recusively():
    my_dict, = one_format.load_recusively("level1/level2", dictionary)
    assert sorted(my_dict) == sorted({"level3": [1,2,3], "level3a": [1,2,3]})
    #my_dict, = one_format.load_recusively("level1a/level2b", dictionary)
    #assert sorted(my_dict) == [1,2,3]
    my_dict, = one_format.load_recusively("level1/level2a/level3c", dictionary)
    assert sorted(my_dict) == [1,2,3]
    my_dict, = one_format.load_recusively("level1a", dictionary)
    assert sorted(my_dict) == sorted({"level2b": [1,2,3]})


class TestOneFormat(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
             shutil.rmtree("./.outdir")
        os.mkdir("./.outdir")

    def test_check_definition_of_inclination(self):
        parameters = ["mass_1", "tilt1", "tilt2", "a1", "a2",
                      "inclination"]
        parameters = one_format.OneFormat._check_definition_of_inclination(
            parameters)
        assert parameters[5] == "theta_jn"
        parameters = ["mass_1", "tilt1", "tilt2", "a1", "a2",
                      "theta_jn"]
        parameters = one_format.OneFormat._check_definition_of_inclination(
            parameters)
        assert parameters[5] == "theta_jn"
        parameters = ["mass_1", "a1x", "a1y", "a1z", "a2x", "a2y", "a2z",
                      "inclination"]
        parameters = one_format.OneFormat._check_definition_of_inclination(
            parameters)
        assert parameters[7] == "iota"

    def test_extension(self):
        f = {"posterior": {"waveform_approximant": ["approx1"], "mass_1": [10]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        fil = one_format.OneFormat("./.outdir/test_deepdish.h5")
        assert fil.extension == "h5"

        with open("./.outdir/test_json.json", "w") as g:
            json.dump(f, g)
        fil = one_format.OneFormat("./.outdir/test_json.json")
        assert fil.extension == "json" 

    def test_lalinference(self):
        f = {"posterior": {"waveform_approximant": ["approx1"]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.OneFormat("./.outdir/test_deepdish.h5", None)
        assert f.lalinference_hdf5_format == False

    def test_bilby(self):
        f = {"posterior": {"waveform_approximant": ["approx1"]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.OneFormat("./.outdir/test_deepdish.h5", None) 
        assert f.bilby_hdf5_format == True

    def test_approximant(self):
        f = {"posterior": {"waveform_approximant": ["approx1"], "mass_1": [10]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.OneFormat("./.outdir/test_deepdish.h5", None)
        assert f.approximant == "approx1"

    def test_parameters(self):
        f = {"posterior": {"mass_1": [1], "mass_2": [2]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.OneFormat("./.outdir/test_deepdish.h5", None)
        print(f.parameters)
        assert all(i in f.parameters for i in ["mass_1", "mass_2"])

    def test_samples(self):
        f = {"posterior": {"mass_1": [2., 2.], "mass_2": [2., 2.]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.OneFormat("./.outdir/test_deepdish.h5", None)
        assert f.samples == [[2., 2.], [2., 2.]]

    def test_load_in_json_file(self):
        f = {"posterior": {"mass_1": [20], "mass_2": [10]}}
        with open("./.outdir/test_json.json", "w") as g:
            json.dump(f, g)
        fil = one_format.OneFormat("./.outdir/test_json.json")
        assert sorted(fil.parameters) == ["mass_1", "mass_2"]

    def test_load_in_lalinference_hdf5_file(self):
        fil = one_format.OneFormat("tests/files/lalinference_example.h5")
        assert sorted(fil.parameters) == [
            'H1_optimal_snr', 'log_likelihood', 'mass_1']

    def test_load_in_bilby_hdf5_file(self):
        fil = one_format.OneFormat("tests/files/bilby_example.h5")
        assert sorted(fil.parameters) == [
            "H1_optimal_snr", "log_likelihood", "mass_1"]

    def test_load_in_dat_file(self):
        with open("./.outdir/test_dat.dat", "w") as f:
            header = ["H1_optimal_snr m1 m2\n"]
            samples = [["1.0 2.0 3.0\n"], ["1.0 2.0 3.0\n"]]
            f.writelines(header)
            f.writelines(samples[0])
            f.writelines(samples[1])
        fil = one_format.OneFormat("./.outdir/test_dat.dat")
        assert sorted(fil.parameters) == ['H1_optimal_snr', 'mass_1', 'mass_2']
        assert sorted(fil.samples) == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
