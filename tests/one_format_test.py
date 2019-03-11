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

import deepdish

from pesummary.file import one_format

import pytest


class TestOneFormat(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
             shutil.rmtree("./.outdir")
        os.mkdir("./.outdir")

    def test_keys(self):
        f = h5py.File("./.outdir/test.h5", "w")
        f.create_dataset("x", np.array([1]))
        f.create_dataset("y", np.array([1]))
        f.close()
        keys = one_format.one_format.keys("./.outdir/test.h5")
        assert keys == ["x", "y"]

    def test_approximant(self):
        f = {"posterior": {"waveform_approximant": ["approx1"]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.one_format("./.outdir/test_deepdish.h5", None)
        assert f.approximant == "approx1" 

    def test_lalinference(self):
        f = {"posterior": {"waveform_approximant": ["approx1"]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.one_format("./.outdir/test_deepdish.h5", None)
        assert f.lalinference == False

    def test_bilby(self):
        f = {"posterior": {"waveform_approximant": ["approx1"]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.one_format("./.outdir/test_deepdish.h5", None) 
        assert f.bilby == True

    def test_parameters(self):
        f = {"posterior": {"mass_1": [1], "mass_2": [2]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.one_format("./.outdir/test_deepdish.h5", None)
        assert all(i in f.parameters for i in ["mass_1", "mass_2"])

    def test_samples(self):
        f = {"posterior": {"mass_1": [2., 2.], "mass_2": [2., 2.]}}
        deepdish.io.save("./.outdir/test_deepdish.h5", f)
        f = one_format.one_format("./.outdir/test_deepdish.h5", None)
        assert f.samples == [[2., 2.], [2., 2.]]
         
