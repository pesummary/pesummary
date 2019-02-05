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

import argparse

from pesummary.bin.inputs import command_line, Input, PostProcessing

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


class TestInput(object):

    def setup(self):
        self.parser = command_line()
        self.opts = self.parser.parse_args(["--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir", "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org", "--gracedb", "grace"])
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

    def test_gracedb(self):
        assert self.inputs.gracedb == "grace"

    def test_dump(self):
        assert self.inputs.dump == False

    def test_detectors(self):
        assert self.inputs.detectors == ["H1"]

    def test_labels(self):
        assert self.inputs.labels == ["grace_H1"]

class TestPostProcessing(object):

    def setup(self):
        self.parser = command_line()
        self.opts = self.parser.parse_args(["--approximant", "IMRPhenomPv2",
            "--webdir", "./.outdir", "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org", "--gracedb", "grace"])
        self.inputs = Input(self.opts)
        self.postprocessing = PostProcessing(self.inputs)

    def test_colors(self):
        assert self.postprocessing.colors == ['#a6b3d0', '#baa997', '#FF6347',
            '#FFA500', '#003366']
