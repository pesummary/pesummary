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

from pesummary.utils import utils
from pesummary.utils import run_checks
from pesummary.bin.summarypages import command_line

import h5py
import numpy as np

import pytest

class TestUtils(object):

    def setup(self):
        directory = './.outdir'
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

    def test_directory_creation(self):
        directory = './.outdir/test_dir'
        assert os.path.isdir(directory) == False
        utils.make_dir(directory)
        assert os.path.isdir(directory) == True

    def test_url_guess(self):
        host = ["raven", "cit", "ligo-wa", "uwm", "phy.syr.edu", "vulcan",
                "atlas", "iucca"]
        expected = ["https://geo2.arcca.cf.ac.uk/~albert.einstein/test",
                    "https://ldas-jobs.ligo.caltech.edu/~albert.einstein/test",
                    "https://ldas-jobs.ligo-wa.caltech.edu/~albert.einstein/test",
                    "https://ldas-jobs.phys.uwm.edu/~albert.einstein/test",
                    "https://sugar-jobs.phy.syr.edu/~albert.einstein/test",
                    "https://galahad.aei.mpg.de/~albert.einstein/test",
                    "https://atlas1.atlas.aei.uni-hannover.de/~albert.einstein/test",
                    "https://ldas-jobs.gw.iucaa.in/~albert.einstein/test"]
        user = "albert.einstein"
        webdir = '/home/albert.einstein/public_html/test'
        for i,j in zip(host, expected):
            url = utils.guess_url(webdir, i, user)
            assert url == j


class TestChecks(object):

    def setup(self):
        self.parser = command_line()
        directory = './.outdir'

    def test_check_condition(self):
        with pytest.raises(Exception) as info:
            condition = True
            run_checks.check_condition(condition, "error")
        assert str(info.value) == "error"

    def test_no_webdir(self):
        opts = self.parser.parse_args(['--webdir', None])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "Please provide a web directory" in str(info.value)

    def test_make_webdir_if_it_does_not_exist(self):
        assert os.path.isdir("./.outdir/path") == False
        opts = self.parser.parse_args(['--webdir', './.outdir/path',
                                       '--approximant', 'approx',
                                       '--samples', './tests/files/bilby_example.h5'])
        run_checks.run_checks(opts)
        assert os.path.isdir("./.outdir/path") == True

    def test_invalid_existing_directory(self):
        if os.path.isdir("./.existing"):
            shutil.rmtree("./.existing")
        opts = self.parser.parse_args(['--existing_webdir', './.existing'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
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
            run_checks.run_checks(opts)
        assert "Please give the base directory" in str(info.value)

    def test_no_samples(self):
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--samples', None])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "Please provide a results file" in str(info.value)

    def test_no_approximant(self):
        f = h5py.File("./.outdir/test.h5", "w")
        approx = np.array([b"none"], dtype="S")
        f.create_dataset("approximant", data=approx)
        f.close()
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--samples', './.outdir/test.h5'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "Failed to extract the approximant" in str(info.value)

    def test_nsamples_not_equal_to_napproximants(self):
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--samples', './.outdir/test.h5',
                                       '--approximant', 'approx1', 'approx2'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "results files does not match the number of" in str(info.value)

    def test_no_existing_directory(self):
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--samples', './.outdir/test.h5',
                                       '--approximant', 'approx',
                                       '--add_to_existing'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "Please provide a current html page" in str(info.value)

    def test_result_file_does_not_exist(self):
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--baseurl', './.outdir',
                                       '--samples', 'test.h5',
                                       '--approximant', 'approx'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "File test.h5 does not exist" in str(info.value)

    def test_already_used_results_file(self):
        if os.path.isdir("./.outdir/samples"):
            shutil.rmtree("./.outdir/samples")
        os.mkdir("./.outdir/samples")
        shutil.copyfile("./.outdir/test.h5", "./.outdir/samples/approx_test.h5")
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--baseurl', './.outdir',
                                       '--samples', './.outdir/test.h5',
                                       '--approximant', 'approx'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "Have you already generated a summary page" in str(info.value)

    def test_nconfig_not_equal_to_nsamples(self):
        opts = self.parser.parse_args(['--webdir', './.outdir',
                                       '--samples', './.outdir/test.h5',
                                       '--approximant', 'approx1',
                                       '--config', 'one.ini', 'two.ini'])
        with pytest.raises(Exception) as info:
            run_checks.run_checks(opts)
        assert "files match the number of configuration" in str(info.value) 
