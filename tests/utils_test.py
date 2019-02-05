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

from pesummary.utils import utils
from pesummary.bin.summarypages import command_line

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
 
    def test_check_condition(self):
        with pytest.raises(Exception) as info:
            condition = True
            utils.check_condition(condition, "error")
        assert str(info.value) == "error"

    def test_rename_group_in_hf5_file(self):
        f = h5py.File("./.outdir/rename_group.h5")
        group = f.create_group("group")
        group.create_dataset("example", data=np.array([10]))
        f.close()
        utils.rename_group_or_dataset_in_hf5_file("./.outdir/rename_group.h5",
            group=["group", "replaced"])
        f = h5py.File("./.outdir/rename_group.h5")
        assert list(f.keys()) == ["replaced"]
        assert list(f["replaced"].keys()) == ["example"]
        assert len(f["replaced/example"]) == 1
        assert f["replaced/example"][0] == 10
        f.close()

    def test_rename_dataset_in_hf5_file(self):
        f = h5py.File("./.outdir/rename_dataset.h5")
        group = f.create_group("group")
        group.create_dataset("example", data=np.array([10]))
        f.close()
        utils.rename_group_or_dataset_in_hf5_file("./.outdir/rename_dataset.h5",
            dataset=["group/example", "group/replaced"])
        f = h5py.File("./.outdir/rename_dataset.h5")
        assert list(f.keys()) == ["group"]
        assert list(f["group"].keys()) == ["replaced"]
        assert len(f["group/replaced"]) == 1
        assert f["group/replaced"][0] == 10
        f.close()

    def test_rename_unknown_hf5_file(self):
        with pytest.raises(Exception) as info:
            utils.rename_group_or_dataset_in_hf5_file("./.outdir/unknown.h5",
                group=["None", "replaced"])
        assert "does not exist" in str(info.value) 

    def test_add_new_dataset_to_hdf_file(self):
        content = np.array([10])
        utils.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content)
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["new_dataset", "test"]
        assert list(f["test"].keys()) == ["example"]
        assert len(f["new_dataset"]) == 1
        assert f["new_dataset"][0] == 10

    def test_add_new_group_to_hdf_file(self):
        content = np.array([10])
        utils.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content, group="test")
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["test"]
        assert sorted(list(f["test"].keys())) == ["example", "new_dataset"]
        assert len(f["test/new_dataset"]) == 1
        assert f["test/new_dataset"][0] == 10

    def test_replace_dataset_to_hdf_file(self):
        content = np.array([0])
        utils.add_content_to_hdf_file("./.outdir/add_content.h5",
                                      "new_dataset", content, group="test")
        f = h5py.File("./.outdir/add_content.h5")
        assert sorted(list(f.keys())) == ["test"]
        assert sorted(list(f["test"].keys())) == ["example", "new_dataset"]
        assert len(f["test/new_dataset"]) == 1
        assert f["test/new_dataset"][0] == 0

    def test_combine_hdf_files(self):
        f = h5py.File("./.outdir/combine_hdf_files.h5", "w")
        label = f.create_group("label")
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
        label = g.create_group("label")
        group = label.create_group("approx2")
        group.create_dataset("parameter_names", data=parameters)
        group.create_dataset("samples", data=samples)
        group.create_dataset("injection_parameters", data=parameters)
        group.create_dataset("injection_data", data=injection_data)

        utils.combine_hdf_files("./.outdir/combine_hdf_files.h5",
                                "./.outdir/combine_hdf_files_new.h5")
        f = h5py.File("./.outdir/combine_hdf_files.h5")
        assert sorted(list(f["label"].keys())) == ["approx1", "approx2"]
        assert sorted(list(f["label/approx1"].keys())) == ["injection_data", 
            "injection_parameters", "parameter_names", "samples"]
        assert [i for i in f["label/approx1"]["samples"]] == [[1], [2]]
        assert [i for i in f["label/approx1"]["parameter_names"]] == [b"m1"]
        assert [i for i in f["label/approx1"]["injection_parameters"]] == [b"m1"]
        assert math.isnan(f["label/approx1"]["injection_data"][0])
        assert sorted(list(f["label/approx2"].keys())) == ["injection_data",
            "injection_parameters", "parameter_names", "samples"]
        assert [i for i in f["label/approx2"]["samples"]] == [[1], [2]]
        assert [i for i in f["label/approx2"]["parameter_names"]] == [b"m1"]
        assert [i for i in f["label/approx2"]["injection_parameters"]] == [b"m1"]
        assert math.isnan(f["label/approx2"]["injection_data"][0])

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

def test_logger():
    with LogCapture() as l:
        utils.logger.info("info")
        utils.logger.warning("warning")
    l.check(("PESummary", "INFO", "info"),
            ("PESummary", "WARNING", "warning"),)
