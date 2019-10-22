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
import h5py
import numpy as np

import pesummary
import cli
from pesummary.utils import utils
from pesummary._version_helper import GitInformation, PackageInformation
from pesummary._version_helper import get_version_information

import pytest
from testfixtures import LogCapture


class TestGitInformation(object):
    """Class to test the GitInformation helper class
    """
    def setup(self):
        """Setup the TestGitInformation class
        """
        self.git = GitInformation()

    def test_last_commit_info(self):
        """Test the last_commit_info property
        """
        assert len(self.git.last_commit_info) == 2
        assert isinstance(self.git.last_commit_info[0], str)
        assert isinstance(self.git.last_commit_info[1], str)

    def test_last_version(self):
        """Test the last_version property
        """
        assert isinstance(self.git.last_version, str)

    def test_status(self):
        """Test the status property
        """
        assert isinstance(self.git.status, str)

    def test_builder(self):
        """Test the builder property
        """
        assert isinstance(self.git.builder, str)

    def test_build_date(self):
        """Test the build_date property
        """
        assert isinstance(self.git.build_date, str)


class TestPackageInformation(object):
    """Class to test the PackageInformation helper class
    """
    def setup(self):
        """Setup the TestPackageInformation class
        """
        self.package = PackageInformation()

    def test_package_info(self):
        """Test the package_info property
        """
        assert isinstance(self.package.package_info, str)


class TestUtils(object):
    """Class to test pesummary.utils.utils
    """
    def setup(self):
        """Setup the TestUtils class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")
 
    def test_check_condition(self):
        """Test the check_condition method
        """
        with pytest.raises(Exception) as info:
            condition = True
            utils.check_condition(condition, "error")
        assert str(info.value) == "error"

    def test_rename_group_in_hf5_file(self):
        """Test the rename_group_in_hf5_file method
        """
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

    def test_make_dir(self):
        """Test the make_dir method
        """
        assert not os.path.isdir(os.path.join(".outdir", "test"))
        utils.make_dir(os.path.join(".outdir", "test"))
        assert os.path.isdir(os.path.join(".outdir", "test"))
        with open(os.path.join(".outdir", "test", "test.dat"), "w") as f:
            f.writelines(["test"])
        utils.make_dir(os.path.join(".outdir", "test"))
        assert os.path.isfile(os.path.join(".outdir", "test", "test.dat"))

    def test_resample_posterior_distribution(self):
        """Test the resample_posterior_distribution method
        """
        data = np.random.normal(1, 0.1, 1000)
        resampled = utils.resample_posterior_distribution([data], 500)
        assert len(resampled) == 500
        assert np.round(np.mean(resampled), 1) == 1.
        assert np.round(np.std(resampled), 1) == 0.1

    def test_gw_results_file(self):
        """Test the gw_results_file method
        """
        from base import namespace

        opts = namespace({"gw": True, "psd": True})
        assert utils.gw_results_file(opts)
        opts = namespace({"webdir": ".outdir"})
        assert not utils.gw_results_file(opts)

    def test_functions(self):
        """Test the functions method
        """
        from base import namespace

        opts = namespace({"gw": True, "psd": True})
        funcs = utils.functions(opts)
        assert funcs["input"] == pesummary.gw.inputs.GWInput
        assert funcs["PlotGeneration"] == cli.summaryplots.GWPlotGeneration
        assert funcs["WebpageGeneration"] == cli.summarypages.GWWebpageGeneration
        assert funcs["MetaFile"] == pesummary.gw.file.meta_file.GWMetaFile

        opts = namespace({"webdir": ".outdir"})
        funcs = utils.functions(opts)
        assert funcs["input"] == pesummary.core.inputs.Input
        assert funcs["PlotGeneration"] == cli.summaryplots.PlotGeneration
        assert funcs["WebpageGeneration"] == cli.summarypages.WebpageGeneration
        assert funcs["MetaFile"] == pesummary.core.file.meta_file.MetaFile

    def test_get_version_information(self):
        """Test the get_version_information method
        """
        assert isinstance(get_version_information(), str)


def test_logger():
    with LogCapture() as l:
        utils.logger.info("info")
        utils.logger.warning("warning")
    l.check(("PESummary", "INFO", "info"),
            ("PESummary", "WARNING", "warning"),)
