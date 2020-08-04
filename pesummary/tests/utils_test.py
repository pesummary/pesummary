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
import copy

import pesummary
from pesummary.io import write
import pesummary.cli as cli
from pesummary.utils import utils
from pesummary.utils.tqdm import tqdm
from pesummary.utils.dict import Dict
from pesummary.utils.samples_dict import (
    Array, SamplesDict, MCMCSamplesDict, MultiAnalysisSamplesDict
)
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
        self.git = GitInformation(directory="/builds/lscsoft/pesummary/")

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
        pi = self.package.package_info
        assert isinstance(pi, list)
        pkg = pi[0]
        assert "name" in pkg
        assert "version" in pkg
        if "build_string" in pkg:  # conda only
            assert "channel" in pkg


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
        from .base import namespace

        opts = namespace({"gw": True, "psd": True})
        assert utils.gw_results_file(opts)
        opts = namespace({"webdir": ".outdir"})
        assert not utils.gw_results_file(opts)

    def test_functions(self):
        """Test the functions method
        """
        from .base import namespace

        opts = namespace({"gw": True, "psd": True})
        funcs = utils.functions(opts)
        assert funcs["input"] == pesummary.gw.inputs.GWInput
        assert funcs["MetaFile"] == pesummary.gw.file.meta_file.GWMetaFile

        opts = namespace({"webdir": ".outdir"})
        funcs = utils.functions(opts)
        assert funcs["input"] == pesummary.core.inputs.Input
        assert funcs["MetaFile"] == pesummary.core.file.meta_file.MetaFile

    def test_get_version_information(self):
        """Test the get_version_information method
        """
        assert isinstance(get_version_information(), str)


class TestGelmanRubin(object):
    """Test the Gelman Rubin calculation
    """
    def test_same_as_lalinference(self):
        """Test the Gelman rubin output from pesummary is the same as
        the one coded in LALInference
        """
        from lalinference.bayespputils import Posterior
        from pesummary.utils.utils import gelman_rubin

        header = ["a", "b", "logL", "chain"]
        for _ in np.arange(100):
            samples = np.array(
                [
                    np.random.uniform(np.random.random(), 0.1, 3).tolist() +
                    [np.random.randint(1, 3)] for _ in range(10)
                ]
            )
            obj = Posterior([header, np.array(samples)])
            R = obj.gelman_rubin("a")
            chains = np.unique(obj["chain"].samples)
            chain_index = obj.names.index("chain")
            param_index = obj.names.index("a")
            data, _ = obj.samples()
            chainData=[
                data[data[:,chain_index] == chain, param_index] for chain in
                chains
            ]
            np.testing.assert_almost_equal(
                gelman_rubin(chainData, decimal=10), R, 7
            )

    def test_same_samples(self):
        """Test that when passed two identical chains (perfect convergence),
        the Gelman Rubin is 1
        """
        from pesummary.core.plots.plot import gelman_rubin

        samples = np.random.uniform(1, 0.5, 10)
        R = gelman_rubin([samples, samples])
        assert R == 1


class TestSamplesDict(object):
    """Test the SamplesDict class
    """
    def setup(self):
        self.parameters = ["a", "b"]
        self.samples = [
            np.random.uniform(10, 0.5, 100), np.random.uniform(200, 10, 100)
        ]
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        write(
            self.parameters, np.array(self.samples).T, outdir=".outdir",
            filename="test.dat", file_format="dat"
        )

    def teardown(self):
        """Remove the files created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_initalize(self):
        """Test that the two ways to initialize the SamplesDict class are
        equivalent
        """
        base = SamplesDict(self.parameters, self.samples)
        other = SamplesDict(
            {
                param: sample for param, sample in zip(
                    self.parameters, self.samples
                )
            }
        )
        assert base.parameters == other.parameters
        assert sorted(base.parameters) == sorted(self.parameters)
        np.testing.assert_almost_equal(base.samples, other.samples)
        assert sorted(list(base.keys())) == sorted(list(other.keys()))
        np.testing.assert_almost_equal(base.samples, self.samples)
        class_method = SamplesDict.from_file(
            ".outdir/test.dat", add_zero_likelihood=False
        )
        np.testing.assert_almost_equal(class_method.samples, self.samples)

    def test_properties(self):
        """Test that the properties of the SamplesDict class are correct
        """
        import pandas as pd

        dataset = SamplesDict(self.parameters, self.samples)
        assert sorted(dataset.minimum.keys()) == sorted(self.parameters)
        assert dataset.minimum["a"] == np.min(self.samples[0])
        assert dataset.minimum["b"] == np.min(self.samples[1])
        assert dataset.median["a"] == np.median(self.samples[0])
        assert dataset.median["b"] == np.median(self.samples[1])
        assert dataset.mean["a"] == np.mean(self.samples[0])
        assert dataset.mean["b"] == np.mean(self.samples[1])
        assert dataset.number_of_samples == len(self.samples[1])
        assert len(dataset.downsample(10)["a"]) == 10
        dataset = SamplesDict(self.parameters, self.samples)
        assert len(dataset.discard_samples(10)["a"]) == len(self.samples[0]) - 10
        p = dataset.to_pandas()
        assert isinstance(p, pd.core.frame.DataFrame)
        remove = dataset.pop("a")
        assert list(dataset.keys()) == ["b"]


class TestMultiAnalysisSamplesDict(object):
    """Test the MultiAnalysisSamplesDict class
    """
    def setup(self):
        self.parameters = ["a", "b"]
        self.samples = [
            [np.random.uniform(10, 0.5, 100), np.random.uniform(100, 10, 100)],
            [np.random.uniform(5, 0.5, 100), np.random.uniform(80, 10, 100)],
        ]
        self.labels = ["one", "two"]
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        for num, _samples in enumerate(self.samples):
            write(
                self.parameters, np.array(_samples).T, outdir=".outdir",
                filename="test_{}.dat".format(num + 1), file_format="dat"
            )

    def teardown(self):
        """Remove the files created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_initalize(self):
        """Test the different ways to initalize the class
        """
        dataframe = MultiAnalysisSamplesDict(
            self.parameters, self.samples, labels=["one", "two"]
        )
        assert sorted(list(dataframe.keys())) == sorted(self.labels)
        assert sorted(list(dataframe["one"])) == sorted(["a", "b"])
        assert sorted(list(dataframe["two"])) == sorted(["a", "b"])
        np.testing.assert_almost_equal(
            dataframe["one"]["a"], self.samples[0][0]
        )
        np.testing.assert_almost_equal(
            dataframe["one"]["b"], self.samples[0][1]
        )
        np.testing.assert_almost_equal(
            dataframe["two"]["a"], self.samples[1][0]
        )
        np.testing.assert_almost_equal(
            dataframe["two"]["b"], self.samples[1][1]
        )
        _other = MCMCSamplesDict({
            label: {
                param: self.samples[num][idx] for idx, param in enumerate(
                    self.parameters
                )
            } for num, label in enumerate(self.labels)
        })
        class_method = MultiAnalysisSamplesDict.from_files(
            {'one': ".outdir/test_1.dat", 'two': ".outdir/test_2.dat"},
            add_zero_likelihood=False
        )
        for other in [_other, class_method]:
            assert sorted(other.keys()) == sorted(dataframe.keys())
            assert sorted(other["one"].keys()) == sorted(
                dataframe["one"].keys()
            )
            np.testing.assert_almost_equal(
                other["one"]["a"], dataframe["one"]["a"]
            )
            np.testing.assert_almost_equal(
                other["one"]["b"], dataframe["one"]["b"]
            )
            np.testing.assert_almost_equal(
                other["two"]["a"], dataframe["two"]["a"]
            )
            np.testing.assert_almost_equal(
                other["two"]["b"], dataframe["two"]["b"]
            )
        

    def test_different_samples_for_different_analyses(self):
        """Test that nothing breaks when different samples have different parameters
        """
        data = {
            "one": {
                "a": np.random.uniform(10, 0.5, 100),
                "b": np.random.uniform(5, 0.5, 100)
            }, "two": {
                "a": np.random.uniform(10, 0.5, 100)
            }
        }
        dataframe = MultiAnalysisSamplesDict(data)
        assert sorted(dataframe["one"].keys()) == sorted(data["one"].keys())
        assert sorted(dataframe["two"].keys()) == sorted(data["two"].keys())
        np.testing.assert_almost_equal(
            dataframe["one"]["a"], data["one"]["a"]
        )
        np.testing.assert_almost_equal(
            dataframe["one"]["b"], data["one"]["b"]
        )
        np.testing.assert_almost_equal(
            dataframe["two"]["a"], data["two"]["a"]
        )
        with pytest.raises(ValueError):
            transpose = dataframe.T


class TestMCMCSamplesDict(object):
    """Test the MCMCSamplesDict class
    """
    def setup(self):
        self.parameters = ["a", "b"]
        self.chains = [
            [np.random.uniform(10, 0.5, 100), np.random.uniform(100, 10, 100)],
            [np.random.uniform(5, 0.5, 100), np.random.uniform(80, 10, 100)]
        ]

    def test_initalize(self):
        """Test the different ways to initalize the class
        """
        dataframe = MCMCSamplesDict(self.parameters, self.chains)
        assert sorted(list(dataframe.keys())) == sorted(
            ["chain_{}".format(num) for num in range(len(self.chains))]
        )
        assert sorted(list(dataframe["chain_0"].keys())) == sorted(["a", "b"])
        assert sorted(list(dataframe["chain_1"].keys())) == sorted(["a", "b"])
        np.testing.assert_almost_equal(
            dataframe["chain_0"]["a"], self.chains[0][0]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_0"]["b"], self.chains[0][1]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_1"]["a"], self.chains[1][0]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_1"]["b"], self.chains[1][1]
        )
        other = MCMCSamplesDict({
            "chain_{}".format(num): {
                param: self.chains[num][idx] for idx, param in enumerate(
                    self.parameters
                )
            } for num in range(len(self.chains))
        })
        assert sorted(other.keys()) == sorted(dataframe.keys())
        assert sorted(other["chain_0"].keys()) == sorted(
            dataframe["chain_0"].keys()
        )
        np.testing.assert_almost_equal(
            other["chain_0"]["a"], dataframe["chain_0"]["a"]
        )
        np.testing.assert_almost_equal(
            other["chain_0"]["b"], dataframe["chain_0"]["b"]
        )
        np.testing.assert_almost_equal(
            other["chain_1"]["a"], dataframe["chain_1"]["a"]
        )
        np.testing.assert_almost_equal(
            other["chain_1"]["b"], dataframe["chain_1"]["b"]
        )

    def test_unequal_chain_length(self):
        """Test that when inverted, the chains keep their unequal chain
        length
        """
        chains = [
            [np.random.uniform(10, 0.5, 100), np.random.uniform(100, 10, 100)],
            [np.random.uniform(5, 0.5, 200), np.random.uniform(80, 10, 200)]
        ]
        dataframe = MCMCSamplesDict(self.parameters, chains)
        transpose = dataframe.T
        assert len(transpose["a"]["chain_0"]) == 100
        assert len(transpose["a"]["chain_1"]) == 200
        assert dataframe.number_of_samples == {
            "chain_0": 100, "chain_1": 200
        }
        assert dataframe.minimum_number_of_samples == 100
        assert transpose.number_of_samples == dataframe.number_of_samples
        assert transpose.minimum_number_of_samples == \
            dataframe.minimum_number_of_samples
        combined = dataframe.combine
        assert combined.number_of_samples == 300
        np.testing.assert_almost_equal(
            np.concatenate(
                [dataframe["chain_0"]["a"], dataframe["chain_1"]["a"]]
            ), combined["a"]
        )

    def test_properties(self):
        """Test that the properties of the MCMCSamplesDict class are correct
        """
        dataframe = MCMCSamplesDict(self.parameters, self.chains)
        transpose = dataframe.T
        np.testing.assert_almost_equal(
            dataframe["chain_0"]["a"], transpose["a"]["chain_0"]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_0"]["b"], transpose["b"]["chain_0"]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_1"]["a"], transpose["a"]["chain_1"]
        )
        np.testing.assert_almost_equal(
            dataframe["chain_1"]["b"], transpose["b"]["chain_1"]
        )
        average = dataframe.average
        transpose_average = transpose.average
        for param in self.parameters:
            np.testing.assert_almost_equal(
                average[param], transpose_average[param]
            )
        assert dataframe.total_number_of_samples == 200
        assert dataframe.total_number_of_samples == \
            transpose.total_number_of_samples
        combined = dataframe.combine
        assert combined.number_of_samples == 200
        np.testing.assert_almost_equal(
            np.concatenate(
                [dataframe["chain_0"]["a"], dataframe["chain_1"]["a"]]
            ), combined["a"]
        )

    def test_burnin_removal(self):
        """Test that the different methods for removing the samples as burnin
        as expected
        """
        uniform = np.random.uniform
        parameters = ["a", "b", "cycle"]
        chains = [
            [uniform(10, 0.5, 100), uniform(100, 10, 100), uniform(1, 0.8, 100)],
            [uniform(5, 0.5, 100), uniform(80, 10, 100), uniform(1, 0.8, 100)],
            [uniform(1, 0.8, 100), uniform(90, 10, 100), uniform(1, 0.8, 100)]
        ]
        dataframe = MCMCSamplesDict(parameters, chains)
        burnin = dataframe.burnin(algorithm="burnin_by_step_number")
        idxs = np.argwhere(chains[0][2] > 0)
        assert len(burnin["chain_0"]["a"]) == len(idxs)
        dataframe = MCMCSamplesDict(parameters, chains)
        burnin = dataframe.burnin(10, algorithm="burnin_by_first_n")
        assert len(burnin["chain_0"]["a"]) == 90
        dataframe = MCMCSamplesDict(parameters, chains)
        burnin = dataframe.burnin(
            10, algorithm="burnin_by_first_n", step_number=True
        )
        assert len(burnin["chain_0"]["a"]) == len(idxs) - 10
        
        


class TestArray(object):
    """Test the Array class
    """
    def test_properties(self):
        samples = np.random.uniform(100, 10, 100)
        array = Array(samples)
        assert array.average(type="mean") == np.mean(samples)
        assert array.average(type="median") == np.median(samples)
        assert array.standard_deviation == np.std(samples)
        np.testing.assert_almost_equal(
            array.confidence_interval(percentile=[5, 95]),
            [np.percentile(array, 5), np.percentile(array, 95)]
        )

    def test_weighted_percentile(self):
        x = np.random.normal(100, 20, 10000)
        weights = np.array([np.random.randint(100) for _ in range(10000)])
        array = Array(x, weights=weights)
        numpy = np.percentile(np.repeat(x, weights), 90)
        pesummary = array.confidence_interval(percentile=90)
        np.testing.assert_almost_equal(numpy, pesummary, 6)


class TestTQDM(object):
    """Test the pesummary.utils.tqdm.tqdm class
    """
    def setup(self):
        self._range = range(100)
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_basic_iterator(self):
        """Test that the core functionality of the tqdm class remains
        """
        for j in tqdm(self._range):
            _ = j*j

    def test_interaction_with_logger(self):
        """Test that tqdm interacts nicely with logger
        """
        from pesummary.utils.utils import logger, LOG_FILE

        with open("./.outdir/test.dat", "w") as f:
            for j in tqdm(self._range, logger=logger, file=f):
                _ = j*j

        with open("./.outdir/test.dat", "r") as f:
            lines = f.readlines()
            assert "PESummary" in lines[-1]
            assert "INFO" in lines[-1]
        

def test_jensen_shannon_divergence():
    """Test that the `jensen_shannon_divergence` method returns the same
    values as the scipy function
    """
    from scipy.spatial.distance import jensenshannon
    from scipy import stats

    samples = [
        np.random.uniform(5, 4, 100),
        np.random.uniform(5, 4, 100)
    ]
    x = np.linspace(np.min(samples), np.max(samples), 100)
    kde = [stats.gaussian_kde(i)(x) for i in samples]
    _scipy = jensenshannon(*kde)**2
    _pesummary = utils.jensen_shannon_divergence(samples, decimal=9)
    np.testing.assert_almost_equal(_scipy, _pesummary)

    from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde

    _pesummary = utils.jensen_shannon_divergence(
        samples, decimal=9, kde=Bounded_1d_kde, xlow=4.5, xhigh=5.5
    )


def test_make_cache_style_file():
    """Test that the `make_cache_style_file` works as expected
    """
    from pesummary.utils.utils import make_cache_style_file

    sty = os.path.expanduser("~/.cache/pesummary/style/matplotlib_rcparams.sty")
    with open("test.sty", "w") as f:
        f.writelines(["test : 10"])
    make_cache_style_file("test.sty")
    assert os.path.isfile(sty)
    with open(sty, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert lines[0] == "test : 10"


def test_logger():
    with LogCapture() as l:
        utils.logger.info("info")
        utils.logger.warning("warning")
    l.check(("PESummary", "INFO", "info"),
            ("PESummary", "WARNING", "warning"),)


class TestDict(object):
    """Class to test the NestedDict object
    """
    def test_initiate(self):
        """Initiate the Dict class
        """
        from pesummary.gw.file.psd import PSD

        x = Dict(
            {"a": [[10, 20], [10, 20]]}, value_class=PSD,
            value_columns=["value", "value2"]
        )
        assert list(x.keys()) == ["a"]
        np.testing.assert_almost_equal(x["a"], [[10, 20], [10, 20]])
        assert isinstance(x["a"], PSD)
        np.testing.assert_almost_equal(x["a"].value, [10, 10])
        np.testing.assert_almost_equal(x["a"].value2, [20, 20])

        x = Dict(
            ["a"], [[[10, 20], [10, 20]]], value_class=PSD,
            value_columns=["value", "value2"]
        )
        assert list(x.keys()) == ["a"]
        np.testing.assert_almost_equal(x["a"], [[10, 20], [10, 20]])
        assert isinstance(x["a"], PSD)
        np.testing.assert_almost_equal(x["a"].value, [10, 10])
        np.testing.assert_almost_equal(x["a"].value2, [20, 20])


def make_cache_style_file(style_file):
    """Make a cache directory which stores the style file you wish to use
    when plotting

    Parameters
    ----------
    style_file: str
        path to the style file that you wish to use when plotting
    """
    make_dir(CACHE_DIR)
    shutil.copyfile(
        style_file, os.path.join(CACHE_DIR, "matplotlib_rcparams.sty")
    )


def get_matplotlib_style_file():
    """Return the path to the matplotlib style file that you wish to use
    """
    return os.path.join(CACHE_DIR, "matplotlib_rcparams.sty")
