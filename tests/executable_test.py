import os
import shutil
import glob
import subprocess
import numpy as np

from base import make_result_file, get_list_of_plots, get_list_of_files
import pytest
from pesummary.utils.exceptions import InputError
import importlib


class Base(object):
    """Base class for testing the executables
    """
    def launch(self, command_line):
        """
        """
        args = command_line.split(" ")
        executable = args[0]
        cla = args[1:]
        module = importlib.import_module("pesummary.cli.{}".format(executable))
        print(cla)
        module.main(args=[i for i in cla if i != " " and i != ""])


class TestSummaryPages(Base):
    """Test the `summarypages` executable with trivial examples
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(gw=False, extension="json")
        os.rename(".outdir/test.json", ".outdir/example.json")
        make_result_file(gw=False, extension="hdf5")
        os.rename(".outdir/test.h5", ".outdir/example2.h5")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def check_output(self, number=1):
        """Check the output from the summarypages executable
        """
        assert os.path.isfile(".outdir/home.html")
        plots = get_list_of_plots(gw=False, number=number)
        assert all(
            i == j for i, j in zip(
                sorted(plots), sorted(glob.glob("./.outdir/plots/*.png"))
            )
        )
        files = get_list_of_files(gw=False, number=number)
        assert all(
            i == j for i, j in zip(
                sorted(files), sorted(glob.glob("./.outdir/html/*.html"))
            )
        )

    def test_prior_input(self):
        """Check that `summarypages` works when a prior file is passed from
        the command line
        """
        import importlib

        for package in ["core", "gw"]:
            gw = True if package == "gw" else False
            module = importlib.import_module(
                "pesummary.{}.file.read".format(package)
            )
            make_result_file(gw=gw, extension="json")
            os.rename(".outdir/test.json", ".outdir/prior.json")
            command_line = (
                "summarypages --webdir .outdir --samples .outdir/example.json "
                "--prior_file .outdir/prior.json --labels test"
            )
            command_line += " --gw" if gw else ""
            self.launch(command_line)
            f = module.read(".outdir/samples/posterior_samples.h5")
            stored = f.priors["samples"]["test"]
            f = module.read(".outdir/prior.json")
            original = f.samples_dict
            for param in original.keys():
                np.testing.assert_almost_equal(
                    original[param], stored[param]
                )

    def test_calibration_and_psd(self):
        """Test that the calibration and psd files are passed appropiately
        """
        from pesummary.gw.file.read import read
        from base import make_psd, make_calibration

        make_psd()
        make_calibration()
        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--psd H1:.outdir/psd.dat --calibration L1:.outdir/calibration.dat "
            "--labels test"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        psd = np.genfromtxt(".outdir/psd.dat")
        calibration = np.genfromtxt(".outdir/calibration.dat")
        np.testing.assert_almost_equal(f.psd["test"]["H1"], psd[:-2])
        np.testing.assert_almost_equal(
            f.priors["calibration"]["test"]["L1"], calibration
        )

    def test_gracedb(self):
        """Test that when the gracedb ID is passed from the command line it is
        correctly stored in the meta data
        """
        from pesummary.gw.file.read import read

        command_line = (
            "summarypages --webdir .outdir --samples .outdir/example.json "
            "--gracedb G17864 --gw --labels test"
        )
        self.launch(command_line)
        f = read(".outdir/samples/posterior_samples.h5")
        assert "gracedb" in f.extra_kwargs[0]["meta_data"]
        assert "G17864" == f.extra_kwargs[0]["meta_data"]["gracedb"]

    def test_single(self):
        """Test on a single input
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json --label core0"
        )
        self.launch(command_line)
        self.check_output(number=1)

    def test_mcmc(self):
        """Test the `--mcmc_samples` command line argument
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            "--label core0 --mcmc_samples"
        )
        self.launch(command_line)
        self.check_output(number=1)

    def test_mcmc_more_than_label(self):
        """Test that the code fails with the `--mcmc_samples` command line
        argument when multiple labels are passed.
        """
        command_line = (
            "summarypages --webdir .outdir --samples "
            ".outdir/example.json .outdir/example2.h5 "
            ".outdir/example.json .outdir/example2.h5 "
            "--label core0 core1 --mcmc_samples"
        )
        with pytest.raises(InputError): 
            self.launch(command_line)


class TestSummaryClassification(Base):
    """Test the `summaryclassification` executable
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True, gw=True, pesummary_label="test")
        os.rename(".outdir/test.json", ".outdir/pesummary.json")
        make_result_file(bilby=True, gw=True)
        os.rename(".outdir/test.json", ".outdir/bilby.json")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def check_output(self):
        """Check the output from the `summaryclassification` executable
        """
        import glob
        import json

        files = glob.glob(".outdir/*")
        assert ".outdir/test_default_prior_pe_classification.json" in files
        assert ".outdir/test_default_pepredicates_bar.png" in files
        with open(".outdir/test_default_prior_pe_classification.json", "r") as f:
            data = json.load(f)
        assert all(
            i in data.keys() for i in [
                "BNS", "NSBH", "BBH", "MassGap", "HasNS", "HasRemnant"
            ]
        )

    def test_result_file(self):
        """Test the `summaryclassification` executable for a random result file
        """
        command_line = (
            "summaryclassification --webdir .outdir --samples "
            ".outdir/bilby.json --prior default --label test"
        )
        self.launch(command_line)
        self.check_output()

    def test_pesummary_file(self):
        """Test the `summaryclassification` executable for a pesummary metafile
        """
        command_line = (
            "summaryclassification --webdir .outdir --samples "
            ".outdir/pesummary.json --prior default"
        )
        self.launch(command_line)
        self.check_output()


class TestSummaryClean(Base):
    """Test the `summaryclean` executable
    """
    def setup(self):
        """Setup the SummaryClassification class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_clean(self):
        """Test the `summaryclean` executable
        """
        import h5py

        parameters = ["mass_ratio"]
        data = [[0.5], [0.5], [-1.5]]
        h5py_data = np.array(
            [tuple(i) for i in data], dtype=[tuple([i, 'float64']) for i in
            parameters]
        )
        f = h5py.File(".outdir/test.hdf5", "w")
        lalinference = f.create_group("lalinference")
        nest = lalinference.create_group("lalinference_nest")
        samples = nest.create_dataset("posterior_samples", data=h5py_data)
        f.close()
        command_line = (
            "summaryclean --webdir .outdir --samples .outdir/test.hdf5 "
            "--file_format dat --labels test"
        )
        self.launch(command_line)
        self.check_output()

    def check_output(self):
        """Check the output from the `summaryclean` executable
        """
        from pesummary.gw.file.read import read

        f = read(".outdir/pesummary_test.dat")
        print(f.samples_dict["mass_ratio"])
        assert len(f.samples_dict["mass_ratio"]) == 2
        assert all(i == 0.5 for i in f.samples_dict["mass_ratio"])


class _SummaryCombine_Metafiles(Base):
    """Test the `summarycombine_metafile` executable
    """
    def test_combine(self, gw=False):
        """Test the executable for 2 metafiles
        """
        make_result_file(pesummary=True, pesummary_label="label2")
        os.rename(".outdir/test.json", ".outdir/test2.json")
        make_result_file(pesummary=True)
        command_line = (
            "summarycombine --webdir .outdir "
            "--samples .outdir/test.json .outdir/test2.json "
            "--save_to_json"
        )
        if gw:
            command_line += " --gw"
        self.launch(command_line)

    def check_output(self, gw=False):
        if gw:
            from pesummary.gw.file.read import read
        else:
            from pesummary.core.file.read import read

        assert os.path.isfile(".outdir/samples/posterior_samples.json")
        combined = read(".outdir/samples/posterior_samples.json")
        for f in [".outdir/test.json", ".outdir/test2.json"]:
            data = read(f)
            labels = data.labels
            assert all(i in combined.labels for i in labels)
            assert all(
                all(
                    data.samples_dict[j][num] == combined.samples_dict[i][j][num]
                    for num in range(data.samples_dict[j])
                ) for j in data.samples_dict.keys()
            )


class TestCoreSummaryCombine_Metafiles(_SummaryCombine_Metafiles):
    """Test the `summarycombine_metafile` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_combine(self):
        """Test the executable for 2 metafiles
        """
        super(TestCoreSummaryCombine_Metafiles, self).test_combine(gw=False)

    def check_output(self):
        super(TestCoreSummaryCombine_Metafiles, self).check_output(gw=False)


class TestGWSummaryCombine_Metafiles(_SummaryCombine_Metafiles):
    """Test the `summarycombine_metafile` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(pesummary=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_combine(self):
        """Test the executable for 2 metafiles
        """
        super(TestGWSummaryCombine_Metafiles, self).test_combine(gw=True)

    def check_output(self, gw=True):
        super(TestGWSummaryCombine_Metafiles, self).check_output(gw=True)


class TestSummaryReview(Base):
    """Test the `summaryreview` executable
    """
    def setup(self):
        """Setup the SummaryCombine_Metafiles class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(lalinference=True)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_review(self):
        """Test the `summaryreview` script for a `lalinference` result file
        """
        command_line = (
            "summaryreview --webdir .outdir --samples .outdir/test.hdf5"
        )
        self.launch(command_line)


class TestSummaryModify(Base):
    """Test the `summarymodify` executable
    """
    def setup(self):
        """Setup the SummaryModify class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(
            pesummary=True, pesummary_label="replace", extension="hdf5"
        )

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_modify(self):
        """Test the `summarymodify` script
        """
        import h5py

        command_line = (
            "summarymodify --webdir .outdir --samples .outdir/test.h5 "
            "--labels replace:new"
        )
        self.launch(command_line)
        modified_data = h5py.File(".outdir/modified_posterior_samples.h5", "r")
        data = h5py.File(".outdir/test.h5", "r")
        assert "replace" not in list(modified_data.keys())
        assert "new" in list(modified_data.keys())
        for key in data["replace"].keys():
            assert key in modified_data["new"].keys()
            assert all(i == j for i, j in zip(
                data["replace"][key], modified_data["new"][key]
            ))
            for i, j in zip(data["replace"][key], modified_data["new"][key]):
                try:
                    if isinstance(data["replace"][key][i],h5py._hl.dataset.Dataset):
                        try:
                            assert all(k == l for k, l in zip(
                                data["replace"][key][i],
                                modified_data["new"][key][j]
                            ))
                        except ValueError:
                            assert all(
                                all(m == n for m, n in zip(k, l)) for k, l in zip(
                                    data["replace"][key][i],
                                    modified_data["new"][key][j]
                                )
                            )
                except TypeError:
                    pass
        data.close()
        modified_data.close()


class TestSummaryRecreate(Base):
    """Test the `summaryrecreate` executable
    """
    def setup(self):
        """Setup the SummaryRecreate class
        """
        import configparser

        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read("tests/files/config_lalinference.ini")
        config_dictionary = dict(config._sections)
        config_dictionary["paths"]["webdir"] = (
            "./{}/webdir".format(os.environ["USER"])
        )
        make_result_file(
            pesummary=True, pesummary_label="recreate", extension="hdf5",
            config=config_dictionary
        )
        with open("GW150914.txt", "w") as f:
            f.writelines(["115"])

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_recreate(self):
        """Test the `summaryrecreate` script
        """
        import configparser

        command_line = (
            "summaryrecreate --rundir .outdir --samples .outdir/test.h5 "
        )
        self.launch(command_line)
        assert os.path.isdir(os.path.join(".outdir", "recreate"))
        assert os.path.isfile(os.path.join(".outdir", "recreate", "config.ini"))
        assert os.path.isdir(os.path.join(".outdir", "recreate", "outdir"))
        assert os.path.isdir(os.path.join(".outdir", "recreate", "outdir", "caches"))
        config = configparser.ConfigParser()
        config.read(os.path.join(".outdir", "recreate", "config.ini"))
        original_config = configparser.ConfigParser()
        original_config.read("tests/files/config_lalinference.ini")
        for a, b in zip(
          sorted(config.sections()), sorted(original_config.sections())
        ):
            assert a == b
            for key, item in config[a].items():
                assert config[b][key] == item
        command_line = (
            "summaryrecreate --rundir .outdir_modify --samples .outdir/test.h5 "
            "--config_override approx:IMRPhenomPv3HM srate:4096"
        )
        self.launch(command_line)
        config = configparser.ConfigParser()
        config.read(os.path.join(".outdir_modify", "recreate", "config.ini"))
        original_config = configparser.ConfigParser()
        original_config.read("tests/files/config_lalinference.ini")
        for a, b in zip(
          sorted(config.sections()), sorted(original_config.sections())
        ):
            assert a == b
            for key, item in config[a].items():
                if key == "approx":
                    assert original_config[b][key] != item
                    assert config[b][key] == "IMRPhenomPv3HM"
                elif key == "srate":
                    assert original_config[b][key] != item
                    assert config[b][key] == "4096"
                elif key == "webdir":
                    pass
                else:
                    assert original_config[b][key] == item
