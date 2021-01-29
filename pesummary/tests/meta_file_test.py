# Licensed under an MIT style license -- see LICENSE.md

import json
import os
import shutil

import h5py
import numpy as np

from pesummary.gw.file import meta_file
from pesummary.gw.file.meta_file import _GWMetaFile
from pesummary.gw.inputs import GWInput
from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.array import Array
from .base import data_dir

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def test_recursively_save_dictionary_to_hdf5_file():
    if os.path.isdir("./.outdir_recursive"):
        shutil.rmtree("./.outdir_recursive")
    os.makedirs("./.outdir_recursive")

    data = {
               "H1_L1_IMRPhenomPv2": {
                   "posterior_samples": {
                       "parameters": ["mass_1", "mass_2"],
                       "samples": [[10, 2], [50, 5], [100, 90]]
                       },
               },
               "H1_L1_IMRPhenomP": {
                   "posterior_samples": {
                       "parameters": ["ra", "dec"],
                       "samples": [[0.5, 0.8], [1.2, 0.4], [0.9, 1.5]]
                       },
               },
               "H1_SEOBNRv4": {
                   "posterior_samples": {
                       "parameters": ["psi", "phi"],
                       "samples": [[1.2, 0.2], [3.14, 0.1], [0.5, 0.3]]
                   }
               },
          }

    with h5py.File("./.outdir_recursive/test.h5", "w") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(
            f, data, extra_keys=list(data.keys()))

    f = h5py.File("./.outdir_recursive/test.h5", "r")
    assert sorted(list(f.keys())) == sorted(list(data.keys()))
    assert sorted(
        list(f["H1_L1_IMRPhenomPv2/posterior_samples"].keys())) == sorted(
            ["parameters", "samples"]
    )
    assert f["H1_L1_IMRPhenomPv2/posterior_samples/parameters"][0].decode("utf-8") == "mass_1"
    assert f["H1_L1_IMRPhenomPv2/posterior_samples/parameters"][1].decode("utf-8") == "mass_2"
    assert f["H1_L1_IMRPhenomP/posterior_samples/parameters"][0].decode("utf-8") == "ra"
    assert f["H1_L1_IMRPhenomP/posterior_samples/parameters"][1].decode("utf-8") == "dec"
    assert f["H1_SEOBNRv4/posterior_samples/parameters"][0].decode("utf-8") == "psi"
    assert f["H1_SEOBNRv4/posterior_samples/parameters"][1].decode("utf-8") == "phi"

    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomPv2/posterior_samples/samples"][0],
            [10, 2]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomPv2/posterior_samples/samples"][1],
            [50, 5]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomPv2/posterior_samples/samples"][2],
            [100, 90]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomP/posterior_samples/samples"][0],
            [0.5, 0.8]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomP/posterior_samples/samples"][1],
            [1.2, 0.4]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_L1_IMRPhenomP/posterior_samples/samples"][2],
            [0.9, 1.5]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_SEOBNRv4/posterior_samples/samples"][0],
            [1.2, 0.2]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_SEOBNRv4/posterior_samples/samples"][1],
            [3.14, 0.1]
        )
    )
    assert all(
        i == j for i,j in zip(f["H1_SEOBNRv4/posterior_samples/samples"][2],
            [0.5, 0.3]
        )
    )


def test_softlinks():
    """
    """
    if os.path.isdir("./.outdir_softlinks"):
        shutil.rmtree("./.outdir_softlinks")
    os.makedirs("./.outdir_softlinks")

    data = {
        "label1": {
            "psds": {
                "H1": [[10, 20], [30, 40]],
                "L1": [[10, 20], [30, 40]]
            },
            "config_file": {
                "paths": {
                    "webdir": "example/webdir"
                },
                "condor": {
                    "lalsuite-install": "/example/install",
                    "executable": "%(lalsuite-install)s/executable",
                    "memory": 1000
                },
            },
        },
        "label2": {
            "psds": {
                "H1": [[10, 22], [30, 40]],
                "L1": [[10, 20], [30, 45]]
            },
            "config_file": {
                "paths": {
                    "webdir": "example/webdir2"
                },
                "condor": {
                    "lalsuite-install": "/example/install2",
                    "executable": "%(lalsuite-install)s/executable",
                    "memory": 1000.0
                }
            }
        }
    }

    simlinked_dict = _GWMetaFile._create_softlinks(data)
    repeated_entries = [
        {
            "label1/psds/H1": [
                data["label1"]["psds"]["H1"],
                simlinked_dict["label1"]["psds"]["H1"]
            ],
            "label1/psds/L1": [
                data["label1"]["psds"]["L1"],
                simlinked_dict["label1"]["psds"]["L1"]
            ]
        },
        {
            "label1/config_file/condor/executable": [
                data["label1"]["config_file"]["condor"]["executable"],
                simlinked_dict["label1"]["config_file"]["condor"]["executable"]
            ],
            "label2/config_file/condor/executable": [
                data["label2"]["config_file"]["condor"]["executable"],
                simlinked_dict["label2"]["config_file"]["condor"]["executable"]
            ]
        },
        {
            "label1/config_file/condor/memory": [
                data["label1"]["config_file"]["condor"]["memory"],
                simlinked_dict["label1"]["config_file"]["condor"]["memory"]
            ],
            "label2/config_file/condor/memory": [
                data["label2"]["config_file"]["condor"]["memory"],
                simlinked_dict["label2"]["config_file"]["condor"]["memory"]
            ]
        }
    ]
    for repeat in repeated_entries:
        keys = list(repeat.keys())
        assert \
            repeat[keys[0]][1] == "softlink:/{}".format(keys[1]) and \
            repeat[keys[1]][1] == repeat[keys[1]][0] or \
            repeat[keys[1]][1] == "softlink:/{}".format(keys[0]) and \
            repeat[keys[0]][1] == repeat[keys[0]][0]

    print(simlinked_dict)
    with h5py.File("./.outdir_softlinks/test.h5", "w") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(
            f, simlinked_dict, extra_keys=meta_file.DEFAULT_HDF5_KEYS + ["label1", "label2"])

    with h5py.File("./.outdir_softlinks/no_softlink.h5", "w") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(
            f, data, extra_keys=meta_file.DEFAULT_HDF5_KEYS + ["label1", "label2"])

    softlink_size = os.stat("./.outdir_softlinks/test.h5").st_size
    no_softlink_size = os.stat('./.outdir_softlinks/no_softlink.h5').st_size
    assert softlink_size < no_softlink_size

    with h5py.File("./.outdir_softlinks/test.h5", "r") as f:
        assert \
            f["label2"]["config_file"]["condor"]["executable"][0] == \
            f["label1"]["config_file"]["condor"]["executable"][0]
        assert \
            all(
                i == j for i, j in zip(
                    f["label1"]["psds"]["H1"][0], f["label1"]["psds"]["L1"][0]
                )
            )
        assert \
            all(
                i == j for i, j in zip(
                    f["label1"]["psds"]["H1"][1], f["label1"]["psds"]["L1"][1]
                )
            )
     

class TestMetaFile(object):
    """Class the test the pesummary.gw.file.meta_file._GWMetaFile class
    """
    def setup(self):
        """Setup the Test class
        """
        if not os.path.isdir(".outdir/samples"):
            os.makedirs(".outdir/samples")

        self.samples = np.array([np.random.random(10) for i in range(15)])
        self.input_parameters = [
            "mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_jl",
            "phi_12", "psi", "theta_jn", "ra", "dec", "luminosity_distance",
            "geocent_time", "log_likelihood"]
        self.input_data = {"EXP1": SamplesDict(self.input_parameters, self.samples)}
        distance = np.random.random(10) * 500
        self.input_data["EXP1"]["luminosity_distance"] = Array(distance)
        self.input_labels = ["EXP1"]
        self.input_file_version = {"EXP1": "3.0"}
        self.input_injection_data = np.random.random(15)
        self.input_injection = {"EXP1": {
            i: j for i, j in zip(self.input_parameters, self.input_injection_data)}}
        self.input_file_kwargs = {"EXP1": {
            "sampler": {"flow": 10}, "meta_data": {"samplerate": 10}
        }}
        self.input_config = [data_dir + "/config_lalinference.ini"]
        psd_data = GWInput.extract_psd_data_from_file(data_dir + "/psd_file.txt")
        self.psds = {"EXP1": {"H1": psd_data}}
        calibration_data = GWInput.extract_calibration_data_from_file(
            data_dir + "/calibration_envelope.txt")
        self.calibration = {"EXP1": {"H1": calibration_data}}

        object = _GWMetaFile(
            self.input_data, self.input_labels, self.input_config,
            self.input_injection, self.input_file_version, self.input_file_kwargs,
            webdir=".outdir", psd=self.psds, calibration=self.calibration)
        object.make_dictionary()
        object.save_to_json(object.data, object.meta_file)
        object = _GWMetaFile(
            self.input_data, self.input_labels, self.input_config,
            self.input_injection, self.input_file_version, self.input_file_kwargs,
            webdir=".outdir", psd=self.psds, calibration=self.calibration,
            hdf5=True)
        object.make_dictionary()
        object.save_to_hdf5(
            object.data, object.labels, object.samples, object.meta_file
        )

        with open(".outdir/samples/posterior_samples.json", "r") as f:
            self.json_file = json.load(f)
        self.hdf5_file = h5py.File(".outdir/samples/posterior_samples.h5", "r")

    def teardown(self):
        """Remove all files and directories created from this class
        """
        self.hdf5_file.close()
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_parameters(self):
        """Test the parameters stored in the metafile
        """
        for num, data in enumerate([self.json_file, self.hdf5_file]):
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            if num == 0:
                assert list(
                    sorted(data["EXP1"]["posterior_samples"].keys())) == [
                        "parameter_names", "samples"]
            if num == 0:
                try:
                    assert all(
                        i.decode("utf-8") == j for i, j in zip(
                            sorted(data["EXP1"]["posterior_samples"]["parameter_names"]),
                            sorted(self.input_parameters)))
                except AttributeError:
                    assert all(
                        i == j for i, j in zip(
                            sorted(data["EXP1"]["posterior_samples"]["parameter_names"]),
                            sorted(self.input_parameters)))
            else:
                try:
                    assert all(
                        i.decode("utf-8") == j for i, j in zip(
                            sorted(data["EXP1"]["posterior_samples"].dtype.names),
                            sorted(self.input_parameters)))
                except AttributeError:
                    assert all(
                        i == j for i, j in zip(
                            sorted(data["EXP1"]["posterior_samples"].dtype.names),
                            sorted(self.input_parameters)))

    def test_samples(self):
        """Test the samples stored in the metafile
        """
        for num, data in enumerate([self.json_file, self.hdf5_file]):
            if num == 0:
                parameters = data["EXP1"]["posterior_samples"]["parameter_names"]
                samples = np.array(data["EXP1"]["posterior_samples"]["samples"]).T
            else:
                parameters = [j for j in data["EXP1"]["posterior_samples"].dtype.names]
                samples = np.array([j.tolist() for j in data["EXP1"]["posterior_samples"]]).T
            posterior_data = {"EXP1": {i: j for i, j in zip(parameters, samples)}}
            for param, samp in posterior_data["EXP1"].items():
                if isinstance(param, bytes):
                    param = param.decode("utf-8")
                for ind in np.arange(len(samp)):
                    np.testing.assert_almost_equal(
                        samp[ind], self.input_data["EXP1"][param][ind]
                    )

    def test_file_version(self):
        """Test the file version stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            for i, j in zip(data["EXP1"]["version"], [self.input_file_version["EXP1"]]):
                version = i
                if isinstance(i, bytes):
                    version = version.decode("utf-8")
                assert version == j

    def test_meta_data(self):
        """Test the meta data stored in the metafile
        """
        for num, data in enumerate([self.json_file, self.hdf5_file]):
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            assert sorted(
                list(data["EXP1"]["meta_data"].keys())) == ["meta_data", "sampler"]
            assert all(
               all(
                   k == l for k, l in zip(
                       self.input_file_kwargs["EXP1"][i],
                       data["EXP1"]["meta_data"][j]
                       )
               ) for i, j in zip(
                   sorted(self.input_file_kwargs["EXP1"].keys()),
                   sorted(data["EXP1"]["meta_data"].keys())
               )
            )

            try:
                assert all(
                    all(
                        self.input_file_kwargs["EXP1"][i][k] == data["EXP1"]["meta_data"][j][l]
                        for k, l in zip(
                            self.input_file_kwargs["EXP1"][i],
                            data["EXP1"]["meta_data"][j]
                            )
                    ) for i, j in zip(
                        sorted(self.input_file_kwargs["EXP1"].keys()),
                        sorted(data["EXP1"]["meta_data"].keys())
                    )
                )
            except Exception:
                assert all(
                    all(
                        self.input_file_kwargs["EXP1"][i][k] == data["EXP1"]["meta_data"][j][l][0]
                        for k, l in zip(
                            self.input_file_kwargs["EXP1"][i],
                            data["EXP1"]["meta_data"][j]
                            )
                    ) for i, j in zip(
                        sorted(self.input_file_kwargs["EXP1"].keys()),
                        sorted(data["EXP1"]["meta_data"].keys())
                    )
                )

    def test_psd(self):
        """Test the psd is stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            assert list(
                data["EXP1"]["psds"].keys()) == ["H1"]
            for i, j in zip(self.psds["EXP1"]["H1"], data["EXP1"]["psds"]["H1"]):
                for k, l in zip(i, j):
                    assert k == l

    def test_calibration(self):
        """Test the calibration envelope is stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            assert list(
                data["EXP1"]["calibration_envelope"].keys()) == ["H1"]
            for i, j in zip(self.calibration["EXP1"]["H1"], data["EXP1"]["calibration_envelope"]["H1"]):
                for k, l in zip(i, j):
                    assert k == l

    def test_config(self):
        """Test the configuration file is stored in the metafile
        """
        import configparser

        config_data = []
        for i in self.input_config:
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(i)
            config_data.append(config)

        for num, data in enumerate([self.json_file, self.hdf5_file]):
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            assert all(
                i == j for i, j in zip(
                    sorted(list(config_data[0].sections())),
                    sorted(list(data["EXP1"]["config_file"].keys()))))
            all_options = {
                i: {
                    j: k for j, k in config_data[0][i].items()
                } for i in config_data[0].sections()
            }

            assert all(
                all(
                    k == l for k, l in zip(
                        sorted(all_options[i]),
                        sorted(data["EXP1"]["config_file"][j])
                    )
                ) for i, j in zip(
                    sorted(list(all_options.keys())),
                    sorted(list(data["EXP1"]["config_file"].keys()))
                )
            )

            if num == 0:
                assert all(
                    all(
                        all_options[i][k] == data["EXP1"]["config_file"][j][l]
                        for k, l in zip(
                            sorted(all_options[i]),
                            sorted(data["EXP1"]["config_file"][j])
                        )
                    ) for i, j in zip(
                        sorted(list(all_options.keys())),
                        sorted(list(data["EXP1"]["config_file"].keys()))
                    )
                )
            if num == 1:
                assert all(
                    all(
                        all_options[i][k] == data["EXP1"]["config_file"][j][l][0].decode("utf-8")
                        for k, l in zip(
                            sorted(all_options[i]),
                            sorted(data["EXP1"]["config_file"][j])
                        )
                    ) for i, j in zip(
                        sorted(list(all_options.keys())),
                        sorted(list(data["EXP1"]["config_file"].keys()))
                    )
                )

    def test_injection_data(self):
        """Test the injection data stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert sorted(list(data.keys())) == sorted(
                self.input_labels + ["version", "history"]
            )
            if data == self.json_file:
                for num, i in enumerate(list(self.input_injection["EXP1"].keys())):
                    assert self.input_injection["EXP1"][i] == data["EXP1"]["injection_data"]["samples"][num]
            else:
                for num, i in enumerate(list(self.input_injection["EXP1"].keys())):
                    assert self.input_injection["EXP1"][i] == data["EXP1"]["injection_data"][i]
