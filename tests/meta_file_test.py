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

import json
import os
import shutil

import h5py
import numpy as np

from pesummary.gw.file import meta_file
from pesummary.gw.file.meta_file import _GWMetaFile
from pesummary.gw.inputs import GWInput
from pesummary.utils.utils import SamplesDict, Array


def test_recursively_save_dictionary_to_hdf5_file():
    if os.path.isdir("./.outdir"):
        shutil.rmtree("./.outdir")
    os.makedirs("./.outdir")

    data = {
               "posterior_samples": {
                   "H1_L1_IMRPhenomPv2": {
                       "parameters": ["mass_1", "mass_2"],
                       "samples": [[10, 2], [50, 5], [100, 90]]
                       },
                   "H1_L1_IMRPhenomP": {
                       "parameters": ["ra", "dec"],
                       "samples": [[0.5, 0.8], [1.2, 0.4], [0.9, 1.5]]
                       },
                   "H1_SEOBNRv4": {
                       "parameters": ["psi", "phi"],
                       "samples": [[1.2, 0.2], [3.14, 0.1], [0.5, 0.3]]
                       }
                   },
               }

    with h5py.File("./.outdir/test.h5") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(f, data)

    f = h5py.File("./.outdir/test.h5", "r")
    assert sorted(list(f.keys())) == sorted(["posterior_samples"])
    assert sorted(list(f["posterior_samples"].keys())) == sorted(
        ["H1_L1_IMRPhenomPv2", "H1_L1_IMRPhenomP", "H1_SEOBNRv4"]
    )
    assert sorted(
        list(f["posterior_samples/H1_L1_IMRPhenomPv2"].keys())) == sorted(
            ["parameters", "samples"]
    )
    assert f["posterior_samples/H1_L1_IMRPhenomPv2/parameters"][0].decode("utf-8") == "mass_1"
    assert f["posterior_samples/H1_L1_IMRPhenomPv2/parameters"][1].decode("utf-8") == "mass_2"
    assert f["posterior_samples/H1_L1_IMRPhenomP/parameters"][0].decode("utf-8") == "ra"
    assert f["posterior_samples/H1_L1_IMRPhenomP/parameters"][1].decode("utf-8") == "dec"
    assert f["posterior_samples/H1_SEOBNRv4/parameters"][0].decode("utf-8") == "psi"
    assert f["posterior_samples/H1_SEOBNRv4/parameters"][1].decode("utf-8") == "phi"

    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][0],
            [10, 2]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][1],
            [50, 5]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomPv2/samples"][2],
            [100, 90]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][0],
            [0.5, 0.8]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][1],
            [1.2, 0.4]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_L1_IMRPhenomP/samples"][2],
            [0.9, 1.5]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][0],
            [1.2, 0.2]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][1],
            [3.14, 0.1]
        )
    )
    assert all(
        i == j for i,j in zip(f["posterior_samples/H1_SEOBNRv4/samples"][2],
            [0.5, 0.3]
        )
    )


def test_softlinks():
    """
    """
    if os.path.isdir("./.outdir"):
        shutil.rmtree("./.outdir")
    os.makedirs("./.outdir")

    data = {
        "psds": {
            "label1": {
                "H1": [[10, 20], [30, 40]],
                "L1": [[10, 20], [30, 40]]
            },
            "label2": {
                "H1": [[10, 22], [30, 40]],
                "L1": [[10, 20], [30, 45]]
            },
        },
        "config_file": {
            "label1": {
                "paths": {
                    "webdir": "example/webdir"
                },
                "condor": {
                    "lalsuite-install": "/example/install",
                    "executable": "%(lalsuite-install)s/executable",
                    "memory": 1000
                },
            },
            "label2": {
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
            "psds/label1/H1": [
                data["psds"]["label1"]["H1"],
                simlinked_dict["psds"]["label1"]["H1"]
            ],
            "psds/label1/L1": [
                data["psds"]["label1"]["L1"],
                simlinked_dict["psds"]["label1"]["L1"]
            ]
        },
        {
            "config_file/label1/condor/executable": [
                data["config_file"]["label1"]["condor"]["executable"],
                simlinked_dict["config_file"]["label1"]["condor"]["executable"]
            ],
            "config_file/label2/condor/executable": [
                data["config_file"]["label2"]["condor"]["executable"],
                simlinked_dict["config_file"]["label2"]["condor"]["executable"]
            ]
        },
        {
            "config_file/label1/condor/memory": [
                data["config_file"]["label1"]["condor"]["memory"],
                simlinked_dict["config_file"]["label1"]["condor"]["memory"]
            ],
            "config_file/label2/condor/memory": [
                data["config_file"]["label2"]["condor"]["memory"],
                simlinked_dict["config_file"]["label2"]["condor"]["memory"]
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
    with h5py.File("./.outdir/test.h5") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(
            f, simlinked_dict, extra_keys=meta_file.DEFAULT_HDF5_KEYS)

    with h5py.File("./.outdir/no_softlink.h5") as f:
        meta_file.recursively_save_dictionary_to_hdf5_file(
            f, data, extra_keys=meta_file.DEFAULT_HDF5_KEYS)

    softlink_size = os.stat("./.outdir/test.h5").st_size
    no_softlink_size = os.stat('./.outdir/no_softlink.h5').st_size
    assert softlink_size < no_softlink_size

    with h5py.File("./.outdir/test.h5", "r") as f:
        assert \
            f["config_file"]["label2"]["condor"]["executable"][0] == \
            f["config_file"]["label1"]["condor"]["executable"][0]
        assert \
            all(
                i == j for i, j in zip(
                    f["psds"]["label1"]["H1"][0], f["psds"]["label1"]["L1"][0]
                )
            )
        assert \
            all(
                i == j for i, j in zip(
                    f["psds"]["label1"]["H1"][1], f["psds"]["label1"]["L1"][1]
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
        self.input_config = ["./tests/files/config_lalinference.ini"]
        psd_data = GWInput.extract_psd_data_from_file("./tests/files/psd_file.txt")
        self.psds = {"EXP1": {"H1": psd_data}}
        calibration_data = GWInput.extract_calibration_data_from_file(
            "./tests/files/calibration_envelope.txt")
        self.calibration = {"EXP1": {"H1": calibration_data}}

        object = _GWMetaFile(
            self.input_data, self.input_labels, self.input_config,
            self.input_injection, self.input_file_version, self.input_file_kwargs,
            webdir=".outdir", psd=self.psds, calibration=self.calibration)
        object.make_dictionary()
        object.save_to_json()
        object = _GWMetaFile(
            self.input_data, self.input_labels, self.input_config,
            self.input_injection, self.input_file_version, self.input_file_kwargs,
            webdir=".outdir", psd=self.psds, calibration=self.calibration,
            hdf5=True)
        object.make_dictionary()
        object.save_to_hdf5()

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
            assert list(data["posterior_samples"].keys()) == self.input_labels
            if num == 0:
                assert list(
                    sorted(data["posterior_samples"]["EXP1"].keys())) == [
                        "parameter_names", "samples"]
            if num == 0:
                try:
                    assert all(
                        i.decode("utf-8") == j for i, j in zip(
                            sorted(data["posterior_samples"]["EXP1"]["parameter_names"]),
                            sorted(self.input_parameters)))
                except AttributeError:
                    assert all(
                        i == j for i, j in zip(
                            sorted(data["posterior_samples"]["EXP1"]["parameter_names"]),
                            sorted(self.input_parameters)))
            else:
                try:
                    assert all(
                        i.decode("utf-8") == j for i, j in zip(
                            sorted(data["posterior_samples"]["EXP1"].dtype.names),
                            sorted(self.input_parameters)))
                except AttributeError:
                    assert all(
                        i == j for i, j in zip(
                            sorted(data["posterior_samples"]["EXP1"].dtype.names),
                            sorted(self.input_parameters)))

    def test_samples(self):
        """Test the samples stored in the metafile
        """
        for num, data in enumerate([self.json_file, self.hdf5_file]):
            if num == 0:
                parameters = data["posterior_samples"]["EXP1"]["parameter_names"]
                samples = np.array(data["posterior_samples"]["EXP1"]["samples"]).T
            else:
                parameters = [j for j in data["posterior_samples"]["EXP1"].dtype.names]
                samples = np.array([j.tolist() for j in data["posterior_samples"]["EXP1"]]).T
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
            for i, j in zip(data["version"]["EXP1"], [self.input_file_version["EXP1"]]):
                version = i
                if isinstance(i, bytes):
                    version = version.decode("utf-8")
                assert version == j

    def test_meta_data(self):
        """Test the meta data stored in the metafile
        """
        for num, data in enumerate([self.json_file, self.hdf5_file]):
            assert list(data["meta_data"].keys()) == self.input_labels
            assert sorted(
                list(data["meta_data"]["EXP1"].keys())) == ["meta_data", "sampler"]
            assert all(
               all(
                   k == l for k, l in zip(
                       self.input_file_kwargs["EXP1"][i],
                       data["meta_data"]["EXP1"][j]
                       )
               ) for i, j in zip(
                   sorted(self.input_file_kwargs["EXP1"].keys()),
                   sorted(data["meta_data"]["EXP1"].keys())
               )
            )

            try:
                assert all(
                    all(
                        self.input_file_kwargs["EXP1"][i][k] == data["meta_data"]["EXP1"][j][l]
                        for k, l in zip(
                            self.input_file_kwargs["EXP1"][i],
                            data["meta_data"]["EXP1"][j]
                            )
                    ) for i, j in zip(
                        sorted(self.input_file_kwargs["EXP1"].keys()),
                        sorted(data["meta_data"]["EXP1"].keys())
                    )
                )
            except Exception:
                assert all(
                    all(
                        self.input_file_kwargs["EXP1"][i][k] == data["meta_data"]["EXP1"][j][l][0]
                        for k, l in zip(
                            self.input_file_kwargs["EXP1"][i],
                            data["meta_data"]["EXP1"][j]
                            )
                    ) for i, j in zip(
                        sorted(self.input_file_kwargs["EXP1"].keys()),
                        sorted(data["meta_data"]["EXP1"].keys())
                    )
                )

    def test_psd(self):
        """Test the psd is stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert list(data["psds"].keys()) == self.input_labels
            assert list(
                data["psds"]["EXP1"].keys()) == ["H1"]
            for i, j in zip(self.psds["EXP1"]["H1"], data["psds"]["EXP1"]["H1"]):
                for k, l in zip(i, j):
                    assert k == l

    def test_calibration(self):
        """Test the calibration envelope is stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert list(data["calibration_envelope"].keys()) == self.input_labels
            assert list(
                data["calibration_envelope"]["EXP1"].keys()) == ["H1"]
            for i, j in zip(self.calibration["EXP1"]["H1"], data["calibration_envelope"]["EXP1"]["H1"]):
                for k, l in zip(i, j):
                    assert k == l

    def test_config(self):
        """Test the configuration file is stored in the metafile
        """
        import configparser

        config_data = []
        for i in self.input_config:
            config = configparser.ConfigParser()
            config.read(i)
            config_data.append(config)

        for num, data in enumerate([self.json_file, self.hdf5_file]):
            assert list(data["config_file"].keys()) == self.input_labels
            assert all(
                i == j for i, j in zip(
                    sorted(list(config_data[0].sections())),
                    sorted(list(data["config_file"]["EXP1"].keys()))))
            all_options = {
                i: {
                    j: k for j, k in config_data[0][i].items()
                } for i in config_data[0].sections()
            }

            assert all(
                all(
                    k == l for k, l in zip(
                        sorted(all_options[i]),
                        sorted(data["config_file"]["EXP1"][j])
                    )
                ) for i, j in zip(
                    sorted(list(all_options.keys())),
                    sorted(list(data["config_file"]["EXP1"].keys()))
                )
            )

            if num == 0:
                assert all(
                    all(
                        all_options[i][k] == data["config_file"]["EXP1"][j][l]
                        for k, l in zip(
                            sorted(all_options[i]),
                            sorted(data["config_file"]["EXP1"][j])
                        )
                    ) for i, j in zip(
                        sorted(list(all_options.keys())),
                        sorted(list(data["config_file"]["EXP1"].keys()))
                    )
                )
            if num == 1:
                assert all(
                    all(
                        all_options[i][k] == data["config_file"]["EXP1"][j][l][0].decode("utf-8")
                        for k, l in zip(
                            sorted(all_options[i]),
                            sorted(data["config_file"]["EXP1"][j])
                        )
                    ) for i, j in zip(
                        sorted(list(all_options.keys())),
                        sorted(list(data["config_file"]["EXP1"].keys()))
                    )
                )

    def test_injection_data(self):
        """Test the injection data stored in the metafile
        """
        for data in [self.json_file, self.hdf5_file]:
            assert list(data["injection_data"].keys()) == self.input_labels
            for num, i in enumerate(list(self.input_injection["EXP1"].keys())):
                assert self.input_injection["EXP1"][i] == data["injection_data"]["EXP1"]["injection_values"][num]
