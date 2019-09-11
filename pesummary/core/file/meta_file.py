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

import numpy as np
import h5py
import json
import configparser

import pesummary
from pesummary.utils.utils import logger, make_dir
from pesummary.core.file.read import read as Read
from pesummary.utils.utils import get_version_information


def _recursively_save_dictionary_to_hdf5_file(f, dictionary, current_path=None):
    """Recursively save a dictionary to a hdf5 file

    Parameters
    ----------
    f: h5py._hl.files.File
        the open hdf5 file that you would like the data to be saved to
    dictionary: dict
        dictionary of data
    current_path: optional, str
        string to indicate the level to save the data in the hdf5 file
    """
    try:
        f.create_group("posterior_samples")
        if "config_file" in dictionary.keys():
            f.create_group("config_file")
        if "injection_data" in dictionary.keys():
            f.create_group("injection_data")
        if "version" in dictionary.keys():
            f.create_group("version")
        if "meta_data" in dictionary.keys():
            f.create_group("meta_data")
    except Exception:
        pass
    if current_path is None:
        current_path = []

    for k, v in dictionary.items():
        if isinstance(v, dict):
            try:
                f["/".join(current_path)].create_group(k)
            except Exception:
                pass
            path = current_path + [k]
            _recursively_save_dictionary_to_hdf5_file(f, v, path)
        elif isinstance(v, list):
            import math

            if isinstance(v[0], str):
                f["/".join(current_path)].create_dataset(k, data=np.array(
                    v, dtype="S"
                ))
            elif isinstance(v[0], list):
                f["/".join(current_path)].create_dataset(k, data=np.array(v))
            elif math.isnan(v[0]):
                f["/".join(current_path)].create_dataset(k, data=np.array(
                    ["NaN"] * len(v), dtype="S"
                ))
            elif isinstance(v[0], float):
                f["/".join(current_path)].create_dataset(k, data=np.array(v))
        elif isinstance(v, (str, bytes)):
            f["/".join(current_path)].create_dataset(k, data=np.array(
                [v], dtype="S"
            ))
        elif isinstance(v, float):
            f["/".join(current_path)].create_dataset(k, data=np.array(
                [v], dtype='<f4'))
        elif v == {}:
            f["/".join(current_path)].create_dataset(k, data=np.array("NaN"))


class _MetaFile(object):
    """This is a base class to handle the functions to generate a meta file
    """
    def __init__(self, parameters, samples, labels, config,
                 injection_data, file_versions, file_kwargs,
                 webdir=None, result_files=None, hdf5=False,
                 existing_version=None, existing_label=None,
                 existing_parameters=None, existing_samples=None,
                 existing_injection=None, existing_metadata=None,
                 existing_config=None):
        self.data = {}
        self.webdir = webdir
        self.result_files = result_files
        self.parameters = parameters
        self.samples = samples
        self.labels = labels
        self.config = config
        self.injection_data = injection_data
        self.file_versions = file_versions
        self.file_kwargs = file_kwargs
        self.hdf5 = hdf5
        self.existing_version = existing_version
        self.existing_label = existing_label
        self.existing_parameters = existing_parameters
        self.existing_samples = existing_samples
        self.existing_injection = existing_injection
        self.existing_metadata = existing_metadata
        self.existing_config = existing_config

        if self.existing_label is None:
            self.existing_label = [None]

    @property
    def meta_file(self):
        if not self.hdf5:
            return self.webdir + "/samples/posterior_samples.json"
        return self.webdir + "/samples/posterior_samples.h5"

    @staticmethod
    def convert_to_list(data):
        for num, i in enumerate(data):
            if isinstance(data[num], np.ndarray):
                data[num] = list(i)
            for idx, j in enumerate(i):
                if isinstance(data[num][idx], np.ndarray):
                    data[num][idx] = list(i)
            data[num] = [float(j) for j in data[num]]
        return data

    def generate_meta_file_data(self):
        """Generate dictionary of data which will go into the meta_file
        """
        self._make_dictionary()

    def _make_dictionary(self):
        """
        """
        if self.existing_parameters is not None:
            self._make_dictionary_structure(
                self.existing_label, config=self.existing_config
            )
            for num, i in enumerate(self.existing_label):
                self._add_data(i, self.existing_parameters[num],
                               self.existing_samples[num],
                               self.existing_injection[num],
                               version=self.existing_version[num],
                               config=self.existing_config,
                               meta_data=self.existing_metadata[num]
                               )
        self._make_dictionary_structure(self.labels, config=self.config
                                        )
        pesummary_version = get_version_information()

        for num, i in enumerate(self.labels):
            if i not in self.existing_label:
                injection = [self.injection_data[num]["%s" % (i)] for i in
                             self.parameters[num]]
                if self.config and not isinstance(self.config[num], dict):
                    config = self._grab_config_data_from_data_file(self.config[num]) \
                        if self.config and num < len(self.config) else None
                elif self.config:
                    config = self.config[num]
                else:
                    config = None
                self._add_data(i, self.parameters[num], self.samples[num],
                               injection, version=self.file_versions[num],
                               config=config, pesummary_version=pesummary_version,
                               meta_data=self.file_kwargs[num]
                               )

    def _grab_config_data_from_data_file(self, file):
        """Return the config data as a dictionary

        Parameters
        ----------
        file: str
            path to the configuration file
        """
        data = {}
        config = configparser.ConfigParser()
        try:
            config.read(file)
            sections = config.sections()
        except Exception as e:
            sections = None
            logger.info("Unable to open %s because %s. The data will not be "
                        "stored in the meta file" % (file, e))
        if sections:
            for i in sections:
                data[i] = {}
                for key in config["%s" % (i)]:
                    data[i][key] = config["%s" % (i)]["%s" % (key)]
        return data

    def _add_data(self, label, parameters, samples, injection, version,
                  config=None, pesummary_version=None, meta_data=None):
        """Add data to the stored dictionary

        Parameters
        ----------
        label: str
            the name of the second level in the dictionary
        parameters: list
            list of parameters that were used in the study
        samples: list
            list of samples that wee used in th study
        version: list
            list of versions for each result file used in the study
        config: dict, optional
            data associated with the configuration file
        """
        self.data["posterior_samples"][label] = {
            "parameter_names": list(parameters),
            "samples": self.convert_to_list(samples)
        }
        self.data["injection_data"][label] = {
            "injection_values": list(injection)
        }
        self.data["version"][label] = [version]
        self.data["meta_data"][label] = meta_data

        if config:
            self.data["config_file"][label] = config
        if pesummary_version:
            self.data["version"]["pesummary"] = [pesummary_version]

    def _make_dictionary_structure(self, label, config=None):
        for num, i in enumerate(label):
            self._add_label(
                "posterior_samples", i,
            )
            self._add_label(
                "injection_data", i,
            )
            self._add_label(
                "version", i
            )
            self._add_label(
                "meta_data", i
            )
            if config:
                self._add_label(
                    "config_file", i
                )

    def _add_label(self, base_level, label):
        if base_level not in list(self.data.keys()):
            self.data[base_level] = {}
        if label not in list(self.data[base_level].keys()):
            self.data[base_level][label] = {}

    def save_to_json(self):
        if self.webdir is None:
            raise Exception("No web directory has been provided")
        with open(self.meta_file, "w") as f:
            json.dump(self.data, f, indent=4, sort_keys=True)

    def save_to_hdf5(self):
        if self.webdir is None:
            raise Exception("No web dirctory has been provided")
        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)

    def generate_dat_file(self):
        """Generate a single .dat file that contains all the samples for a
        given analysis
        """
        if self.webdir is None:
            raise Exception("No web directory has been provided")
        self.savedir = "%s/samples/dat" % (self.webdir)
        if not os.path.isdir(self.savedir):
            make_dir(self.savedir)
        if not self.result_files:
            raise Exception("Unable to generate dat files for parameters "
                            "because no result files have been passed")
        for num, i in enumerate(self.result_files):
            if "posterior_samples.h5" not in i:
                make_dir("%s/%s" % (
                    self.savedir, self.labels[num]))
                for idx, j in enumerate(self.parameters[num]):
                    data = [k[idx] for k in self.samples[num]]
                    np.savetxt("%s/%s/%s_%s_%s_samples.dat" % (
                        self.savedir, self.labels[num],
                        self.labels[num], self.result_files[num].split("/")[-1], j), data,
                        fmt="%s")


class MetaFile(pesummary.core.inputs.PostProcessing):
    """This class handles the creation of a meta file storing all information
    from the analysis
    """
    def __init__(self, inputs):
        super(MetaFile, self).__init__(inputs)
        logger.info("starting to generate the meta file")
        if self.add_to_existing:
            existing_file = Read(self.existing_meta_file)
            existing_parameters = existing_file.parameters
            existing_samples = existing_file.samples
            existing_labels = existing_file.labels
            existing_config = existing_file.config
            existing_injection = existing_file.injection_parameters
            existing_version = existing_file.input_version
            existing_metadata = existing_file.extra_kwargs
        else:
            existing_parameters = None
            existing_samples = None
            existing_labels = None
            existing_config = None
            existing_injection = None
            existing_version = None
            existing_metadata = None

        meta_file = _MetaFile(self.parameters, self.samples, self.labels,
                              self.config, self.injection_data,
                              self.file_versions, self.file_kwargs, hdf5=self.hdf5,
                              webdir=self.webdir, result_files=self.result_files,
                              existing_version=existing_version,
                              existing_label=existing_labels,
                              existing_parameters=existing_parameters,
                              existing_samples=existing_samples,
                              existing_injection=existing_injection,
                              existing_metadata=existing_metadata,
                              existing_config=existing_config)
        meta_file.generate_meta_file_data()

        if not self.hdf5:
            meta_file.save_to_json()
        else:
            meta_file.save_to_hdf5()

        meta_file.generate_dat_file()
        logger.info("finished generating the meta file. the meta file can be "
                    "viewed here: %s" % (meta_file.meta_file))
