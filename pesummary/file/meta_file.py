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

from pesummary.utils.utils import logger
from pesummary.inputs import PostProcessing
from pesummary.file.existing import ExistingFile
from pesummary.utils.utils import make_dir


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
        if "psds" in dictionary.keys():
            f.create_group("psds")
        if "calibration_envelope" in dictionary.keys():
            f.create_group("calibration_envelope")
        if "config_file" in dictionary.keys():
            f.create_group("config_file")
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
            if isinstance(v[0], str):
                f["/".join(current_path)].create_dataset(k, data=np.array(
                    v, dtype="S"
                ))
            elif isinstance(v[0], list):
                f["/".join(current_path)].create_dataset(k, data=np.array(v))
        elif isinstance(v, (str, bytes)):
            f["/".join(current_path)].create_dataset(k, data=np.array(
                [v], dtype="S"
            ))


class MetaFile(PostProcessing):
    """This class handles the creation of a meta file storing all information
    from the analysis

    Attributes
    ----------
    meta_file: str
        name of the meta file storing all information
    """
    def __init__(self, inputs):
        super(MetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        self.data = {}
        self.existing_label = None
        self.existing_approximant = None
        self.existing_parameters = None
        self.existing_samples = None
        self.generate_meta_file_data()

        if not self.hdf5:
            self.save_to_json()
        else:
            self.save_to_hdf5()

        self.generate_dat_file()
        logger.info("Finished generating the meta file. The meta file can be "
                    "viewed here: %s" % (self.meta_file))

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
        return data

    def generate_meta_file_data(self):
        """Generate dictionary of data which will go into the meta_file
        """
        if self.existing:
            existing_file = ExistingFile(self.existing)
            self.existing_parameters = existing_file.existing_parameters
            self.existing_samples = existing_file.existing_samples
            self.existing_approximant = existing_file.existing_approximant
            self.existing_label = existing_file.existing_labels
        self._make_dictionary()

    def _make_dictionary(self):
        if self.existing_label:
            self._make_dictionary_structure(
                self.existing_label, self.existing_approximant
            )
            for num, i in enumerate(self.existing_label):
                self._add_data(i, self.existing_approximant[num],
                               self.existing_parameters[num],
                               self.existing_samples[num],
                               )
        self._make_dictionary_structure(self.labels, self.approximant,
                                        psd=self.psds,
                                        calibration=self.calibration,
                                        config=self.config
                                        )
        for num, i in enumerate(self.labels):
            psd = self._grab_psd_data_from_data_files(
                self.psds, self.psd_labels) if self.psds else None
            calibration = self._grab_calibration_data_from_data_files(
                self.calibration, self.calibration_labels) if self.calibration \
                else None
            config = self._grab_config_data_from_data_file(self.config[num]) if \
                self.config and num < len(self.config) else None
            self._add_data(i, self.approximant[num], self.parameters[num],
                           self.samples[num], psd=psd, calibration=calibration,
                           config=config
                           )

    def _grab_psd_data_from_data_files(self, files, psd_labels):
        """Return the psd data as a dictionary

        Parameters
        ----------
        files: list
            list of psd files
        psd_labels: list
            list of labels associated with each psd file
        """
        return {psd_labels[num]: [[i, j] for i, j in zip(
            self._grab_frequencies_from_psd_data_file(f),
            self._grab_strains_from_psd_data_file(f)
        )] for num, f in enumerate(files)}

    def _grab_calibration_data_from_data_files(self, files, calibration_labels):
        """Return the calibration envelope data as a dictionary

        Parameters
        ----------
        files: list
            list of calibration envelope files
        calibration_labels:
            list of labels associated with each calibration envelope file
        """
        dictionary = {}
        for num, i in enumerate(calibration_labels):
            dictionary[i] = [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6]] for k in files[num]]
        return dictionary

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

    def _add_data(self, label, approximant, parameters, samples,
                  psd=None, calibration=None, config=None):
        """Add data to the stored dictionary

        Parameters
        ----------
        label: str
            the name of the second level in the dictionary
        approximant: str
            the name of the third level in the dictionary
        parameters: list
            list of parameters that were used in the study
        samples: list
            list of samples that wee used in th study
        psd: list, optional
            list of frequencies and strains associated with the psd
        calibration: list, optional
            list of data associated with the calibration envelope
        config: dict, optional
            data associated with the configuration file
        """
        self.data["posterior_samples"][label][approximant] = {
            "parameter_names": list(parameters),
            "samples": self.convert_to_list(samples)
        }
        if psd:
            for i in list(psd.keys()):
                self.data["psds"][label][approximant][i] = psd[i]
        if calibration:
            for i in list(calibration.keys()):
                self.data["calibration_envelope"][label][approximant][i] = \
                    calibration[i]
        if config:
            self.data["config_file"][label][approximant] = config

    def _make_dictionary_structure(self, label, approximant, psd=None,
                                   calibration=None, config=None):
        for num, i in enumerate(label):
            self._add_label_and_approximant(
                "posterior_samples", i, approximant[num]
            )
            if psd:
                self._add_label_and_approximant("psds", i, approximant[num])
            if calibration:
                self._add_label_and_approximant(
                    "calibration_envelope", i, approximant[num]
                )
            if config:
                self._add_label_and_approximant(
                    "config_file", i, approximant[num]
                )

    def _add_label_and_approximant(self, base_level, label, approximant):
        if base_level not in list(self.data.keys()):
            self.data[base_level] = {}
        if label not in list(self.data[base_level].keys()):
            self.data[base_level][label] = {}
        self.data[base_level][label][approximant] = {}

    def save_to_json(self):
        with open(self.meta_file, "w") as f:
            json.dump(self.data, f, indent=4, sort_keys=True)

    def save_to_hdf5(self):
        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)

    def generate_dat_file(self):
        """Generate a single .dat file that contains all the samples for a
        given analysis
        """
        self.savedir = "%s/samples/dat" % (self.webdir)
        if not os.path.isdir(self.savedir):
            make_dir(self.savedir)
        for num, i in enumerate(self.result_files):
            if "posterior_samples.h5" not in i:
                make_dir("%s/%s_%s" % (
                    self.savedir, self.labels[num], self.approximant[num]))
                for idx, j in enumerate(self.parameters[num]):
                    data = [k[idx] for k in self.samples[num]]
                    np.savetxt("%s/%s_%s/%s_%s_%s_samples.dat" % (
                        self.savedir, self.labels[num], self.approximant[num],
                        self.labels[num], self.approximant[num], j), data,
                        fmt="%s")
