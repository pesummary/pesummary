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

import numpy as np
import h5py
import configparser

from pesummary.utils.utils import logger
from pesummary.gw.inputs import GWPostProcessing
from pesummary.gw.file.existing import GWExistingFile
from pesummary.core.file.meta_file import MetaFile


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
        if "approximant" in dictionary.keys():
            f.create_group("approximant")
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


class GWMetaFile(GWPostProcessing, MetaFile):
    """This class handles the creation of a meta file storing all information
    from the analysis

    Attributes
    ----------
    meta_file: str
        name of the meta file storing all information
    """
    def __init__(self, inputs):
        super(GWMetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        self.data = {}
        self.existing_label = [None]
        self.existing_approximant = None
        self.existing_parameters = None
        self.existing_samples = None
        self.existing_psd = None
        self.existing_calibration = None
        self.existing_config = None

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
            existing_file = GWExistingFile(self.existing)
            self.existing_parameters = existing_file.existing_parameters
            self.existing_samples = existing_file.existing_samples
            self.existing_label = existing_file.existing_labels
            self.existing_psd = existing_file.existing_psd
            self.existing_calibration = existing_file.existing_calibration
            self.existing_config = existing_file.existing_config
            self.existing_approximant = existing_file.existing_approximant
        self._make_dictionary()

    def _make_dictionary(self):
        if self.existing:
            self._make_dictionary_structure(
                self.existing_label,
                psd=self.existing_psd,
                approx=self.existing_approximant,
                calibration=self.existing_calibration,
                config=self.existing_config
            )
            for num, i in enumerate(self.existing_label):
                self._add_data(i,
                               self.existing_parameters[num],
                               self.existing_samples[num],
                               approximant=self.existing_approximant[num],
                               psd=self.existing_psd,
                               calibration=self.existing_calibration,
                               config=self.existing_config
                               )

        self._make_dictionary_structure(self.labels,
                                        psd=self.psds,
                                        approx=self.approximant,
                                        calibration=self.calibration,
                                        config=self.config
                                        )
        for num, i in enumerate(self.labels):
            if i not in self.existing_label:
                psd_frequencies = self.psd_frequencies[num] if self.psds else \
                    None
                psd_strains = self.psd_strains[num] if self.psds else None
                psd = self._combine_psd_frequency_strain(
                    psd_frequencies, psd_strains, self.psd_labels[num]) if \
                    self.psds else None
                calibration_envelopes = self.calibration_envelopes[num] if \
                    self.calibration else None
                calibration = self._combine_calibration_envelopes(
                    calibration_envelopes, self.calibration_labels[num]) if \
                    self.calibration else None
                config = self._grab_config_data_from_data_file(self.config[num]) \
                    if self.config and num < len(self.config) else None
                approximant = self.approximant if self.approximant else \
                    [None] * len(self.samples)
                self._add_data(i, self.parameters[num],
                               self.samples[num], psd=psd,
                               calibration=calibration, config=config,
                               approximant=approximant[num]
                               )

    def _combine_psd_frequency_strain(self, frequencies, strains, psd_labels):
        """Return the psd data as a dictionary

        Parameters
        ----------
        frequencies: list
            list of frequencies for the different IFOs
        strains: list
            list of strains for the different IFOs
        psd_labels: list
            list of psd labels corresponding to the different frequencies
        """
        return {psd_labels[num]: [[j, k] for j, k in zip(i, strains[num])] for
                num, i in enumerate(frequencies)}

    def _combine_calibration_envelopes(self, calibration_envelopes,
                                       calibration_labels):
        """Return the calibration data as a dictionary

        Parameters
        ----------
        calibration_envelopes: list
            list of calibration envelopes for the different IFOs
        calibration_labels: list
            list of calibration labels corresponding to the different
            calibration envelopes
        """
        return {calibration_labels[num]: [
            [j[0], j[1], j[2], j[3], j[4], j[5], j[6]] for j in i] for num, i
            in enumerate(calibration_envelopes)}

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

    def _add_data(self, label, parameters, samples, approximant=None,
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
        self.data["posterior_samples"][label] = {
            "parameter_names": list(parameters),
            "samples": self.convert_to_list(samples)
        }
        if psd:
            for i in list(psd.keys()):
                self.data["psds"][label][i] = psd[i]
        if calibration:
            for i in list(calibration.keys()):
                self.data["calibration_envelope"][label][i] = \
                    calibration[i]
        if config:
            self.data["config_file"][label] = config
        if approximant:
            self.data["approximant"][label] = approximant

    def _make_dictionary_structure(self, label, psd=None, approx=None,
                                   calibration=None, config=None):
        for num, i in enumerate(label):
            self._add_label(
                "posterior_samples", i
            )
            if psd:
                self._add_label("psds", i)
            if calibration:
                self._add_label(
                    "calibration_envelope", i
                )
            if config:
                self._add_label(
                    "config_file", i,
                )
            if approx:
                self._add_label(
                    "approximant", i,
                )

    def save_to_hdf5(self):
        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)
