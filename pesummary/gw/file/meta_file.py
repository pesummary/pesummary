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
from pesummary.utils.utils import get_version_information
from pesummary.gw.inputs import GWPostProcessing
from pesummary.gw.file.read import read as GWRead
from pesummary.core.file.meta_file import _MetaFile


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
            f["/".join(current_path)].create_dataset(k, data=np.array([v]))
        elif v == {}:
            f["/".join(current_path)].create_dataset(k, data=np.array("NaN"))


class _GWMetaFile(_MetaFile):
    """This class handles the creation of a meta file storing all information
    from the analysis

    Attributes
    ----------
    meta_file: str
        name of the meta file storing all information
    """
    def __init__(self, parameters, samples, labels, config,
                 injection_data, file_versions, file_kwargs,
                 calibration_envelopes=None, calibration_labels=None,
                 psd=None, approximant=None, calibration=None,
                 webdir=None, result_files=None, hdf5=False,
                 existing_version=None, existing_label=None,
                 existing_parameters=None, existing_samples=None,
                 existing_psd=None, existing_calibration=None,
                 existing_approximant=None, existing_config=None,
                 existing_injection=None, existing_metadata=None):
        super(_GWMetaFile, self).__init__(
            parameters, samples, labels, config,
            injection_data, file_versions, file_kwargs,
            webdir=webdir, result_files=result_files, hdf5=hdf5,
            existing_version=existing_version, existing_label=existing_label,
            existing_parameters=existing_parameters,
            existing_samples=existing_samples,
            existing_injection=existing_injection,
            existing_metadata=existing_metadata,
            existing_config=existing_config)
        self.calibration = calibration
        self.psds = psd
        self.approximant = approximant
        self.existing_psd = existing_psd
        self.existing_calibration = existing_calibration
        self.existing_approximant = existing_approximant

    def _make_dictionary(self):
        if self.existing_parameters is not None:
            self._make_dictionary_structure(
                self.existing_label,
                version=self.existing_version,
                psd=self.existing_psd,
                approx=self.existing_approximant,
                calibration=self.existing_calibration,
                config=self.existing_config,
                meta_data=self.existing_metadata
            )
            for num, i in enumerate(self.existing_label):
                self._add_data(i,
                               self.existing_parameters[num],
                               self.existing_samples[num],
                               self.existing_injection[num],
                               version=self.existing_version[num],
                               approximant=self.existing_approximant[num],
                               psd=self.existing_psd,
                               calibration=self.existing_calibration,
                               config=self.existing_config,
                               meta_data=self.existing_metadata[num]
                               )

        self._make_dictionary_structure(self.labels,
                                        version=self.file_versions,
                                        psd=self.psds,
                                        approx=self.approximant,
                                        calibration=self.calibration,
                                        config=self.config,
                                        meta_data=self.file_kwargs
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
                approximant = self.approximant if self.approximant else \
                    [None] * len(self.samples)
                self._add_data(i, self.parameters[num],
                               self.samples[num], injection,
                               version=self.file_versions[num], psd=self.psds[num],
                               calibration=self.calibration[num], config=config,
                               approximant=approximant[num],
                               pesummary_version=pesummary_version,
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

    def _add_data(self, label, parameters, samples, injection, version=None,
                  approximant=None, psd=None, calibration=None, config=None,
                  pesummary_version=None, meta_data=None):
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
        self.data["injection_data"][label] = {
            "injection_values": list(injection)
        }
        self.data["version"][label] = [version]
        self.data["meta_data"][label] = meta_data
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
        if pesummary_version:
            self.data["version"]["pesummary"] = [pesummary_version]

    def _make_dictionary_structure(self, label, version=None, psd=None,
                                   approx=None, calibration=None, config=None,
                                   meta_data=None):
        for num, i in enumerate(label):
            self._add_label(
                "posterior_samples", i
            )
            self._add_label(
                "injection_data", i
            )
            self._add_label(
                "version", i
            )
            self._add_label(
                "meta_data", i
            )
            if psd:
                self._add_label("psds", i)
            if calibration:
                if type(calibration) == list and any(i for i in calibration):
                    self._add_label(
                        "calibration_envelope", i
                    )
                elif type(calibration) == dict:
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
        if self.webdir is None:
            raise Exception("No web dirctory has been provided")
        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)


class GWMetaFile(GWPostProcessing):
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
        if self.add_to_existing:
            existing_file = GWRead(self.existing_meta_file)
            existing_parameters = existing_file.parameters
            existing_samples = existing_file.samples
            existing_labels = existing_file.labels
            existing_psd = existing_file.psd
            existing_calibration = existing_file.calibration
            existing_config = existing_file.config
            existing_approximant = existing_file.approximant
            existing_injection = existing_file.injection_parameters
            existing_version = existing_file.input_version
            existing_metadata = existing_file.extra_kwargs
        else:
            existing_parameters = None
            existing_samples = None
            existing_labels = None
            existing_psd = None
            existing_calibration = None
            existing_config = None
            existing_approximant = None
            existing_injection = None
            existing_version = None
            existing_metadata = None

        calibration_list, psd_list = [], []
        for num, i in enumerate(self.labels):
            cond1 = num < len(self.calibration_envelopes)
            if self.calibration_envelopes and cond1:
                if self.calibration_envelopes[num] is not None:
                    calibration = self._combine_calibration_envelopes(
                        self.calibration_envelopes[num],
                        self.calibration_labels[num])
                    calibration_list.append(calibration)
                else:
                    calibration_list.append(None)
            else:
                calibration_list.append(None)

            if self.psds and num < len(self.psds):
                psd_frequencies = self.psd_frequencies[num]
                psd_strains = self.psd_strains[num]
                psd_list.append(self._combine_psd_frequency_strain(
                    psd_frequencies, psd_strains, self.psd_labels[num]))
            else:
                psd_list.append(None)

        meta_file = _GWMetaFile(self.parameters, self.samples, self.labels,
                                self.config, self.injection_data,
                                self.file_versions, self.file_kwargs,
                                calibration=calibration_list,
                                psd=psd_list, hdf5=self.hdf5,
                                webdir=self.webdir, result_files=self.result_files,
                                existing_version=existing_version,
                                existing_label=existing_labels,
                                existing_parameters=existing_parameters,
                                existing_samples=existing_samples,
                                existing_psd=existing_psd,
                                existing_calibration=existing_calibration,
                                existing_approximant=existing_approximant,
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

    @staticmethod
    def _combine_calibration_envelopes(calibration_envelopes, calibration_labels):
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

    @staticmethod
    def _combine_psd_frequency_strain(frequencies, strains, psd_labels):
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
