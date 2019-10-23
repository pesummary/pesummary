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

import pesummary
from pesummary import __version__
from pesummary.utils.utils import logger, make_dir
from pesummary.core.file.meta_file import _MetaFile
from pesummary.gw.inputs import GWPostProcessing


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
        elif isinstance(v, (list, pesummary.utils.utils.Array, np.ndarray)):
            import math

            if isinstance(v[0], str):
                f["/".join(current_path)].create_dataset(k, data=np.array(
                    v, dtype="S"
                ))
            elif isinstance(v[0], (list, pesummary.utils.utils.Array, np.ndarray)):
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
    def __init__(
        self, samples, labels, config, injection_data, file_versions,
        file_kwargs, calibration=None, psd=None, approximant=None, webdir=None,
        result_files=None, hdf5=False, existing_version=None,
        existing_label=None, existing_samples=None, existing_psd=None,
        existing_calibration=None, existing_approximant=None,
        existing_config=None, existing_injection=None,
        existing_metadata=None, priors={}, outdir=None, existing=None,
        existing_priors={}, existing_metafile=None
    ):
        self.calibration = calibration
        self.psds = psd
        self.approximant = approximant
        self.existing_psd = existing_psd
        self.existing_calibration = existing_calibration
        self.existing_approximant = existing_approximant
        super(_GWMetaFile, self).__init__(
            samples, labels, config, injection_data, file_versions,
            file_kwargs, webdir=webdir, result_files=result_files, hdf5=hdf5,
            priors=priors, existing_version=existing_version,
            existing_label=existing_label, existing_samples=existing_samples,
            existing_injection=existing_injection,
            existing_metadata=existing_metadata,
            existing_config=existing_config, existing_priors={}, outdir=outdir,
            existing=existing, existing_metafile=existing_metafile
        )
        if self.existing_labels is None:
            self.existing_labels = [None]
        if self.existing is not None:
            self.add_existing_data()

    def _make_dictionary(self):
        """Generate a single dictionary which stores all information
        """
        dictionary = {
            "posterior_samples": {},
            "injection_data": {},
            "version": {},
            "meta_data": {},
            "priors": {},
            "config_file": {},
            "approximant": {},
            "psds": {},
            "calibration_envelope": {}
        }

        dictionary["priors"] = self.priors
        for num, label in enumerate(self.labels):
            parameters = self.samples[label].keys()
            samples = np.array([self.samples[label][i] for i in parameters]).T
            dictionary["posterior_samples"][label] = {
                "parameter_names": list(parameters),
                "samples": samples
            }
            dictionary["injection_data"][label] = {
                "injection_values": [
                    self.injection_data[label][i] for i in parameters
                ]
            }
            dictionary["version"][label] = [self.file_versions[label]]
            dictionary["version"]["pesummary"] = [__version__]
            dictionary["meta_data"][label] = self.file_kwargs[label]

            if self.config != {} and self.config[num] is not None and not isinstance(self.config[num], dict):
                config = self._grab_config_data_from_data_file(self.config[num])
                dictionary["config_file"][label] = config
            elif self.config[num] is not None:
                dictionary["config_file"][label] = self.config[num]
            if self.calibration != {} and self.calibration[label] is not None:
                dictionary["calibration_envelope"][label] = {
                    key: item for key, item in self.calibration[label].items()
                    if item is not None
                }
            else:
                dictionary["calibration_envelope"][label] = {}
            if self.psds != {} and self.psds[label] is not None:
                dictionary["psds"][label] = {
                    key: item for key, item in self.psds[label].items() if item
                    is not None
                }
            else:
                dictionary["psds"][label] = {}
            if self.approximant is not None and self.approximant[label] is not None:
                dictionary["approximant"][label] = self.approximant[label]
            else:
                dictionary["approximant"][label] = {}
        self.data = dictionary

    def save_to_hdf5(self):
        """Save the metafile as a hdf5 file
        """
        import h5py

        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)


class GWMetaFile(GWPostProcessing):
    """This class handles the creation of a metafile storing all information
    from the analysis
    """
    def __init__(self, inputs):
        super(GWMetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        if self.add_to_existing:
            from pesummary.gw.file.read import read as GWRead

            existing = self.existing
            existing_metafile = self.existing_metafile
            existing_samples = self.existing_samples
            existing_labels = self.existing_labels
            existing_psd = self.existing_psd
            existing_calibration = self.existing_calibration
            existing_config = self.existing_config
            existing_approximant = self.existing_approximant
            existing_injection = self.existing_injection_data
            existing_version = self.existing_file_version
            existing_metadata = self.existing_file_kwargs
            existing_priors = self.existing_priors
        else:
            existing_metafile = None
            existing_samples = None
            existing_labels = None
            existing_psd = None
            existing_calibration = None
            existing_config = None
            existing_approximant = None
            existing_injection = None
            existing_version = None
            existing_metadata = None
            existing = None
            existing_priors = {}

        meta_file = _GWMetaFile(
            self.samples, self.labels, self.config, self.injection_data,
            self.file_version, self.file_kwargs, calibration=self.calibration,
            psd=self.psd, hdf5=self.hdf5, webdir=self.webdir,
            result_files=self.result_files, existing_version=existing_version,
            existing_label=existing_labels, existing_samples=existing_samples,
            existing_psd=existing_psd, existing_calibration=existing_calibration,
            existing_approximant=existing_approximant,
            existing_injection=existing_injection,
            existing_metadata=existing_metadata,
            existing_config=existing_config, priors=self.priors,
            existing_priors=existing_priors, existing=existing,
            existing_metafile=existing_metafile, approximant=self.approximant
        )
        meta_file.make_dictionary()
        if not self.hdf5:
            meta_file.save_to_json()
        else:
            meta_file.save_to_hdf5()
        meta_file.save_to_dat()
        meta_file.write_marginalized_posterior_to_dat()
        logger.info(
            "Finishing generating the meta file. The meta file can be viewed "
            "here: {}".format(meta_file.meta_file)
        )
