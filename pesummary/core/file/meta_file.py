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
import json

import pesummary
from pesummary import __version__
from pesummary.core.inputs import PostProcessing
from pesummary.utils.utils import make_dir, logger
from pesummary import conf


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
            f["/".join(current_path)].create_dataset(k, data=np.array(
                [v], dtype='<f4'))
        elif v == {}:
            f["/".join(current_path)].create_dataset(k, data=np.array("NaN"))


class PESummaryJsonEncoder(json.JSONEncoder):
    """Personalised JSON encoder for PESummary
    """
    def default(self, obj):
        """Return a json serializable object for 'obj'

        Parameters
        ----------
        obj: object
            object you wish to make json serializable
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class _MetaFile(object):
    """This is a base class to handle the functions to generate a meta file
    """
    def __init__(
        self, samples, labels, config, injection_data, file_versions,
        file_kwargs, webdir=None, result_files=None, hdf5=False, priors={},
        existing_version=None, existing_label=None, existing_samples=None,
        existing_injection=None, existing_metadata=None, existing_config=None,
        existing_priors={}, existing_metafile=None, outdir=None, existing=None
    ):
        self.data = {}
        self.webdir = webdir
        self.result_files = result_files
        self.samples = samples
        self.labels = labels
        self.config = config
        self.injection_data = injection_data
        self.file_versions = file_versions
        self.file_kwargs = file_kwargs
        self.hdf5 = hdf5
        self.priors = priors
        self.existing_version = existing_version
        self.existing_labels = existing_label
        self.existing_samples = existing_samples
        self.existing_injection = existing_injection
        self.existing_file_kwargs = existing_metadata
        self.existing_config = existing_config
        self.existing_priors = existing_priors
        self.existing_metafile = existing_metafile
        self.existing = existing
        self.outdir = outdir

        if self.existing_labels is None:
            self.existing_labels = [None]
        if self.existing is not None:
            self.add_existing_data()

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, outdir):
        if outdir is not None:
            self._outdir = os.path.abspath(outdir)
        elif self.webdir is not None:
            self._outdir = os.path.join(self.webdir, "samples")
        else:
            raise Exception("Please provide an output directory for the data")

    @property
    def file_name(self):
        base = "posterior_samples.{}"
        if self.hdf5:
            return base.format("h5")
        return base.format("json")

    @property
    def meta_file(self):
        return os.path.join(self.outdir, self.file_name)

    def make_dictionary(self):
        """Wrapper function for _make_dictionary
        """
        self._make_dictionary()

    def _make_dictionary(self):
        """Generate a single dictionary which stores all information
        """
        dictionary = {
            "posterior_samples": {},
            "injection_data": {},
            "version": {},
            "meta_data": {},
            "priors": {},
            "config_file": {}
        }

        dictionary["priors"] = self.priors
        for num, label in enumerate(self.labels):
            parameters = self.samples[label].keys()
            samples = np.array([self.samples[label][i] for i in parameters]).T
            dictionary["posterior_samples"][label] = {
                "parameter_names": list(parameters),
                "samples": samples.tolist()
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
        self.data = dictionary

    @staticmethod
    def _grab_config_data_from_data_file(file):
        """Return the config data as a dictionary

        Parameters
        ----------
        file: str
            path to the configuration file
        """
        import configparser

        data = {}
        config = configparser.ConfigParser()
        try:
            config.read(file)
            sections = config.sections()
        except Exception as e:
            sections = None
            logger.info(
                "Unable to open %s because %s. The data will not be stored in "
                "the meta file" % (file, e)
            )
        if sections:
            for i in sections:
                data[i] = {}
                for key in config["%s" % (i)]:
                    data[i][key] = config["%s" % (i)]["%s" % (key)]
        return data

    @staticmethod
    def write_to_dat(file_name, samples, header=None):
        """Write samples to a .dat file

        Parameters
        ----------
        file_name: str
            the name of the file that you wish to write the samples to
        samples: np.ndarray
            1d/2d array of samples to write to file
        header: list, optional
            List of strings to write at the beginning of the file
        """
        np.savetxt(
            file_name, samples, delimiter=conf.delimiter,
            header=conf.delimiter.join(header), comments=""
        )

    def write_marginalized_posterior_to_dat(self):
        """Write the marginalized posterior for each parameter to a .dat file
        """
        for label in self.labels:
            if not os.path.isdir(os.path.join(self.outdir, label)):
                make_dir(os.path.join(self.outdir, label))
            for param, samples in self.samples[label].items():
                self.write_to_dat(
                    os.path.join(
                        self.outdir, label, "{}_{}.dat".format(label, param)
                    ), samples, header=[param]
                )

    def save_to_json(self):
        """Save the metafile as a json file
        """
        with open(self.meta_file, "w") as f:
            json.dump(self.data, f, indent=4, sort_keys=True, cls=PESummaryJsonEncoder)

    def save_to_hdf5(self):
        """Save the metafile as a hdf5 file
        """
        import h5py

        with h5py.File(self.meta_file, "w") as f:
            _recursively_save_dictionary_to_hdf5_file(f, self.data)

    def save_to_dat(self):
        """Save the samples to a .dat file
        """
        for label in self.labels:
            parameters = self.samples[label].keys()
            samples = np.array([self.samples[label][i] for i in parameters])
            self.write_to_dat(
                os.path.join(self.outdir, "{}_pesummary.dat".format(label)),
                samples.T, header=parameters
            )

    def add_existing_data(self):
        """
        """
        from pesummary.utils.utils import _add_existing_data

        self = _add_existing_data(self)


class MetaFile(PostProcessing):
    """This class handles the creation of a metafile storing all information
    from the analysis
    """
    def __init__(self, inputs):
        from pesummary.utils.utils import logger

        super(MetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        meta_file = _MetaFile(
            self.samples, self.labels, self.config,
            self.injection_data, self.file_version, self.file_kwargs,
            hdf5=self.hdf5, webdir=self.webdir, result_files=self.result_files,
            existing_version=self.existing_file_version, existing_label=self.existing_labels,
            existing_samples=self.existing_samples,
            existing_injection=self.existing_injection_data,
            existing_metadata=self.existing_file_kwargs,
            existing_config=self.existing_config, existing=self.existing,
            existing_priors=self.existing_priors,
            existing_metafile=self.existing_metafile
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
