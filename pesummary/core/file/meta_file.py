# Licensed under an MIT style license -- see LICENSE.md

import inspect
import os
import numpy as np
import json
import copy

import pesummary
from pesummary import __version__
from pesummary.core.inputs import PostProcessing
from pesummary.utils.dict import Dict
from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.utils import make_dir, logger
from pesummary.utils.decorators import open_config
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
DEFAULT_HDF5_KEYS = ["version", "history"]


def recursively_save_dictionary_to_hdf5_file(
    f, dictionary, current_path=None, extra_keys=DEFAULT_HDF5_KEYS,
    compression=None
):
    """Recursively save a dictionary to a hdf5 file

    Parameters
    ----------
    f: h5py._hl.files.File
        the open hdf5 file that you would like the data to be saved to
    dictionary: dict
        dictionary of data
    current_path: optional, str
        string to indicate the level to save the data in the hdf5 file
    compression: int, optional
        optional filter to apply for compression. If you do not want to
        apply compression, compression = None. Default None.
    """
    def _safe_create_hdf5_group(hdf5_file, key):
        if key not in hdf5_file.keys():
            hdf5_file.create_group(key)
    for key in extra_keys:
        if key in dictionary:
            _safe_create_hdf5_group(hdf5_file=f, key=key)
    if current_path is None:
        current_path = []

    for k, v in dictionary.items():
        if isinstance(v, dict):
            if k not in f["/" + "/".join(current_path)].keys():
                f["/".join(current_path)].create_group(k)
            path = current_path + [k]
            recursively_save_dictionary_to_hdf5_file(
                f, v, path, extra_keys=extra_keys, compression=compression
            )
        else:
            if isinstance(dictionary, Dict):
                attrs = dictionary.extra_kwargs
            else:
                attrs = {}
            create_hdf5_dataset(
                key=k, value=v, hdf5_file=f, current_path=current_path,
                compression=compression, attrs=attrs
            )


def create_hdf5_dataset(
    key, value, hdf5_file, current_path, compression=None, attrs={}
):
    """
    Create a hdf5 dataset in place

    Parameters
    ----------
    key: str
        Key for the new dataset
    value: array-like, str
        Data to store. If you wish to create a softlink to another dataset
        then value should be a string in the form `softlink:/path/to/dataset`
        where `/path/to/dataset/` is the path to the dataset which you wish to
        create a softlink to
    hdf5_file: h5py.File
        hdf5 file object
    current_path: str
        Current string withing the hdf5 file
    compression: int, optional
        optional filter to apply for compression. If you do not want to
        apply compression, compression = None. Default None.
    attrs: dict, optional
        optional list of attributes to store alongside the dataset
    """
    error_message = "Cannot process {}={} from list with type {} for hdf5"
    array_types = (list, pesummary.utils.samples_dict.Array, np.ndarray)
    numeric_types = (float, int, np.number)
    string_types = (str, bytes)
    SOFTLINK = False

    if isinstance(value, array_types):
        import math

        if not len(value):
            data = np.array([])
        elif isinstance(value[0], string_types):
            data = np.array(value, dtype="S")
        elif isinstance(value[0], array_types):
            data = np.array(np.vstack(value))
        elif isinstance(value[0], (tuple, np.record, np.recarray)):
            data = value
        elif all(isinstance(_value, (bool, np.bool_)) for _value in value):
            data = np.array([str(_value) for _value in value], dtype="S")
        elif all(_value is None for _value in value):
            data = np.array(["None"] * len(value), dtype="S")
        elif isinstance(value[0], np.void) and value.dtype.names:
            data = value
        elif math.isnan(value[0]):
            data = np.array(["NaN"] * len(value), dtype="S")
        elif isinstance(value[0], numeric_types):
            data = np.array(value)
        else:
            raise TypeError(error_message.format(key, value, type(value[0])))
    elif isinstance(value, string_types[0]) and "softlink:" in value:
        import h5py

        SOFTLINK = True
        hdf5_file["/".join(current_path + [key])] = h5py.SoftLink(
            value.split("softlink:")[1]
        )
    elif isinstance(value, string_types[0]) and "external:" in value:
        import h5py

        SOFTLINK = True
        substring = value.split("external:")[1]
        _file, _path = substring.split("|")
        hdf5_file["/".join(current_path + [key])] = h5py.ExternalLink(
            _file, _path
        )
    elif isinstance(value, string_types):
        data = np.array([value], dtype="S")
    elif isinstance(value, numeric_types):
        data = np.array([value])
    elif isinstance(value, (bool, np.bool_)):
        data = np.array([str(value)], dtype="S")
    elif isinstance(value, np.complex):
        key += "_amp"
        data = np.array(np.abs(value))
    elif value == {}:
        data = np.array(np.array("NaN"))
    elif inspect.isclass(value) or inspect.isfunction(value):
        data = np.array([value.__module__ + value.__name__], dtype="S")
    elif inspect.ismodule(value):
        data = np.array([value.__name__], dtype="S")
    elif value is None:
        data = np.array(["None"], dtype="S")
    else:
        raise TypeError(error_message.format(key, value, type(value)))
    if not SOFTLINK:
        if compression is not None and len(data) > conf.compression_min_length:
            kwargs = {"compression": "gzip", "compression_opts": compression}
        else:
            kwargs = {}
        try:
            dset = hdf5_file["/".join(current_path)].create_dataset(
                key, data=data, **kwargs
            )
        except ValueError:
            dset = hdf5_file.create_dataset(key, data=data, **kwargs)
        if len(attrs):
            dset.attrs.update(attrs)


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
        if inspect.isfunction(obj):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool, np.bool_, bool)):
            return str(obj)
        elif isinstance(obj, bytes):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class _MetaFile(object):
    """This is a base class to handle the functions to generate a meta file
    """
    def __init__(
        self, samples, labels, config, injection_data, file_versions,
        file_kwargs, webdir=None, result_files=None, hdf5=False, priors={},
        existing_version=None, existing_label=None, existing_samples=None,
        existing_injection=None, existing_metadata=None, existing_config=None,
        existing_priors={}, existing_metafile=None, outdir=None, existing=None,
        package_information={}, mcmc_samples=False, filename=None,
        external_hdf5_links=False, hdf5_compression=None, history=None,
        descriptions=None
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
        self.file_name = filename
        self.external_hdf5_links = external_hdf5_links
        self.hdf5_compression = hdf5_compression
        self.history = history
        self.descriptions = descriptions
        if self.history is None:
            from pesummary.utils.utils import history_dictionary

            try:
                _user = os.environ["USER"]
            except KeyError:
                _user = ''
            self.history = history_dictionary(creator=_user)
        if self.descriptions is None:
            self.descriptions = {
                label: "No description found" for label in self.labels
            }
        elif not all(label in self.descriptions.keys() for label in self.labels):
            for label in self.labels:
                if label not in self.descriptions.keys():
                    self.descriptions[label] = "No description found"
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
        self.package_information = package_information
        if not len(package_information):
            from pesummary.core.inputs import _Input
            self.package_information = _Input.get_package_information()
        self.mcmc_samples = mcmc_samples

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
        return self._file_name

    @file_name.setter
    def file_name(self, file_name):
        if file_name is not None:
            self._file_name = file_name
        else:
            base = "posterior_samples.{}"
            if self.hdf5:
                self._file_name = base.format("h5")
            else:
                self._file_name = base.format("json")

    @property
    def meta_file(self):
        return os.path.join(os.path.abspath(self.outdir), self.file_name)

    def make_dictionary(self):
        """Wrapper function for _make_dictionary
        """
        self._make_dictionary()

    @property
    def _dictionary_structure(self):
        if self.mcmc_samples:
            posterior = "mcmc_chains"
        else:
            posterior = "posterior_samples"
        dictionary = {
            label: {
                posterior: {}, "injection_data": {}, "version": {},
                "meta_data": {}, "priors": {}, "config_file": {}
            } for label in self.labels
        }
        dictionary["version"] = self.package_information
        dictionary["version"]["pesummary"] = [__version__]
        dictionary["history"] = self.history
        return dictionary

    def _make_dictionary(self):
        """Generate a single dictionary which stores all information
        """
        if self.mcmc_samples:
            posterior = "mcmc_chains"
        else:
            posterior = "posterior_samples"
        dictionary = self._dictionary_structure
        if self.file_kwargs is not None and isinstance(self.file_kwargs, dict):
            if "webpage_url" in self.file_kwargs.keys():
                dictionary["history"]["webpage_url"] = self.file_kwargs["webpage_url"]
            else:
                dictionary["history"]["webpage_url"] = "None"
        for num, label in enumerate(self.labels):
            parameters = self.samples[label].keys()
            samples = np.array([self.samples[label][i] for i in parameters]).T
            dictionary[label][posterior] = {
                "parameter_names": list(parameters), "samples": samples.tolist()
            }
            dictionary[label]["injection_data"] = {
                "parameters": list(parameters),
                "samples": [
                    self.injection_data[label][i] for i in parameters
                ]
            }
            dictionary[label]["version"] = [self.file_versions[label]]
            dictionary[label]["description"] = [self.descriptions[label]]
            dictionary[label]["meta_data"] = self.file_kwargs[label]
            if self.config != {} and self.config[num] is not None and \
                    not isinstance(self.config[num], dict):
                config = self._grab_config_data_from_data_file(self.config[num])
                dictionary[label]["config_file"] = config
            elif self.config[num] is not None:
                dictionary[label]["config_file"] = self.config[num]
            for key in self.priors.keys():
                if label in self.priors[key].keys():
                    dictionary[label]["priors"][key] = self.priors[key][label]
        self.data = dictionary

    @staticmethod
    @open_config(index=0)
    def _grab_config_data_from_data_file(file):
        """Return the config data as a dictionary

        Parameters
        ----------
        file: str
            path to the configuration file
        """
        config = file
        sections = config.sections()
        data = {}
        if config.error:
            logger.info(
                "Unable to open %s with configparser because %s. The data will "
                "not be stored in the meta file" % (
                    config.path_to_file, config.error
                )
            )
        if sections != []:
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

    @staticmethod
    def _convert_posterior_samples_to_numpy(
        dictionary, mcmc_samples=False, index=None
    ):
        """Convert the posterior samples from a column-major dictionary
        to a row-major numpy array

        Parameters
        ----------
        dictionary: dict
            dictionary of posterior samples to convert to a numpy array.
        mcmc_samples: Bool, optional
            if True, the dictionary contains seperate mcmc chains

        Examples
        --------
        >>> dictionary = {"mass_1": [1,2,3], "mass_2": [1,2,3]}
        >>> dictionry = _Metafile._convert_posterior_samples_to_numpy(
        ...     dictionary
        ... )
        >>> print(dictionary)
        ... rec.array([(1., 1.), (2., 2.), (3., 3.)],
        ...           dtype=[('mass_1', '<f4'), ('mass_2', '<f4')])
        """
        samples = copy.deepcopy(dictionary)
        if mcmc_samples:
            parameters = list(samples.keys())
            chains = samples[parameters[0]].keys()
            data = {
                key: SamplesDict({
                    param: samples[param][key] for param in parameters
                }) for key in chains
            }
            return {
                key: item.to_structured_array() for key, item in data.items()
            }
        return samples.to_structured_array(index=index)

    @staticmethod
    def _create_softlinks(dictionary):
        """Identify duplicated entries in a dictionary and replace them with
        `softlink:/path/to/existing_dataset`. This is required for creating
        softlinks when saved in hdf5 format

        Parameters
        ----------
        dictionary: dict
            nested dictionary of data
        """
        try:
            from pandas.io.json._normalize import nested_to_record
        except ImportError:
            from pandas.io.json.normalize import nested_to_record

        def modify_dict(key, dictionary, replace):
            """
            """
            from functools import reduce
            from operator import getitem

            mod = copy.deepcopy(dictionary)
            key_list = key.split("/")
            reduce(getitem, key_list[:-1], mod)[key_list[-1]] = replace
            return mod

        data = copy.deepcopy(dictionary)
        flat_dictionary = nested_to_record(data, sep='/')
        rev_dictionary = {}
        for key, value in flat_dictionary.items():
            try:
                rev_dictionary.setdefault(value, set()).add(key)
            except TypeError:
                rev_dictionary.setdefault(str(value), set()).add(key)

        for key, values in rev_dictionary.items():
            if len(values) > 1:
                tmp = list(values)
                for val in tmp[1:]:
                    data = modify_dict(val, data, "softlink:/{}".format(tmp[0]))
        return data

    def write_marginalized_posterior_to_dat(self):
        """Write the marginalized posterior for each parameter to a .dat file
        """
        if self.mcmc_samples:
            return
        for label in self.labels:
            if not os.path.isdir(os.path.join(self.outdir, label)):
                make_dir(os.path.join(self.outdir, label))
            for param, samples in self.samples[label].items():
                self.write_to_dat(
                    os.path.join(
                        self.outdir, label, "{}_{}.dat".format(label, param)
                    ), samples, header=[param]
                )

    @staticmethod
    def save_to_json(data, meta_file):
        """Save the metafile as a json file
        """
        with open(meta_file, "w") as f:
            json.dump(
                data, f, indent=4, sort_keys=True,
                cls=PESummaryJsonEncoder
            )

    @staticmethod
    def _seperate_dictionary_for_external_links(
            data, labels, sub_file_name="_{label}.h5"
    ):
        """
        """
        _data = copy.deepcopy(data)
        sub_file_data = {
            label: {
                label: _data[label], "version": _data["version"],
                "history": _data["history"]
            } for label in labels
        }
        meta_file_data = {
            key: item for key, item in _data.items() if key not in labels
        }
        for label in labels:
            meta_file_data[label] = "external:{}|{}".format(
                sub_file_name.format(label=label), label
            )
        return meta_file_data, sub_file_data

    @staticmethod
    def convert_posterior_samples_to_numpy(labels, samples, mcmc_samples=False):
        """Convert a dictionary of multiple posterior samples from a
        column-major dictionary to a row-major numpy array

        Parameters
        ----------
        labels: list
            list of unique labels for each analysis
        samples: MultiAnalysisSamplesDict
            dictionary of multiple posterior samples to convert to a numpy
            array.
        mcmc_samples: Bool, optional
            if True, the dictionary contains seperate mcmc chains

        Examples
        --------
        >>> dictionary = MultiAnalysisSamplesDict(
        ...     {"label": {"mass_1": [1,2,3], "mass_2": [1,2,3]}}
        ... )
        >>> dictionary = _Metafile.convert_posterior_samples_to_numpy(
        ...     dictionary.keys(), dictionary
        ... )
        >>> print(dictionary)
        ... {"label": rec.array([(1., 1.), (2., 2.), (3., 3.)],
        ...           dtype=[('mass_1', '<f4'), ('mass_2', '<f4')])}
        """
        converted_samples = {
            label: _MetaFile._convert_posterior_samples_to_numpy(
                samples[label], mcmc_samples=mcmc_samples
            ) for label in labels
        }
        return converted_samples

    @staticmethod
    def save_to_hdf5(
        data, labels, samples, meta_file, no_convert=False,
        extra_keys=DEFAULT_HDF5_KEYS, mcmc_samples=False,
        external_hdf5_links=False, compression=None, _class=None
    ):
        """Save the metafile as a hdf5 file
        """
        import h5py

        if _class is None:
            _class = _MetaFile
        if mcmc_samples:
            key = "mcmc_chains"
        else:
            key = "posterior_samples"
        if not no_convert:
            _samples = _class.convert_posterior_samples_to_numpy(
                labels, samples, mcmc_samples=mcmc_samples
            )
            for label in labels:
                data[label][key] = _samples[label]
                if "injection_data" in data[label].keys():
                    data[label]["injection_data"] = \
                        _class._convert_posterior_samples_to_numpy(
                            SamplesDict({
                                param: samp for param, samp in zip(
                                    data[label]["injection_data"]["parameters"],
                                    data[label]["injection_data"]["samples"]
                                )
                            }), index=[0]
                    )
        if external_hdf5_links:
            from pathlib import Path

            _dir = Path(meta_file).parent
            name = "_{label}.h5"
            _subfile = os.path.join(_dir, name)
            meta_file_data, sub_file_data = (
                _MetaFile._seperate_dictionary_for_external_links(
                    data, labels, sub_file_name=name
                )
            )
            for label in labels:
                with h5py.File(_subfile.format(label=label), "w") as f:
                    recursively_save_dictionary_to_hdf5_file(
                        f, sub_file_data[label], extra_keys=extra_keys + [label],
                        compression=compression
                    )
            with h5py.File(meta_file, "w") as f:
                recursively_save_dictionary_to_hdf5_file(
                    f, meta_file_data, extra_keys=extra_keys,
                    compression=compression
                )
        else:
            with h5py.File(meta_file, "w") as f:
                recursively_save_dictionary_to_hdf5_file(
                    f, data, extra_keys=extra_keys + labels,
                    compression=compression
                )

    def save_to_dat(self):
        """Save the samples to a .dat file
        """
        def _save(parameters, samples, label):
            """Helper function to save the parameters and samples to file
            """
            self.write_to_dat(
                os.path.join(self.outdir, "{}_pesummary.dat".format(label)),
                samples.T, header=parameters
            )

        if self.mcmc_samples:
            for label in self.labels:
                parameters = list(self.samples[label].keys())
                for chain in self.samples[label][parameters[0]].keys():
                    samples = np.array(
                        [self.samples[label][i][chain] for i in parameters]
                    )
                    _save(parameters, samples, chain)
                    return
        for label in self.labels:
            parameters = self.samples[label].keys()
            samples = np.array([self.samples[label][i] for i in parameters])
            _save(parameters, samples, label)

    def add_existing_data(self):
        """
        """
        from pesummary.utils.utils import _add_existing_data

        self = _add_existing_data(self)


class MetaFile(PostProcessing):
    """This class handles the creation of a metafile storing all information
    from the analysis
    """
    def __init__(self, inputs, history=None):
        from pesummary.utils.utils import logger

        super(MetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        meta_file = _MetaFile(
            self.samples, self.labels, self.config,
            self.injection_data, self.file_version, self.file_kwargs,
            hdf5=self.hdf5, webdir=self.webdir, result_files=self.result_files,
            existing_version=self.existing_file_version, existing_label=self.existing_labels,
            priors=self.priors, existing_samples=self.existing_samples,
            existing_injection=self.existing_injection_data,
            existing_metadata=self.existing_file_kwargs,
            existing_config=self.existing_config, existing=self.existing,
            existing_priors=self.existing_priors,
            existing_metafile=self.existing_metafile,
            package_information=self.package_information,
            mcmc_samples=self.mcmc_samples, filename=self.filename,
            external_hdf5_links=self.external_hdf5_links,
            hdf5_compression=self.hdf5_compression, history=history,
            descriptions=self.descriptions
        )
        meta_file.make_dictionary()
        if not self.hdf5:
            meta_file.save_to_json(meta_file.data, meta_file.meta_file)
        else:
            meta_file.save_to_hdf5(
                meta_file.data, meta_file.labels, meta_file.samples,
                meta_file.meta_file, mcmc_samples=meta_file.mcmc_samples,
                external_hdf5_links=meta_file.external_hdf5_links,
                compression=meta_file.hdf5_compression
            )
        meta_file.save_to_dat()
        meta_file.write_marginalized_posterior_to_dat()
        logger.info(
            "Finishing generating the meta file. The meta file can be viewed "
            "here: {}".format(meta_file.meta_file)
        )
