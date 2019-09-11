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

from pesummary.core.file.formats.base_read import Read
from pesummary.utils.utils import logger

from glob import glob
import os

import h5py
import json
import numpy as np
import configparser


class Array(np.ndarray):
    """Class to add extra functions and methods to np.ndarray

    Parameters
    ----------
    input_aray: list/array
        input list/array

    Attributes
    ----------
    median: float
        median of the input array
    mean: float
        mean of the input array
    """
    def __new__(cls, input_array, likelihood=None):
        obj = np.asarray(input_array).view(cls)
        obj.median = cls.median(obj)
        obj.mean = cls.mean(obj)
        obj.maxL = cls.maxL(obj, likelihood)
        return obj

    @staticmethod
    def median(array):
        """Return the median of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        """
        return np.median(array)

    @staticmethod
    def mean(array):
        """Return the mean of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        """
        return np.mean(array)

    @staticmethod
    def maxL(array, likelihood=None):
        """Return the maximum likelihood value of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        likelihood: np.ndarray, optional
            likelihoods associated with each sample
        """
        if likelihood is not None:
            likelihood = list(likelihood)
            ind = likelihood.index(np.max(likelihood))
            return array[ind]
        return None

    def confidence_interval(self, percentile=None):
        """Return the confidence interval of the array

        Parameters
        ----------
        percentile: int/list, optional
            Percentile or sequence of percentiles to compute, which must be
            between 0 and 100 inclusive
        """
        if percentile is not None:
            if isinstance(percentile, int):
                return np.percentile(self, percentile)
            return np.array([np.percentile(self, i) for i in percentile])
        return np.array([np.percentile(self, i) for i in [10, 90]])

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.median = getattr(obj, 'median', None)
        self.mean = getattr(obj, 'mean', None)
        self.maxL = getattr(obj, 'maxL', None)


class PESummary(Read):
    """This class handles the existing posterior_samples.h5 file

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    existing_file: str
        the path to the existing posterior_samples.h5 file
    existing_approximants: list
        list of approximants that have been used in the previous analysis
    existing_labels: list
        list of labels that have been used in the previous analysis
    existing_samples: nd list
        nd list of samples stored for each approximant used in the previous
        analysis
    """
    def __init__(self, path_to_results_file):
        super(PESummary, self).__init__(path_to_results_file)
        self.load(self._grab_data_from_pesummary_file)
        self.samples_dict = None

    @classmethod
    def load_file(cls, path):
        if os.path.isdir(path):
            files = glob(path + "/*")
            if "home.html" in files:
                path = glob(path + "/samples/posterior_samples*")[0]
            else:
                raise Exception(
                    "Unable to find a file called 'posterior_samples' in "
                    "the directory %s" % (path + "/samples"))
        if not os.path.isfile(path):
            raise Exception("%s does not exist" % (path))
        return cls(path)

    @staticmethod
    def _grab_data_from_pesummary_file(path, **kwargs):
        """
        """
        func_map = {"h5": PESummary._grab_data_from_hdf5_file,
                    "hdf5": PESummary._grab_data_from_hdf5_file,
                    "json": PESummary._grab_data_from_json_file}
        return func_map[Read.extension_from_path(path)](path, **kwargs)

    @staticmethod
    def _convert_hdf5_to_dict(dictionary, path="/"):
        """
        """
        mydict = {}
        for key, item in dictionary[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                mydict[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                mydict[key] = PESummary._convert_hdf5_to_dict(
                    dictionary, path=path + key + "/")
        return mydict

    @staticmethod
    def _grab_data_from_hdf5_file(path, **kwargs):
        """
        """
        function = kwargs.get(
            "grab_data_from_dictionary", PESummary._grab_data_from_dictionary)
        f = h5py.File(path)
        data = PESummary._convert_hdf5_to_dict(f)
        existing_data = function(data)
        f.close()
        return existing_data

    @staticmethod
    def _grab_data_from_json_file(path, **kwargs):
        function = kwargs.get(
            "grab_data_from_dictionary", PESummary._grab_data_from_dictionary)
        with open(path) as f:
            data = json.load(f)
        return function(data)

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        labels = list(dictionary["posterior_samples"].keys())
        existing_structure = {
            i: j for i in labels for j in
            dictionary["posterior_samples"]["%s" % (i)].keys()}
        labels = list(existing_structure.keys())

        parameter_list, sample_list, inj_list, ver_list = [], [], [], []
        meta_data_list = []
        for num, i in enumerate(labels):
            p = [j for j in dictionary["posterior_samples"]["%s" % (i)]["parameter_names"]]
            s = [j for j in dictionary["posterior_samples"]["%s" % (i)]["samples"]]
            if "injection_data" in dictionary.keys():
                inj = [j for j in dictionary["injection_data"]["%s" % (i)]["injection_values"]]
                for num, j in enumerate(inj):
                    if isinstance(j, (str, bytes)):
                        if j.decode("utf-8") == "NaN":
                            inj[num] = float("nan")
                inj_list.append({i: j for i, j in zip(p, inj)})
            if isinstance(p[0], bytes):
                parameter_list.append([j.decode("utf-8") for j in p])
            else:
                parameter_list.append([j for j in p])
            sample_list.append(s)
            config = None
            if "config_file" in dictionary.keys():
                config, = Read.load_recusively("config_file", dictionary)
            if "meta_data" in dictionary.keys():
                data, = Read.load_recusively("meta_data", dictionary)
                meta_data_list.append(data["%s" % (i)])
            else:
                meta_data_list.append({"sampler": {}, "meta_data": {}})
        if "version" in dictionary.keys():
            version, = Read.load_recusively("version", dictionary)
        else:
            version = {i: "No version information found" for i in labels
                       + ["pesummary"]}
        for i in list(version.keys()):
            if i != "pesummary" and isinstance(version[i][0], bytes):
                ver_list.append(version[i][0].decode("utf-8"))
            elif i != "pesummary":
                ver_list.append(version[i][0])
            elif isinstance(version["pesummary"], bytes):
                version["pesummary"] = version["pesummary"].decode("utf-8")
        setattr(PESummary, "labels", labels)
        setattr(PESummary, "config", config)
        setattr(PESummary, "version", version["pesummary"])
        return parameter_list, sample_list, inj_list, ver_list, meta_data_list

    @property
    def samples_dict(self):
        return self._samples_dict

    @samples_dict.setter
    def samples_dict(self, samples_dict):
        if all("log_likelihood" in i for i in self.parameters):
            likelihood_inds = [self.parameters[idx].index("log_likelihood") for
                               idx in range(len(self.labels))]
            likelihoods = [[i[likelihood_inds[idx]] for i in self.samples[idx]]
                           for idx, label in enumerate(self.labels)]
        else:
            likelihoods = [None] * len(self.labels)
        outdict = {
            label: {
                par: Array(
                    [i[num] for i in self.samples[idx]],
                    likelihood=likelihoods[idx]
                )
                for num, par in enumerate(self.parameters[idx])
            }
            for idx, label in enumerate(self.labels)
        }
        self._samples_dict = outdict

    def write_config_to_file(self, label, outdir="./"):
        """Write the config file stored as a dictionary to file

        Parameters
        ----------
        label: str
            the label for the dictionary that you would like to write to file
        outdir: str, optional
            path indicating where you would like to configuration file to be
            saved. Default is current working directory
        """
        if label not in list(self.config.keys()):
            raise Exception("The label %s does not exist." % (label))
        config_dict = self.config[label]
        config = configparser.ConfigParser()
        for i in config_dict.keys():
            config[i] = config_dict[i]

        with open("%s/%s_config.ini" % (outdir, label), "w") as configfile:
            config.write(configfile)

    def to_bilby(self):
        """Convert a PESummary metafile to a bilby results object
        """
        from bilby.core.result import Result
        from bilby.core.prior import PriorDict
        from bilby.core.prior import Uniform
        from pandas import DataFrame

        objects = {}
        for num, i in enumerate(self.labels):
            posterior_data_frame = DataFrame(
                self.samples[num], columns=self.parameters[num])
            priors = PriorDict()
            logger.warn(
                "No prior information is known so setting it to a default")
            priors.update({i: Uniform(-10, 10, 0) for i in self.parameters[num]})
            bilby_object = Result(
                search_parameter_keys=self.parameters[num],
                posterior=posterior_data_frame, label="pesummary_%s" % (i),
                samples=self.samples[num], priors=priors)
            objects[i] = bilby_object
        return objects

    def to_dat(self, label="all", outdir="./"):
        """Convert the samples stored in a PESummary metafile to a .dat file

        Parameters
        ----------
        label: str, optional
            the label of the analysis that you wish to save. By default, all
            samples in the metafile will be saved to seperate files
        outdir: str, optional
            path indicating where you would like to configuration file to be
            saved. Default is current working directory
        """
        if label != "all" and label not in list(self.labels):
            raise Exception("The label %s does not exist." % (label))
        if label == "all":
            label = list(self.labels)
        else:
            label = [label]
        for num, i in enumerate(label):
            ind = self.labels.index(i)
            np.savetxt(
                "%s/pesummary_%s.dat" % (outdir, i), self.samples[ind],
                delimiter="\t", header="\t".join(self.parameters[ind]),
                comments='')
