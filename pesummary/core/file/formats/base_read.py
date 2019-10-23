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
from pesummary.utils.utils import SamplesDict, Array


class Read():
    """Base class to read in a results file
    """
    def __init__(self, path_to_results_file):
        self.path_to_results_file = path_to_results_file
        self.extension = self.extension_from_path(self.path_to_results_file)

    @staticmethod
    def load_from_function(function, path_to_file, **kwargs):
        """Load a file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your file
        path_to_file: str
            path to the file that you wish to load
        kwargs: dict
            all kwargs are passed to the function
        """
        return function(path_to_file, **kwargs)

    @staticmethod
    def check_for_weights(parameters, samples):
        """Check to see if the samples are weighted

        Parameters
        ----------
        parameters: list
            list of parameters stored in the result file
        samples: np.ndarray
            array of samples for each parameter
        """
        likely_names = ["weights", "weight"]
        if any(i in parameters for i in likely_names):
            ind = (
                parameters.index("weights") if "weights" in parameters else
                parameters.index("weight")
            )
            return Array(np.array(samples).T[ind])
        return None

    def load(self, function, **kwargs):
        """Load a results file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your results file
        """
        self.data = self.load_from_function(
            function, self.path_to_results_file, **kwargs)
        if "injection" in self.data.keys():
            self.injection_parameters = self.data["injection"]
        if "version" in self.data.keys():
            self.input_version = self.data["version"]
        else:
            self.input_version = "No version information found"
        if "kwargs" in self.data.keys():
            self.extra_kwargs = self.data["kwargs"]
        else:
            self.extra_kwargs = {"sampler": {}, "meta_data": {}}
            self.extra_kwargs["sampler"]["nsamples"] = len(self.data["samples"])
        if "prior" in self.data.keys():
            self.priors = self.data["prior"]
        if "weights" in self.data.keys():
            self.weights = self.data["weights"]
        else:
            self.weights = self.check_for_weights(
                self.data["parameters"], self.data["samples"]
            )

    @property
    def parameters(self):
        return self.data["parameters"]

    @property
    def samples(self):
        return self.data["samples"]

    @property
    def samples_dict(self):
        return SamplesDict(self.parameters, np.array(self.samples).T)

    @staticmethod
    def paths_to_key(key, dictionary, current_path=None):
        """Return the path to a key stored in a nested dictionary

        Parameters
        ----------
        key: str
            the key that you would like to find
        dictionary: dict
            the nested dictionary that has the key stored somewhere within it
        current_path: str, optional
            the current level in the dictionary
        """
        if current_path is None:
            current_path = []

        for k, v in dictionary.items():
            if k == key:
                yield current_path + [key]
            else:
                if isinstance(v, dict):
                    path = current_path + [k]
                    for z in Read.paths_to_key(key, v, path):
                        yield z

    @staticmethod
    def load_recusively(key, dictionary):
        """Return the entry for a key of format 'a/b/c/d'

        Parameters
        ----------
        key: str
            key of format 'a/b/c/d'
        dictionary: dict
            the dictionary that has the key stored
        """
        if "/" in key:
            key = key.split("/")
        if isinstance(key, (str, float)):
            key = [key]
        if key[-1] in dictionary.keys():
            yield dictionary[key[-1]]
        else:
            old, new = key[0], key[1:]
            for z in Read.load_recusively(new, dictionary[old]):
                yield z

    @staticmethod
    def extension_from_path(path):
        """Return the extension of the file from the file path

        Parameters
        ----------
        path: str
            path to the results file
        """
        extension = path.split(".")[-1]
        return extension

    @staticmethod
    def guess_path_to_samples(path):
        """Guess the path to the posterior samples stored in an hdf5 object

        Parameters
        ----------
        path: str
            path to the results file
        """
        def _find_name(name):
            c1 = "posterior_samples" in name or "posterior" in name
            c2 = "posterior_samples/" not in name and "posterior/" not in name
            if c1 and c2:
                return name

        f = h5py.File(path, 'r')
        _path = f.visit(_find_name)
        f.close()
        return _path

    @staticmethod
    def _grab_params_and_samples_from_json_file(path):
        """Grab the parameters and samples stored in a .json file
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)
        try:
            path, = Read.paths_to_key("posterior", data)
            path = path[0]
            path += "/posterior"
        except Exception:
            path, = Read.paths_to_key("posterior_samples", data)
            path = path[0]
            path += "/posterior_samples"
        reduced_data, = Read.load_recusively(path, data)
        if "content" in list(reduced_data.keys()):
            reduced_data = reduced_data["content"]
        parameters = list(reduced_data.keys())

        samples = [[
            reduced_data[j][i] if not isinstance(reduced_data[j][i], dict)
            else reduced_data[j][i]["real"] for j in parameters] for i in
            range(len(reduced_data[parameters[0]]))]
        return parameters, samples

    @staticmethod
    def _grab_params_and_samples_from_dat_file(path):
        """Grab the parameters and samples in a .dat file
        """
        dat_file = np.genfromtxt(path, names=True)
        parameters = [i for i in dat_file.dtype.names]
        samples = [list(x) for x in dat_file]
        return parameters, samples

    def generate_all_posterior_samples(self):
        """Empty function
        """
        pass

    def add_fixed_parameters_from_config_file(self, config_file):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass

    def _add_fixed_parameters_from_config_file(self, config_file, function):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        function: func
            function you wish to use to extract the information from the
            configuration file
        """
        self.data[0], self.data[1] = function(self.parameters, self.samples, config_file)

    def _add_marginalized_parameters_from_config_file(self, config_file, function):
        """Search the configuration file and add marginalized parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        function: func
            function you wish to use to extract the information from the
            configuration file
        """
        self.data[0], self.data[1] = function(self.parameters, self.samples, config_file)

    def _add_injection_parameters_from_file(self, injection_file, function):
        """Add the injection parameters from file

        Parameters
        ----------
        injection_file: str
            path to injection file
        function: func
            funcion you wish to use to extract the information from the
            injection file
        """
        return function(injection_file)

    def to_dat(self, outdir="./", label=None):
        """Save the PESummary results file object to a dat file

        Parameters
        ----------
        outdir: str
            path to the directory where you would like to save the results file
        label: str
            the label of the result file
        """
        import os

        if not label:
            from time import time

            label = round(time())

        if os.path.isfile("%s/pesummary_%s.dat" % (outdir, label)):
            raise Exception("The file '%s/pesummary_%s.dat' already exists." % (
                outdir, label))

        try:
            np.savetxt(
                "%s/pesummary_%s.dat" % (outdir, label), self.samples,
                delimiter="\t", header="\t".join(self.parameters),
                comments='')
        except Exception:
            raise Exception("Please make sure you have write permission in "
                            "%s" % (outdir))
