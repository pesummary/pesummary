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
from pesummary.utils.samples_dict import SamplesDict, MCMCSamplesDict, Array
from pesummary.utils.utils import logger


class Read(object):
    """Base class to read in a results file

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file

    Methods
    -------
    downsample:
        downsample the posterior samples stored in the result file
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, path_to_results_file):
        self.path_to_results_file = path_to_results_file
        self.mcmc_samples = False
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
        self.parameters = self.data["parameters"]
        self.samples = self.data["samples"]
        if "mcmc_samples" in self.data.keys():
            self.mcmc_samples = self.data["mcmc_samples"]
        if "injection" in self.data.keys():
            if isinstance(self.data["injection"], dict):
                self.injection_parameters = {
                    key.decode("utf-8") if isinstance(key, bytes) else key: val
                    for key, val in self.data["injection"].items()
                }
            elif isinstance(self.data["injection"], list):
                self.injection_parameters = [
                    {
                        key.decode("utf-8") if isinstance(key, bytes) else
                        key: val for key, val in i.items()
                    } for i in self.data["injection"]
                ]
            else:
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
        if "labels" in self.data.keys():
            self.labels = self.data["labels"]
        if "config" in self.data.keys():
            self.config = self.data["config"]
        if "weights" in self.data.keys():
            self.weights = self.data["weights"]
        else:
            self.weights = self.check_for_weights(
                self.data["parameters"], self.data["samples"]
            )

    @property
    def samples_dict(self):
        if self.mcmc_samples:
            return MCMCSamplesDict(
                self.parameters, [np.array(i).T for i in self.samples]
            )
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
    def convert_list_to_item(dictionary):
        """
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                Read.convert_list_to_item(value)
            else:
                if isinstance(value, (list, np.ndarray, Array)):
                    if len(value) == 1 and isinstance(value[0], bytes):
                        dictionary.update({key: value[0].decode("utf-8")})
                    elif len(value) == 1:
                        dictionary.update({key: value[0]})
        return dictionary

    @staticmethod
    def load_recursively(key, dictionary):
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
            try:
                converted_dictionary = Read.convert_list_to_item(
                    dictionary[key[-1]]
                )
                yield converted_dictionary
            except AttributeError:
                yield dictionary[key[-1]]
        else:
            old, new = key[0], key[1:]
            for z in Read.load_recursively(new, dictionary[old]):
                yield z

    @staticmethod
    def edit_dictionary(dictionary, path, value):
        """Replace an entry in a nested dictionary

        Parameters
        ----------
        dictionary: dict
            the nested dictionary that you would like to edit
        path: list
            the path to the key that you would like to edit
        value:
            the replacement
        """
        from functools import reduce
        from operator import getitem

        edit = dictionary.copy()
        reduce(getitem, path[:-1], edit)[path[-1]] = value
        return edit

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
        def _find_name(name, item):
            c1 = "posterior_samples" in name or "posterior" in name
            c2 = isinstance(item, (h5py._hl.dataset.Dataset, np.ndarray))
            try:
                c3 = isinstance(item, h5py._hl.group.Group) and isinstance(
                    item[0], (float, int, np.number)
                )
            except AttributeError:
                c3 = False
            c4 = (
                isinstance(item, h5py._hl.group.Group) and "parameter_names" in
                item.keys() and "samples" in item.keys()
            )
            if c1 and c3:
                paths.append(name)
            elif c1 and c4:
                return paths.append(name)
            elif c1 and c2:
                return paths.append(name)

        f = h5py.File(path, 'r')
        paths = []
        f.visititems(_find_name)
        f.close()
        if len(paths) == 1:
            return paths[0]
        else:
            raise ValueError(
                "Found multiple posterior sample tables in '{}': {}. Not sure "
                "which to load.".format(
                    path, ", ".join(paths)
                )
            )

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
        reduced_data, = Read.load_recursively(path, data)
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

    def generate_all_posterior_samples(self, **kwargs):
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

    def write(self, package="core", **kwargs):
        """Save the data to file

        Parameters
        ----------
        package: str, optional
            package you wish to use when writing the data
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        from pesummary.io import write

        return write(
            self.parameters, self.samples, package=package,
            file_versions=self.input_version, file_kwargs=self.extra_kwargs,
            **kwargs
        )

    def downsample(self, number):
        """Downsample the posterior samples stored in the result file
        """
        from pesummary.utils.utils import resample_posterior_distribution

        if number > self.samples_dict.number_of_samples:
            raise ValueError(
                "Failed to downsample the posterior samples to {} because "
                "there are only {} samples stored in the file.".format(
                    number, self.samples_dict.number_of_samples
                )
            )
        _samples = np.array(resample_posterior_distribution(
            np.array(self.samples).T, number
        ))
        self.samples = _samples.T.tolist()
        self.extra_kwargs["sampler"]["nsamples"] = number

    def to_dat(self, **kwargs):
        """Save the PESummary results file object to a dat file

        Parameters
        ----------
        kwargs: dict
            all kwargs passed to the pesummary.core.file.formats.dat.write_dat
            function
        """
        return self.write(file_format="dat", **kwargs)

    @staticmethod
    def latex_table(samples, parameter_dict=None, labels=None):
        """Return a latex table displaying the passed data.

        Parameters
        ----------
        samples_dict: list
            list of pesummary.utils.utils.SamplesDict objects
        parameter_dict: dict, optional
            dictionary of parameters that you wish to include in the latex
            table. The keys are the name of the parameters and the items are
            the descriptive text. If None, all parameters are included
        """
        table = (
            "\\begin{table}[hptb]\n\\begin{ruledtabular}\n\\begin{tabular}"
            "{l %s}\n" % ("c " * len(samples))
        )
        if labels:
            table += (
                " & " + " & ".join(labels)
            )
            table += "\\\ \n\\hline \\\ \n"
        data = {i: i for i in samples[0].keys()}
        if parameter_dict is not None:
            import copy

            data = copy.deepcopy(parameter_dict)
            for param in parameter_dict.keys():
                if not all(param in samples_dict.keys() for samples_dict in samples):
                    logger.warn(
                        "{} not in list of parameters. Not adding to "
                        "table".format(param)
                    )
                    data.pop(param)

        for param, desc in data.items():
            table += "{}".format(desc)
            for samples_dict in samples:
                median = samples_dict[param].average(type="median")
                confidence = samples_dict[param].confidence_interval()
                table += (
                    " & $%s^{+%s}_{-%s}$" % (
                        np.round(median, 2),
                        np.round(confidence[1] - median, 2),
                        np.round(median - confidence[0], 2)
                    )
                )
            table += "\\\ \n"
        table += (
            "\\end{tabular}\n\\end{ruledtabular}\n\\caption{}\n\\end{table}"
        )
        return table

    @staticmethod
    def latex_macros(
        samples, parameter_dict=None, labels=None, rounding="smart"
    ):
        """Return a latex table displaying the passed data.

        Parameters
        ----------
        samples_dict: list
            list of pesummary.utils.utils.SamplesDict objects
        parameter_dict: dict, optional
            dictionary of parameters that you wish to generate macros for. The
            keys are the name of the parameters and the items are the latex
            macros name you wish to use. If None, all parameters are included.
        rounding: int, optional
            decimal place for rounding. Default uses the
            `pesummary.utils.utils.smart_round` function to round according to
            the uncertainty
        """
        macros = ""
        data = {i: i for i in samples[0].keys()}
        if parameter_dict is not None:
            import copy

            data = copy.deepcopy(parameter_dict)
            for param in parameter_dict.keys():
                if not all(param in samples_dict.keys() for samples_dict in samples):
                    logger.warn(
                        "{} not in list of parameters. Not generating "
                        "macro".format(param)
                    )
                    data.pop(param)
        for param, desc in data.items():
            for num, samples_dict in enumerate(samples):
                if labels:
                    description = "{}{}".format(desc, labels[num])
                else:
                    description = desc

                median = samples_dict[param].average(type="median")
                confidence = samples_dict[param].confidence_interval()
                if rounding == "smart":
                    from pesummary.utils.utils import smart_round

                    median, upper, low = smart_round([
                        median, confidence[1] - median, median - confidence[0]
                    ])
                else:
                    median = np.round(median, rounding)
                    low = np.round(median - confidence[0], rounding)
                    upper = np.round(confidence[1] - median, rounding)
                macros += (
                    "\\def\\%s{$%s_{-%s}^{+%s}$}\n" % (
                        description, median, low, upper
                    )
                )
                macros += (
                    "\\def\\%smedian{$%s$}\n" % (description, median)
                )
                macros += (
                    "\\def\\%supper{$%s$}\n" % (
                        description, np.round(median + upper, 9)
                    )
                )
                macros += (
                    "\\def\\%slower{$%s$}\n" % (
                        description, np.round(median - low, 9)
                    )
                )
        return macros

    def to_latex_table(self, parameter_dict=None, save_to_file=None):
        """Make a latex table displaying the data in the result file.

        Parameters
        ----------
        parameter_dict: dict, optional
            dictionary of parameters that you wish to include in the latex
            table. The keys are the name of the parameters and the items are
            the descriptive text. If None, all parameters are included
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )

        table = self.latex_table([self.samples_dict], parameter_dict)
        if save_to_file is None:
            print(table)
        elif os.path.isfile("{}".format(save_to_file)):
            logger.warn(
                "File {} already exists. Printing to stdout".format(save_to_file)
            )
            print(table)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([table])

    def generate_latex_macros(
        self, parameter_dict=None, save_to_file=None, rounding="smart"
    ):
        """Generate a list of latex macros for each parameter in the result
        file

        Parameters
        ----------
        labels: list, optional
            list of labels that you want to include in the table
        parameter_dict: dict, optional
            dictionary of parameters that you wish to generate macros for. The
            keys are the name of the parameters and the items are the latex
            macros name you wish to use. If None, all parameters are included.
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        rounding: int, optional
            number of decimal points to round the latex macros
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )

        macros = self.latex_macros(
            [self.samples_dict], parameter_dict, rounding=rounding
        )
        if save_to_file is None:
            print(macros)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([macros])
