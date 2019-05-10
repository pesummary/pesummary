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

import configparser
import shutil

import json
import h5py
import deepdish

import numpy as np

from pesummary.core.command_line import command_line
from pesummary.utils.utils import logger


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
                for z in paths_to_key(key, v, path):
                    yield z


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
    if isinstance(key, str):
        key = [key]
    if key[-1] in dictionary.keys():
        yield dictionary[key[-1]]
    else:
        old, new = key[0], key[1:]
        for z in load_recusively(new, dictionary[old]):
            yield z


class OneFormat(object):
    """Class to convert a given results file into a standard format with all
    derived posterior distributions included

    Parameters
    ----------
    fil: str
       path to the results file
    inj: str, optional
       path to the file containing injection information
    config: str, optional
       path to the configuration file

    Attributes
    ----------
    extension: str
        the extension of the input file
    lalinference_hdf5_format: Bool
        Boolean determining if the hdf5 file is of LALInference format
    bilby_hdf5_format: Bool
        Boolean determining if the hdf5 file is of Bilby format
    data: list
        list containing the extracted data from the input file
    parameters: list
        list of parameters stored in the input file
    samples: list
        list of samples stored in the input file
    approximant: str
        the approximant stored in the input file
    """
    def __init__(self, fil, inj=None, config=None):
        logger.info("Extracting the information from %s" % (fil))
        self.fil = fil
        self.inj = inj
        self.config = config
        self.data = None
        self.injection_data = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = None
        self.fixed_data = None
        self.marg_par = None
        if config:
            data = self._extract_data_from_config_file(config)
            self.fixed_data = data["fixed_data"]
            self.marg_par = data["marginalized_parameters"]
            self._config = config

    @property
    def extension(self):
        return self.fil.split(".")[-1]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        func_map = {"json": self._grab_data_from_json_file,
                    "hdf5": self._grab_data_from_hdf5_file,
                    "h5": self._grab_data_from_hdf5_file,
                    "dat": self._grab_data_from_dat_file,
                    "txt": self._grab_data_from_dat_file}
        data = func_map[self.extension]()
        parameters = data[0]
        samples = data[1]

        if self.fixed_data:

            for i in self.fixed_data.keys():
                fixed_parameter = i
                fixed_value = self.fixed_data[i]
                parameters.append(fixed_parameter)
                for num in range(len(samples)):
                    samples[num].append(float(fixed_value))

        if len(data) > 2:
            self._data = [parameters, samples, data[2]]
        else:
            self._data = [parameters, samples]

    @property
    def parameters(self):
        return self.data[0]

    @property
    def samples(self):
        return self.data[1]

    @property
    def approximant(self):
        if len(self.data) > 2:
            return self.data[2]
        return "none"

    @property
    def injection_data(self):
        return self._injection_data

    @injection_data.setter
    def injection_data(self, injection_data):
        if self.inj:
            extension = self.inj.split(".")[-1]
            func_map = {"xml": self._grab_injection_data_from_xml_file,
                        "hdf5": self._grab_injection_data_from_hdf5_file,
                        "h5": self._grab_injection_data_from_hdf5_file}
            self._injection_data = func_map[extension]()
        else:
            self._injection_data = [
                self.parameters, [float("nan")] * len(self.parameters)]

    @property
    def injection_parameters(self):
        return self.injection_data[0]

    @property
    def injection_values(self):
        return self.injection_data[1]

    def _grab_data_from_hdf5_file(self):
        """Grab the data stored in an hdf5 file
        """
        self._data_structure = []
        try:
            return self._grab_data_with_deepdish()
        except Exception as e:
            logger.warning("Failed to open %s with deepdish because %s. "
                           "Trying to grab the data with 'h5py'." % (
                               self.fil, e))
            try:
                return self._grab_data_with_h5py()
            except Exception:
                raise Exception("Failed to extract the data from the results "
                                "file. Please reformat the results file")

    def _add_to_list(self, item):
        self._data_structure.append(item)

    def _grab_data_with_h5py(self):
        """Grab the data stored in an hdf5 file using h5py
        """
        samples = []
        f = h5py.File(self.fil)
        f.visit(self._add_to_list)
        for i in self._data_structure:
            condition1 = "posterior_samples" in i or "posterior" in i or \
                         "samples" in i
            condition2 = "posterior_samples/" not in i and "posterior/" not in \
                         i and "samples/" not in i
            if condition1 and condition2:
                path = i

        if isinstance(f[path], h5py._hl.group.Group):
            parameters = [i for i in f[path].keys()]
            n_samples = len(f[path][parameters[0]])
            samples = [[f[path][i][num] for i in parameters] for num in range(n_samples)]
            cond1 = "loglr" not in parameters or "log_likelihood" not in \
                parameters
            cond2 = "likelihood_stats" in f.keys() and "loglr" in \
                f["likelihood_stats"]
            if cond1 and cond2:
                parameters.append("log_likelihood")
                for num, i in enumerate(samples):
                    samples[num].append(f["likelihood_stats/loglr"][num])
        elif isinstance(f[path], np.ndarray):
            parameters = f[path].dtype.names
            samples = [[i[parameters.index(j)] for j in parameters] for i in f[path]]
        f.close()
        return parameters, samples

    def _grab_data_with_deepdish(self):
        """Grab the data stored in an hdf5 file using deepdish
        """
        approx = "none"
        f = deepdish.io.load(self.fil)
        path, = paths_to_key("posterior", f)
        path = path[0]
        reduced_f, = load_recusively(path, f)
        parameters = [i for i in reduced_f.keys()]
        data = np.zeros([len(reduced_f[parameters[0]]), len(parameters)])
        for num, par in enumerate(parameters):
            for key, i in enumerate(reduced_f[par]):
                data[key][num] = float(np.real(i))
        data = data.tolist()
        for num, par in enumerate(parameters):
            if par == "logL":
                parameters[num] = "log_likelihood"
        return parameters, data, approx

    def _grab_data_from_json_file(self):
        """Grab the data stored in a .json file
        """
        with open(self.fil) as f:
            data = json.load(f)
        path, = paths_to_key("posterior", data)
        path = path[0]
        if "content" in data[path].keys():
            path += "/content"
        reduced_data, = load_recusively(path, data)
        parameters = list(reduced_data.keys())

        samples = [[
            reduced_data[j][i] if not isinstance(reduced_data[j][i], dict)
            else reduced_data[j][i]["real"] for j in parameters] for i in
            range(len(reduced_data[parameters[0]]))]
        return parameters, samples

    def _grab_data_from_dat_file(self):
        """Grab the data stored in a .dat file
        """
        dat_file = np.genfromtxt(self.fil, names=True)
        parameters = [i for i in dat_file.dtype.names]
        indices = [parameters.index(i) for i in parameters]
        samples = [[x[i] for i in indices] for x in dat_file]
        return parameters, samples

    def _extract_data_from_config_file(self, cp):
        """Grab the data from a config file
        """
        config = configparser.ConfigParser()
        try:
            config.read(cp)
            fixed_data = None
            marg_par = None
            if "engine" in config.sections():
                fixed_data = {
                    key.split("fix-")[1]: item for key, item in
                    config.items("engine") if "fix" in key}
                marg_par = {
                    key.split("marg")[1]: item for key, item in
                    config.items("engine") if "marg" in key}
            return {"fixed_data": fixed_data,
                    "marginalized_parameters": marg_par}
        except Exception:
            return {"fixed_data": None,
                    "marginalized_parameters": None}

    def generate_all_posterior_samples(self):
        self._update_injection_data()

    def _update_injection_data(self):
        if self.inj:
            extension = self.inj.split(".")[-1]
            func_map = {"xml": self._grab_injection_data_from_xml_file,
                        "hdf5": self._grab_injection_data_from_hdf5_file,
                        "h5": self._grab_injection_data_from_hdf5_file}
            self._injection_data = func_map[extension]()
        else:
            self._injection_data = [
                self.parameters, [float("nan")] * len(self.parameters)]

    def save(self):
        parameters = np.array(self.parameters, dtype="S")
        injection_parameters = np.array(self.injection_parameters, dtype="S")
        injection_data = np.array(self.injection_values)
        f = h5py.File("%s_temp" % (self.fil), "w")
        posterior_samples_group = f.create_group("posterior_samples")
        group = posterior_samples_group.create_group("label")
        group.create_dataset("parameter_names", data=parameters)
        group.create_dataset("samples", data=self.samples)
        group.create_dataset("injection_parameters", data=injection_parameters)
        group.create_dataset("injection_data", data=injection_data)
        f.close()
        return "%s_temp" % (self.fil)


def add_specific_arguments(parser):
    """Add command line arguments that are specific to pesummary_convert

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments
    """
    parser.add_argument("-o", "--outpath", dest="out",
                        help="location of output file", default=None)
    return parser


def main():
    """Top-level interface for pesummary_convert.py
    """
    parser = command_line()
    parser = add_specific_arguments(parser)
    opts = parser.parse_args()
    if opts.inj_file and len(opts.samples) != len(opts.inj_file):
        raise Exception("Please ensure that the number of results files "
                        "matches the number of injection files")
    if opts.config and len(opts.samples) != len(opts.config):
        raise Exception("Please ensure that the number of results files "
                        "matches the number of configuration files")
    if not opts.inj_file:
        opts.inj_file = []
        for i in range(len(opts.samples)):
            opts.inj_file.append(None)
    if not opts.config:
        opts.config = []
        for i in range(len(opts.samples)):
            opts.config.append(None)
    for num, i in enumerate(opts.samples):
        f = OneFormat(i, opts.inj_file[num], config=opts.config[num])
        f.generate_all_posterior_samples()
        g = f.save()
        if opts.out:
            shutil.move(g, opts.out)
