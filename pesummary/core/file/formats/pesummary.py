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

from glob import glob
import os

import h5py
import json
import numpy as np
import configparser


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
    def _grab_data_from_hdf5_file(path, **kwargs):
        """
        """
        function = kwargs.get(
            "grab_data_from_dictionary", PESummary._grab_data_from_dictionary)
        f = h5py.File(path)
        existing_data = function(f)
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

        parameter_list, sample_list, inj_list = [], [], []
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
        setattr(PESummary, "labels", labels)
        setattr(PESummary, "config", config)
        return parameter_list, sample_list, inj_list

    @property
    def samples_dict(self):
        outdict = {label: {par: [i[num] for i in self.samples[idx]] for num, par
                   in enumerate(self.parameters[idx])} for idx, label in
                   enumerate(self.labels)}
        return outdict

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
        from pandas import DataFrame

        objects = {}
        for num, i in enumerate(self.labels):
            posterior_data_frame = DataFrame(
                self.samples[num], columns=self.parameters[num])
            bilby_object = Result(
                search_parameter_keys=self.parameters[num],
                posterior=posterior_data_frame, label="pesummary_%s" % (i),
                samples=self.samples[num])
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
