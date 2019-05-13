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

from pesummary.core.file.one_format import load_recusively

from glob import glob
import os

import h5py
import json


class ExistingFile(object):
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
    def __init__(self, existing_webdir):
        self.existing = existing_webdir
        self.existing_data = []

    @property
    def existing_file(self):
        if os.path.isfile(self.existing):
            return self.existing
        elif os.path.isdir(self.existing):
            meta_file = glob(self.existing + "/samples/posterior_samples*")
            return meta_file[0]

    @property
    def extension(self):
        return self.existing_file.split(".")[-1]

    @property
    def existing_data(self):
        return self._existing_data

    @existing_data.setter
    def existing_data(self, existing_data):
        func_map = {"h5": self._grab_data_from_hdf5_file,
                    "json": self._grab_data_from_json_file}
        self._existing_data = func_map[self.extension]()

    def _grab_data_from_hdf5_file(self):
        """
        """
        f = h5py.File(self.existing_file)
        existing_data = self._grab_data_from_dictionary(f)
        f.close()
        return existing_data

    def _grab_data_from_json_file(self):
        with open(self.existing_file) as f:
            data = json.load(f)
        return self._grab_data_from_dictionary(data)

    def _grab_data_from_dictionary(self, dictionary):
        """
        """
        labels = list(dictionary["posterior_samples"].keys())
        existing_structure = {
            i: j for i in labels for j in
            dictionary["posterior_samples"]["%s" % (i)].keys()}
        labels = list(existing_structure.keys())

        parameter_list, sample_list = [], []
        for num, i in enumerate(labels):
            p = [j for j in dictionary["posterior_samples"]["%s" % (i)]["parameter_names"]]
            s = [j for j in dictionary["posterior_samples"]["%s" % (i)]["samples"]]
            if isinstance(p[0], bytes):
                parameter_list.append([j.decode("utf-8") for j in p])
            else:
                parameter_list.append([j for j in p])
            sample_list.append(s)
            config = None
            if "config_file" in dictionary.keys():
                config, = load_recusively("config_file/%s" % (i), dictionary)
        return labels, parameter_list, sample_list, config

    @property
    def existing_labels(self):
        return self.existing_data[0]

    @property
    def existing_parameters(self):
        return self.existing_data[1]

    @property
    def existing_samples(self):
        return self.existing_data[2]

    @property
    def existing_config(self):
        return self.existing_config[3]
