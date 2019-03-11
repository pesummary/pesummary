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

import h5py


class ExistingFile():
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
        self.existing_parameters = []
        self.existing_samples = []

    @property
    def existing_file(self):
        return self.existing + "/samples/posterior_samples.h5"

    @property
    def existing_approximant(self):
        structure = self._data_structure_of_results_file()
        approximants = [structure[i] for i in structure.keys()]
        return approximants

    @property
    def existing_labels(self):
        return list(self._data_structure_of_results_file().keys())

    @property
    def existing_samples(self):
        return self._existing_samples

    @existing_samples.setter
    def existing_samples(self, existing_samples):
        sample_list = []
        f = h5py.File(self.existing_file)
        for num, i in enumerate(self.existing_labels):
            s = [j for j in f["posterior_samples/%s/%s/samples" % (
                 i, self.existing_approximant[num])]]
            sample_list.append(s)
        f.close()
        self._existing_samples = sample_list

    @property
    def existing_parameters(self):
        return self._existing_parameters

    @existing_parameters.setter
    def existing_parameters(self, existing_parameters):
        parameter_list = []
        f = h5py.File(self.existing_file)
        for num, i in enumerate(self.existing_labels):
            p = [j for j in f["posterior_samples/%s/%s/parameter_names" % (
                 i, self.existing_approximant[num])]]
            parameter_list.append([j.decode("utf-8") for j in p])
        f.close()
        self._existing_parameters = parameter_list

    def _data_structure_of_results_file(self):
        """Return the structure of the existing results file
        """
        f = h5py.File(self.existing_file)
        labels = list(f["posterior_samples"].keys())
        structure = {i: j for i in labels for j in list(
            f["posterior_samples/%s" % (i)].keys())}
        f.close()
        return structure
