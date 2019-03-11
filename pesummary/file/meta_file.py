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

import shutil
import os

import numpy as np
import h5py

from pesummary.utils.utils import logger
from pesummary.inputs import PostProcessing
from pesummary.utils.utils import check_condition


def make_group_in_hf5_file(base_file, group_path):
    """Make a group in an hdf5 file

    Parameters
    ----------
    base_file: str
        path to the file that you want to add content to
    group_path: str
        the group path that you would like to create
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The file %s does not exist" % (base_file))
    f = h5py.File(base_file, "a")
    f.create_group(group_path)
    f.close()


def add_content_to_hdf_file(base_file, dataset_name, content, group=None):
    """Add new content to an hdf5 file

    Parameters
    ----------
    base_file: str
        path to the file that you want to add content to
    dataset_name: str
        name of the dataset
    content: array
        array of content that you want to add to your hdf5 file
    group: str, optional
        group that you want to add content to. Default if the base of the file
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The file %s does not exist" % (base_file))
    f = h5py.File(base_file, "a")
    if group:
        group = f[group]
        if dataset_name in list(group.keys()):
            del group[dataset_name]
        group.create_dataset(dataset_name, data=content)
    else:
        if dataset_name in list(f.keys()):
            del f[dataset_name]
        f.create_dataset(dataset_name, data=content)
    f.close()


def combine_hdf_files(base_file, new_file):
    """Combine two hdf5 files

    Parameters
    ----------
    base_file: str
        path to the file that you want to add content to
    new_file: str
        path to the file that you want to combine with the base file
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The base file %s does not exist" % (base_file))
    condition = not os.path.isfile(new_file)
    check_condition(condition, "The new file %s does not exist" % (new_file))
    g = h5py.File(new_file)
    label = list(g["posterior_samples"].keys())[0]
    approximant = list(g["posterior_samples/%s" % (label)].keys())[0]
    path = "posterior_samples/%s/%s" % (label, approximant)
    parameters = np.array([i for i in g["%s/parameter_names" % (path)]])
    samples = np.array([i for i in g["%s/samples" % (path)]])
    injection_parameters = np.array(
        [i for i in g["%s/injection_parameters" % (path)]])
    injection_data = np.array([i for i in g["%s/injection_data" % (path)]])
    g.close()

    f = h5py.File(base_file, "a")
    current_labels = list(f["posterior_samples"].keys())
    if label not in current_labels:
        label_group = f["posterior_samples"].create_group(label)
        approx_group = label_group.create_group(approximant)
    else:
        approx_group = f["posterior_samples/%s" % (label)].create_group(approximant)
    approx_group.create_dataset("parameter_names", data=parameters)
    approx_group.create_dataset("samples", data=samples)
    approx_group.create_dataset("injection_parameters", data=injection_parameters)
    approx_group.create_dataset("injection_data", data=injection_data)
    f.close()


class MetaFile(PostProcessing):
    """This class handles the creation of a meta file storing all information
    from the analysis

    Attributes
    ----------
    meta_file: str
        name of the meta file storing all information
    """
    def __init__(self, inputs):
        super(MetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        self.generate_meta_file()
        logger.info("Finished generating the meta file. The meta file can be "
                    "viewed here: %s" % (self.meta_file))

    @property
    def meta_file(self):
        return self.webdir + "/samples/posterior_samples.h5"

    @staticmethod
    def get_keys_from_hdf5_file(f, level=None):
        """
        """
        g = h5py.File(f)
        try:
            if level:
                return list(g[level].keys())
            else:
                return list(g.keys())
        except Exception as e:
            raise Exception("Failed to return the keys in the hdf5 file "
                            "because of %s" % (e))

    def labels_and_approximants_to_include(self):
        """Return the labels and the approximants that are unique and not
        already in the existing file.
        """
        labels = self.labels
        approximants = self.approximant
        names = ["%s_%s" % (i, j) for i, j in zip(
            self.labels, self.approximant)]
        if self.existing_names:
            labels_to_include, approximants_to_include = [], []
            for num, i in enumerate(names):
                if i not in self.existing_names:
                    labels_to_include.append(labels[num])
                    approximants_to_include.append(approximants[num])
            labels = labels_to_include
            approximants = approximants_to_include
        return labels, approximants

    def generate_meta_file(self):
        """Combine data into a single meta file.
        """
        if self.meta_file not in self.result_files:
            self._generate_meta_file_from_scratch()
        else:
            self._add_content_to_meta_file("existing")
        if self.psds:
            self._add_content_to_meta_file("psd")
        """
        if self.calibration:
            self._add_content_to_meta_file("calibration")
        """

    def _add_content_to_meta_file(self, key):
        """Add content to the meta file

        Parameters
        ----------
        key: str
            name of the data you want to append
        """
        content_map = {"existing": self._add_to_existing_meta_file,
                       "psd": self._add_psds_to_meta_file,
                       "calibration": self._add_calibration_to_meta_file}
        content_map[key]()

    def _generate_meta_file_from_scratch(self):
        for num, i in enumerate(self.result_files):
            if num == 0:
                shutil.copyfile(i, self.meta_file)
            else:
                combine_hdf_files(self.meta_file, i)

    def _add_to_existing_meta_file(self):
        for num, i in enumerate(self.result_files):
            if "posterior_samples.h5" not in i:
                combine_hdf_files(self.meta_file, i)

    def _add_psds_to_meta_file(self):
        keys = self.get_keys_from_hdf5_file(self.meta_file)
        labels, approximants = self.labels_and_approximants_to_include()
        if "psds" not in keys:
            make_group_in_hf5_file(self.meta_file, "psds")
        for i in labels:
            if self.existing_labels and i not in self.existing_labels:
                make_group_in_hf5_file(self.meta_file, "psds/%s" % (i))
        for num, i in enumerate(approximants):
            make_group_in_hf5_file(self.meta_file, "psds/%s/%s" % (labels[num], i))
            frequencies = [self._grab_frequencies_from_psd_data_file(j) for j
                           in self.psds]
            strains = [self._grab_strains_from_psd_data_file(j) for j in
                       self.psds]
            for idx, j in enumerate(self.psds):
                content = np.array([
                    (k, l) for k, l in zip(frequencies[idx], strains[idx])],
                    dtype=[("Frequencies", "f"), ("Strain", "f")])
                add_content_to_hdf_file(
                    self.meta_file, self.psd_labels[idx], content,
                    group="psds/%s/%s" % (labels[num], i))

    def _add_calibration_to_meta_file(self):
        """
        """
