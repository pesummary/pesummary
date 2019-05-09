#! /usr/bin/env python

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

import socket
import os
import shutil
from glob import glob

import h5py

from time import time

import pesummary
from pesummary.utils.utils import (guess_url, logger,
                                   rename_group_or_dataset_in_hf5_file)
from pesummary.utils import utils
from pesummary.core.file.one_format import OneFormat
from pesummary.core.file.existing import ExistingFile

__doc__ == "Classes to handle the command line inputs"


class Input(object):
    """Super class to handle command line arguments

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    user: str
        The user who submitted the job
    add_to_existing: Bool
        Boolean to determine if you wish to add to a existing webpage
    existing: str
        Existing web directory
    webdir: str
        Directory to output all information
    baseurl: str
        The url path for the corresponding web directory
    inj_file: List
        List containing paths to the injection file
    result_files: list
        List containing paths to the results files which are being analysed
    config: list
        List containing paths to the configuration files used to generate each
        results files
    approximant: list
        List of approximants used in the analysis to generate each results
        files
    email: str
        The email address to notify when the job has completed
    sensitivity: Bool
        Boolean to determine if you wish to plot the sky sensitivity for
        different detector networks
    gracedb: str
        The gracedb of the event that produced the results files
    dump: Bool
        Boolean to determine if you wish to produce a dumped html page layout
    detectors: list
        List containing the detectors used to generate each results file
    labels: str
        A label for this summary page
    psd: str
        List of the psds used in the analysis
    """
    def __init__(self, opts):
        logger.info("Command line arguments: %s" % (opts))
        self.user = opts.user
        self.existing = opts.existing
        self.webdir = opts.webdir
        self.baseurl = opts.baseurl
        self.inj_file = opts.inj_file
        self.config = opts.config
        self.result_files = opts.samples
        self.email = opts.email
        self.add_to_existing = opts.add_to_existing
        self.dump = opts.dump
        self.hdf5 = opts.save_to_hdf5
        self.existing_labels = []
        self.existing_parameters = []
        self.existing_samples = []
        self.make_directories()
        self.copy_files()
        self.labels = opts.labels

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user):
        try:
            self._user = os.environ["USER"]
            logger.debug("The following user submitted the job %s"
                         % (self._user))
        except Exception as e:
            logger.info("Failed to grab user information because %s. "
                        "Default will be used (%s)" % (e, user))
            self._user = user

    @property
    def host(self):
        return socket.getfqdn()

    @property
    def existing(self):
        return self._existing

    @existing.setter
    def existing(self, existing):
        if not existing:
            self._existing = None
        else:
            self._existing = existing

    @property
    def webdir(self):
        return self._webdir

    @webdir.setter
    def webdir(self, webdir):
        if not webdir and not self.existing:
            raise Exception("Please provide a web directory to store the "
                            "webpages. If this is an existing directory "
                            "pass this path under the --existing_webdir "
                            "argument. If this is a new directory then "
                            "pass this under the --webdir argument")
        elif not webdir and self.existing:
            if not os.path.isdir(self.existing):
                raise Exception("The directory %s does not "
                                "exist" % (self.existing))
            entries = glob(self.existing + "/*")
            if "%s/home.html" % (self.existing) not in entries:
                raise Exception("Please give the base directory of an "
                                "existing output")
            self._webdir = self.existing
        if webdir:
            if not os.path.isdir(webdir):
                logger.debug("Given web directory does not exist. "
                             "Creating it now")
                utils.make_dir(webdir)
            self._webdir = webdir

    @property
    def baseurl(self):
        return self._baseurl

    @baseurl.setter
    def baseurl(self, baseurl):
        self._baseurl = baseurl
        if not baseurl:
            self._baseurl = guess_url(self.webdir, socket.getfqdn(), self.user)
            logger.debug("No url is provided. The url %s will be "
                         "used" % (self._baseurl))

    @property
    def inj_file(self):
        return self._inj_file

    @inj_file.setter
    def inj_file(self, inj_file):
        if inj_file:
            self._inj_file = inj_file
        else:
            self._inj_file = None
        self._inj_file = inj_file

    @property
    def result_files(self):
        return self._result_files

    @property
    def parameters(self):
        return self._parameters

    @property
    def samples(self):
        return self._samples

    @property
    def injection_data(self):
        return self._injection_data

    @result_files.setter
    def result_files(self, samples):
        result_file_list = []
        parameters_list, samples_list, injection_list = [], [], []
        if not samples:
            raise Exception("Please provide a results file")
        if self.inj_file and len(samples) != len(self.inj_file):
            raise Exception("The number of results files does not match the "
                            "number of injection files")
        if not self.inj_file:
            self.inj_file = [None] * len(samples)
        for num, i in enumerate(samples):
            if not os.path.isfile(i):
                raise Exception("File %s does not exist" % (i))
            config = None
            if self.config:
                config = self.config[num]
            p, s, inj = self.convert_to_standard_format(
                i, self.inj_file[num], config_file=config)
            result_file_list.append(i)
            parameters_list.append(p)
            samples_list.append(s)
            injection_list.append(inj)
        self._result_files = result_file_list
        self._parameters = parameters_list
        self._samples = samples_list
        self._injection_data = injection_list

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, email):
        if email:
            self._email = email
        else:
            self._email = None

    @property
    def add_to_existing(self):
        return self._add_to_existing

    @add_to_existing.setter
    def add_to_existing(self, add_to_existing):
        self._add_to_existing = False
        if add_to_existing and not self.existing:
            raise Exception("Please provide a current html page that you "
                            "wish to add content to")
        if not add_to_existing and self.existing:
            logger.debug("Existing html page has been given without "
                         "specifying --add_to_existing flag. This is probably "
                         "and error and so manually adding --add_to_existing "
                         "flag")
            self._add_to_existing = True
        if add_to_existing and self.existing:
            if self.config:
                for i in glob(self.existing + "/config/*"):
                    self.config.append(i)
            self._add_to_existing = True

    @property
    def dump(self):
        return self._dump

    @dump.setter
    def dump(self, dump):
        self._dump = False
        if dump:
            self._dump = True

    @property
    def existing_labels(self):
        return self._existing_labels

    @existing_labels.setter
    def existing_labels(self, existing_labels):
        self._existing_labels = None
        if self.add_to_existing:
            existing = ExistingFile(self.existing)
            self._existing_labels = existing.existing_labels

    @property
    def existing_parameters(self):
        return self._existing_parameters

    @existing_parameters.setter
    def existing_parameters(self, existing_parameters):
        self._existing_parameters = None
        if self.add_to_existing:
            existing = ExistingFile(self.existing)
            self._existing_parameters = existing.existing_parameters

    @property
    def existing_samples(self):
        return self._existing_samples

    @existing_samples.setter
    def existing_samples(self, existing_samples):
        self._existing_samples = None
        if self.add_to_existing:
            existing = ExistingFile(self.existing)
            self._existing_samples = existing.existing_samples

    @property
    def existing_meta_file(self):
        if self.add_to_existing:
            existing = ExistingFile(self.existing)
            return existing.existing_file
        return None

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:
            if len(labels) != len(self.result_files):
                raise Exception(
                    "The number of labels does not match the number of results "
                    "files.")
            duplicated = dict(set(
                (x, labels.count(x)) for x in
                filter(lambda rec: labels.count(rec) > 1, labels)))
            if len(duplicated.keys()) >= 1:
                raise Exception("Please give a unique combination of labels")
            if self.add_to_existing:
                for i in labels:
                    if i in self.existing_labels:
                        raise Exception(
                            "The label '%s' already exists in the existing "
                            "file. Please pass another unique label")
            self._labels = labels
        else:
            self._labels = self._default_labels()
        logger.debug("The label is %s" % (self._labels))

    def make_directories(self):
        """Make the directorys in the web directory to store all information.
        """
        dirs = ["samples", "plots", "js", "html", "css", "plots/corner",
                "config"]
        for i in dirs:
            utils.make_dir(self.webdir + "/%s" % (i))

    def copy_files(self):
        """Copy the relevant files to the web directory.
        """
        logger.info("Copying the files to %s" % (self.webdir))
        path = pesummary.__file__[:-12]
        scripts = glob(path + "/core/js/*.js")
        for i in scripts:
            shutil.copyfile(i, self.webdir + "/js/%s" % (i.split("/")[-1]))
        scripts = glob(path + "/core/css/*.css")
        for i in scripts:
            shutil.copyfile(i, self.webdir + "/css/%s" % (i.split("/")[-1]))
        if self.config:
            for num, i in enumerate(self.config):
                if self.webdir not in i:
                    print(self.result_files[num])
                    shutil.copyfile(i, self.webdir + "/config/"
                                    + self.result_files[num].split("/")[-1] + "_"
                                    + i.split("/")[-1])

    def convert_to_standard_format(self, results_file, injection_file=None,
                                   config_file=None):
        """Convert a results file to standard form.

        Parameters
        ----------
        results_file: str
            Path to the results file that you wish to convert to standard
            form
        injection_file: str, optional
            Path to the injection file that was used in the analysis to
            produce the results_file
        config_file: str, optional
            Path to the configuration file that was used
        """
        f = OneFormat(results_file, injection_file, config=config_file)
        f.generate_all_posterior_samples()
        parameters = f.parameters
        samples = f.samples
        inj_p = f.injection_parameters
        inj_value = f.injection_values
        injection = {i: j for i, j in zip(inj_p, inj_value)}
        return parameters, samples, injection

    def _default_labels(self):
        """Return the defaut labels given your detector network.
        """
        label_list = []
        for num, i in enumerate(self.result_files):
            file_name = ".".join(i.split("/")[-1].split(".")[:-1])
            label_list.append("%s_%s" % (round(time()), file_name))

        duplicates = dict(set(
            (x, label_list.count(x)) for x in
            filter(lambda rec: label_list.count(rec) > 1, label_list)))

        for i in duplicates.keys():
            for j in range(duplicates[i]):
                ind = label_list.index(i)
                label_list[ind] += "_%s" % (j)
        if self.add_to_existing:
            for num, i in enumerate(label_list):
                if i in self.existing_labels:
                    ind = label_list.index(i)
                    label_list[ind] += "_%s" % (num)
        return label_list


class PostProcessing(object):
    """Class to extract parameters from the results files

    Parameters
    ----------
    inputs: argparser
        The parser containing the command line arguments
    colors: list, optional
        colors that you would like to use to display each results file in the
        webpage

    Attributes
    ----------
    parameters: list
        list of parameters that have posterior distributions for each results
        file
    injection_data: list
        list of dictionaries that contain the injection parameters and their
        injected value for each results file
    samples: list
        list of posterior samples for each parameter for each results file
    maxL_samples: list
        list of dictionaries that contain each parameter and their
        corresponding maximum likelihood value for each results file
    same_parameters: list
        List of parameters that all results files have sampled over
    """
    def __init__(self, inputs, colors="default"):
        self.inputs = inputs
        self.webdir = inputs.webdir
        self.baseurl = inputs.baseurl
        self.result_files = inputs.result_files
        self.dump = inputs.dump
        self.email = inputs.email
        self.user = inputs.user
        self.host = inputs.host
        self.config = inputs.config
        self.existing = inputs.existing
        self.add_to_existing = inputs.add_to_existing
        self.labels = inputs.labels
        self.hdf5 = inputs.hdf5
        self.existing_labels = inputs.existing_labels
        self.existing_parameters = inputs.existing_parameters
        self.existing_samples = inputs.existing_samples
        self.existing_meta_file = inputs.existing_meta_file
        self.colors = colors

        self.grab_data_map = {"existing_file": self._data_from_existing_file,
                              "standard_format": self._data_from_standard_format}

        self.parameters = inputs.parameters
        self.samples = inputs.samples
        self.injection_data = inputs.injection_data
        self.same_parameters = []

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        if colors == "default":
            self._colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00",
                            "#CA9161", "#FBAFE4", "#949494", "#ECE133",
                            "#56B4E9"]
        else:
            if not len(self.result_files) <= len(colors):
                raise Exception("Please give the same number of colors as "
                                "results files")
            self._colors = colors

    def _data_from_standard_format(self, result_file, index):
        """Extract data from a file of standard format

        Parameters
        ----------
        result_file: str
            path to the results file that you would like to extrct the
            information from
        index: int
            the index of the results file in self.result_files
        """
        f = h5py.File(result_file, "r")
        p = [i for i in f["posterior_samples/%s/parameter_names" % (
            self.labels[index])]]
        p = [i.decode("utf-8") for i in p]
        s = [i for i in f["posterior_samples/%s/samples" % (
            self.labels[index])]]
        inj_p = [i for i in f["posterior_samples/%s/injection_parameters" % (
            self.labels[index])]]
        inj_p = [i.decode("utf-8") for i in inj_p]
        inj_value = [i for i in f["posterior_samples/%s/injection_data" % (
            self.labels[index])]]
        injection = {i: j for i, j in zip(inj_p, inj_value)}
        return p, s, injection

    def _data_from_existing_file(self, result_file, index):
        """Extract data from a file in the existing format

        Parameters
        ----------
        result_file: str
            path to the results file that you would like to extrct the
            information from
        index: int
            the index of the results file in self.result_files
        """
        p = self.existing_parameters
        s = self.existing_samples
        injection = [
            {i: float("nan") for i in j} for j in self.existing_parameters]
        return p, s, injection

    @property
    def same_parameters(self):
        return self._same_parameters

    @same_parameters.setter
    def same_parameters(self, same_parameters):
        params = list(set.intersection(*[set(l) for l in self.parameters]))
        self._same_parameters = params
