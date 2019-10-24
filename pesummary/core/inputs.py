# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org>
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

import os
import socket
from glob import glob

import numpy as np
import pesummary
from pesummary.core.file.read import read as Read
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import SamplesDict, guess_url, logger, make_dir
from pesummary import conf


class Input(object):
    """Super class to handle the command line arguments

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options

    Attributes
    ----------
    result_files: list
        list of result files passed
    compare_results: list
        list of labels stored in the metafile that you wish to compare
    add_to_existing: Bool
        True if we are adding to an existing web directory
    existing_samples: dict
        dictionary of samples stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_injection_data: dict
        dictionary of injection data stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_file_version: dict
        dictionary of file versions stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_config: list
        list of configuration files stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_labels: list
        list of labels stored in an existing metafile. None if
        `self.add_to_existing` is False
    user: str
        the user who submitted the job
    webdir: str
        the directory to store the webpages, plots and metafile produced
    baseurl: str
        the base url of the webpages
    labels: list
        list of labels used to distinguish the result files
    config: list
        list of configuration files for each result file
    injection_file: list
        list of injection files for each result file
    publication: Bool
        if true, publication quality plots are generated. Default False
    kde_plot: Bool
        if true, kde plots are generated instead of histograms. Default False
    samples: dict
        dictionary of posterior samples stored in the result files
    priors: dict
        dictionary of prior samples stored in the result files
    custom_plotting: list
        list containing the directory and name of python file which contains
        custom plotting functions. Default None
    email: str
        the email address of the user
    dump: Bool
        if True, all plots will be dumped onto a single html page. Default False
    hdf5: Bool
        if True, the metafile is stored in hdf5 format. Default False
    notes: str
        notes that you wish to add to the webpages
    """
    def __init__(self, opts):
        logger.info("Command line arguments: %s" % (opts))
        self.opts = opts
        self.result_files = self.opts.samples
        self.existing = self.opts.existing
        self.compare_results = self.opts.compare_results
        self.add_to_existing = False
        if self.existing is not None:
            self.add_to_existing = True
            self.existing_metafile = None
            self.existing_data = self.grab_data_from_metafile(
                self.existing_metafile, self.existing,
                compare=self.compare_results
            )
            self.existing_samples = self.existing_data[0]
            self.existing_injection_data = self.existing_data[1]
            self.existing_file_version = self.existing_data[2]
            self.existing_file_kwargs = self.existing_data[3]
            self.existing_priors = self.existing_data[4]
            self.existing_config = self.existing_data[5]
            self.existing_labels = self.existing_data[6]
        else:
            self.existing_labels = None
            self.existing_samples = None
            self.existing_file_version = None
            self.existing_file_kwargs = None
            self.existing_priors = None
            self.existing_config = None
            self.existing_injection_data = None
        self.user = self.opts.user
        self.webdir = self.opts.webdir
        self.baseurl = self.opts.baseurl
        self.labels = self.opts.labels
        self.weights = {i: None for i in self.labels}
        self.config = self.opts.config
        self.injection_file = self.opts.inj_file
        self.publication = self.opts.publication
        self.make_directories()
        self.kde_plot = self.opts.kde_plot
        self.priors = self.opts.prior_file
        self.samples = self.opts.samples
        self.burnin = self.opts.burnin
        self.custom_plotting = self.opts.custom_plotting
        self.email = self.opts.email
        self.dump = self.opts.dump
        self.hdf5 = self.opts.save_to_hdf5
        self.palette = self.opts.palette
        self.include_prior = self.opts.include_prior
        self.colors = None
        self.notes = self.opts.notes
        self.copy_files()

    @staticmethod
    def is_pesummary_metafile(proposed_file):
        """Determine if a file is a PESummary metafile or not

        Parameters
        ----------
        proposed_file: str
            path to the file
        """
        extension = proposed_file.split(".")[-1]
        if extension == "h5" or extension == "hdf5" or extension == "hdf":
            from pesummary.core.file.read import is_pesummary_hdf5_file

            return is_pesummary_hdf5_file(proposed_file)
        elif extension == "json":
            from pesummary.core.file.read import is_pesummary_json_file

            return is_pesummary_json_file(proposed_file)
        else:
            return False

    @staticmethod
    def grab_data_from_metafile(existing_file, webdir, compare=None):
        """Grab data from an existing PESummary metafile

        Parameters
        ----------
        existing_file: str
            path to the existing metafile
        webdir: str
            the directory to store the existing configuration file
        compare: list, optional
            list of labels for events stored in an existing metafile that you
            wish to compare
        """
        f = Read(existing_file)
        labels = f.labels
        indicies = np.arange(len(labels))

        if compare:
            indicies = []
            for i in compare:
                if i not in labels:
                    raise InputError(
                        "Label '%s' does not exist in the metafile. The list "
                        "of available labels are %s" % (i, labels)
                    )
                indicies.append(labels.index(i))
            labels = compare

        DataFrame = f.samples_dict
        if f.injection_parameters != []:
            inj_values = f.injection_dict
        else:
            inj_values = {
                i: [float("nan")] * len(DataFrame[i]) for i in labels
            }
        for i in inj_values.keys():
            for param in inj_values[i].keys():
                if inj_values[i][param] == "nan":
                    inj_values[i][param] = float("nan")

        if hasattr(f, "priors") and f.priors != {}:
            priors = f.priors["samples"]
        else:
            priors = {label: {} for label in labels}

        config = []
        if f.config is not None and not all(i is None for i in f.config):
            config = []
            for i in labels:
                config_dir = os.path.join(webdir, "config")
                f.write_config_to_file(i, outdir=config_dir)
                config_file = os.path.join(
                    config_dir, "{}_config.ini".format(i)
                )
                config.append(config_file)
        else:
            for i in labels:
                config.append(None)

        if f.weights is not None:
            weights = {i: f.weights[i] for i in labels}
        else:
            weights = {i: None for i in labels}

        return [
            DataFrame, inj_values,
            {
                i: j for i, j in zip(
                    labels, [f.input_version[ind] for ind in indicies]
                )
            },
            {
                i: j for i, j in zip(
                    labels, [f.extra_kwargs[ind] for ind in indicies]
                )
            }, {}, config, labels, weights
        ]

    @staticmethod
    def grab_data_from_file(file, label, config=None, injection=None):
        """Grab data from a result file containing posterior samples

        Parameters
        ----------
        file: str
            path to the result file
        label: str
            label that you wish to use for the result file
        config: str, optional
            path to a configuration file used in the analysis
        injection: str, optional
            path to an injection file used in the analysis
        """
        f = Read(file)
        if config is not None:
            f.add_fixed_parameters_from_config_file(config)
        if injection:
            f.add_injection_parameters_from_file(injection)
        parameters = f.parameters
        samples = np.array(f.samples).T
        DataFrame = {label: SamplesDict(parameters, samples)}
        kwargs = f.extra_kwargs
        if hasattr(f, "injection_parameters"):
            injection = f.injection_parameters
            if injection is not None:
                for i in parameters:
                    if i not in list(injection.keys()):
                        injection[i] = float("nan")
            else:
                injection = {i: j for i, j in zip(
                    parameters, [float("nan")] * len(parameters))}
        else:
            injection = {i: j for i, j in zip(
                parameters, [float("nan")] * len(parameters))}
        version = f.input_version
        if hasattr(f, "priors"):
            priors = f.priors
        else:
            priors = None
        if hasattr(f, "weights"):
            weights = f.weights
        else:
            weights = None
        return [
            DataFrame, {label: injection}, {label: version}, {label: kwargs},
            {label: priors}, {label: weights}
        ]

    @property
    def existing(self):
        return self._existing

    @existing.setter
    def existing(self, existing):
        self._existing = existing
        if existing is not None:
            self._existing = os.path.abspath(existing)

    @property
    def existing_metafile(self):
        return self._existing_metafile

    @existing_metafile.setter
    def existing_metafile(self, existing_metafile):
        from glob import glob

        self._existing_metafile = existing_metafile
        if not os.path.isdir(os.path.join(self.existing, "samples")):
            raise InputError("Please provide a valid existing directory")
        files = glob(
            os.path.join(self.existing, "samples", "posterior_samples*")
        )
        if len(files) == 0:
            raise InputError(
                "Unable to find an existing metafile in the existing webdir"
            )
        if len(files) > 1:
            raise InputError(
                "Multiple metafiles in the existing directory. Please either "
                "run the `summarycombine_metafile` executable to combine the "
                "meta files or simple remove the unwanted meta file"
            )
        self._existing_metafile = os.path.join(
            self.existing, "samples", files[0]
        )

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user):
        try:
            self._user = os.environ["USER"]
            logger.info(
                conf.overwrite.format("user", conf.user, self._user)
            )
        except Exception as e:
            logger.info(
                "Failed to grab user information because {}. Default will be "
                "used".format(e)
            )
            self._user = user

    @property
    def host(self):
        return socket.getfqdn()

    @property
    def webdir(self):
        return self._webdir

    @webdir.setter
    def webdir(self, webdir):
        cond1 = webdir is None or webdir == "None" or webdir == "none"
        cond2 = (
            self.existing is None or self.existing == "None"
            or self.existing == "none"
        )
        if cond1 and cond2:
            raise InputError(
                "Please provide a web directory to store the webpages. If "
                "you wish to add to an existing webpage, then pass the "
                "existing web directory under the '--existing_webdir' command "
                "line argument. If this is a new set of webpages, then pass "
                "the web directory under the '--webdir' argument"
            )
        elif webdir is None and self.existing is not None:
            if not os.path.isdir(self.existing):
                raise InputError(
                    "The directory {} does not exist".format(self.existing)
                )
            entries = glob(self.existing + "/*")
            if os.path.join(self.existing, "home.html") not in entries:
                raise InputError(
                    "Please give the base directory of an existing output"
                )
            self._webdir = self.existing
        else:
            if not os.path.isdir(webdir):
                logger.debug(
                    "Given web directory does not exist. Creating it now"
                )
                make_dir(webdir)
            self._webdir = os.path.abspath(webdir)

    @property
    def baseurl(self):
        return self._baseurl

    @baseurl.setter
    def baseurl(self, baseurl):
        self._baseurl = baseurl
        if baseurl is None:
            self._baseurl = guess_url(self.webdir, self.host, self.user)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            labels = self.default_labels()
        elif len(np.unique(labels)) != len(labels):
            raise InputError(
                "Please provide unique labels for each result file"
            )
        for num, i in enumerate(labels):
            if "." in i:
                logger.warn(
                    "Replacing the label {} by {} to make it compatible with "
                    "the html pages".format(i, i.replace(".", "_"))
                )
                labels[num] = i.replace(".", "_")
        if self.add_to_existing:
            for i in labels:
                if i in self.existing_labels:
                    raise InputError(
                        "The label '%s' already exists in the existing "
                        "metafile. Please pass another unique label"
                    )
        self._labels = labels

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        if config and len(config) != len(self.labels):
            raise InputError(
                "Please provide a configuration file for each label"
            )
        if config is None:
            config = [None] * len(self.labels)
        self._config = config

    @property
    def injection_file(self):
        return self._injection_file

    @injection_file.setter
    def injection_file(self, injection_file):
        if injection_file and len(injection_file) != len(self.labels):
            if len(injection_file) == 1:
                logger.info(
                    "Only one injection file passed. Assuming the same "
                    "injection for all {} result files".format(len(self.labels))
                )
            else:
                raise InputError(
                    "You have passed {} for {} result files. Please provide an "
                    "injection file for each result file".format(
                        len(self.injection_file), len(self.labels)
                    )
                )
        if injection_file is None:
            injection_file = [None] * len(self.labels)
        self._injection_file = injection_file

    @property
    def injection_data(self):
        return self._injection_data

    @property
    def file_version(self):
        return self._file_version

    @property
    def file_kwargs(self):
        return self._file_kwargs

    @property
    def kde_plot(self):
        return self._kde_plot

    @kde_plot.setter
    def kde_plot(self, kde_plot):
        self._kde_plot = kde_plot
        if kde_plot != conf.kde_plot:
            logger.info(
                conf.overwrite.format("kde_plot", conf.kde_plot, kde_plot)
            )

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if not samples:
            raise InputError("Please provide a results file")
        if len(samples) != len(self.labels):
            logger.info(
                "You have passed {} result files and {} labels. Setting "
                "labels = {}".format(
                    len(samples), len(self.labels), self.labels[:len(samples)]
                )
            )
            self.labels = self.labels[:len(samples)]
        samples_dict, injection_data_dict, prior_dict = {}, {}, {}
        file_version_dict, file_kwargs_dict, weights_dict = {}, {}, {}
        config, labels = None, None
        for num, i in enumerate(samples):
            if not os.path.isfile(i):
                raise Exception("File %s does not exist" % (i))
            data = self.grab_data_from_input(
                i, self.labels[num], config=self.config[num],
                injection=self.injection_file[num]
            )
            if len(data) > 6:
                config = data[5]
                labels = data[6]
                weights_dict = data[7]
                prior_dict = data[4]
                for j in labels:
                    samples_dict[j] = data[0][j]
                    injection_data_dict[j] = data[1][j]
                    file_version_dict[j] = data[2][j]
                    file_kwargs_dict[j] = data[3][j]
            else:
                samples_dict[self.labels[num]] = data[0][self.labels[num]]
                injection_data_dict[self.labels[num]] = data[1][self.labels[num]]
                file_version_dict[self.labels[num]] = data[2][self.labels[num]]
                file_kwargs_dict[self.labels[num]] = data[3][self.labels[num]]
                prior_dict[self.labels[num]] = data[4][self.labels[num]]
                weights_dict[self.labels[num]] = data[5][self.labels[num]]
        self._samples = samples_dict
        self._injection_data = injection_data_dict
        self._file_version = file_version_dict
        self._file_kwargs = file_kwargs_dict
        if labels is not None:
            labels_iter = labels
        else:
            labels_iter = self.labels
        for i in labels_iter:
            if prior_dict != {} and prior_dict[i] != []:
                if self.priors != {} and i in self.priors["samples"].keys():
                    logger.warn(
                        "Replacing the prior file for {} with the prior "
                        "samples stored in the result file".format(i)
                    )
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors != {}:
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
            elif prior_dict != {}:
                if self.priors != {} and i not in self.priors["samples"].keys():
                    self.add_to_prior_dict("samples/" + i, prior_dict[i])
                elif self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, [])
            else:
                if self.priors == {}:
                    self.add_to_prior_dict("samples/" + i, [])
        if config is not None:
            self._config = config
        if labels is not None:
            self.labels = labels
            self.result_files = self.result_files * len(labels)
            self.weights = {i: None for i in self.labels}
        if weights_dict != {}:
            self.weights = weights_dict

    @property
    def burnin(self):
        return self._burnin

    @burnin.setter
    def burnin(self, burnin):
        if burnin is not None:
            samples_lengths = [
                self.samples[key].number_of_samples for key in
                self.samples.keys()
            ]
            if not all(int(burnin) < i for i in samples_lengths):
                raise InputError(
                    "The chosen burnin is larger than the number of samples. "
                    "Please choose a value less than {}".format(
                        np.max(samples_lengths)
                    )
                )
            logger.info(
                conf.overwrite.format("burnin", conf.burnin, burnin)
            )
            conf.burnin = burnin
        for label in self.samples:
            self.samples[label] = self.samples[label].discard_samples(
                int(conf.burnin)
            )

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        self._priors = self.grab_priors_from_inputs(priors)

    @property
    def custom_plotting(self):
        return self._custom_plotting

    @custom_plotting.setter
    def custom_plotting(self, custom_plotting):
        self._custom_plotting = custom_plotting
        if custom_plotting is not None:
            import importlib

            path_to_python_file = os.path.dirname(custom_plotting)
            python_file = os.path.splitext(os.path.basename(custom_plotting))[0]
            if path_to_python_file != "":
                import sys

                sys.path.append(path_to_python_file)
            try:
                mod = importlib.import_module(python_file)
                methods = getattr(mod, "__single_plots__", list()).copy()
                methods += getattr(mod, "__comparion_plots__", list()).copy()
                if len(methods) > 0:
                    self._custom_plotting = [path_to_python_file, python_file]
                else:
                    logger.warn(
                        "No __single_plots__ or __comparison_plots__ in {}. "
                        "If you wish to use custom plotting, then please "
                        "add the variable :__single_plots__ and/or "
                        "__comparison_plots__ in future. No custom plotting "
                        "will be done"
                    )
            except Exception as e:
                logger.warn(
                    "Failed to import {} because {}. No custom plotting will "
                    "be done".format(python_file, e)
                )

    def add_to_prior_dict(self, path, data):
        """Add priors to the prior dictionary

        Parameters
        ----------
        path: str
            the location where you wish to store the prior. If this is inside
            a nested dictionary, then please pass the path as 'a/b'
        data: np.ndarray
            the prior samples
        """
        from functools import reduce

        def build_tree(dictionary, path):
            """Build a dictionary tree from a list of keys

            Parameters
            ----------
            dictionary: dict
                existing dictionary that you wish to add to
            path: list
                list of keys specifying location

            Examples
            --------
            >>> dictionary = {"label": {"mass_1": [1,2,3,4,5,6]}}
            >>> path = ["label", "mass_2"]
            >>> build_tree(dictionary, path)
            {"label": {"mass_1": [1,2,3,4,5,6], "mass_2": {}}}
            """
            if path != [] and path[0] not in dictionary.keys():
                dictionary[path[0]] = {}
            if path != []:
                build_tree(dictionary[path[0]], path[1:])
            return dictionary

        def get_nested_dictionary(dictionary, path):
            """Return a nested dictionary from a list specifying path

            Parameters
            ----------
            dictionary: dict
                existing dictionary that you wish to extract information from
            path: list
                list of keys specifying location

            Examples
            --------
            >>> dictionary = {"label": {"mass_1": [1,2,3,4,5,6]}}
            >>> path = ["label", "mass_1"]
            >>> get_nested_dictionary(dictionary, path)
            [1,2,3,4,5,6]
            """
            return reduce(dict.get, path, dictionary)

        if "/" in path:
            path = path.split("/")
        else:
            path = [path]
        tree = build_tree(self._priors, path)
        nested_dictionary = get_nested_dictionary(self._priors, path[:-1])
        nested_dictionary[path[-1]] = data

    def grab_priors_from_inputs(self, priors):
        """
        """
        from pesummary.core.file.read import read as Read

        prior_dict = {}
        if priors is not None:
            prior_dict = {"samples": {}}
            for i in priors:
                if not os.path.isfile(i):
                    raise InputError("The file {} does not exist".format(i))
            if len(priors) != len(self.labels) and len(priors) == 1:
                logger.warn(
                    "You have only specified a single prior file for {} result "
                    "files. Assuming the same prior file for all result "
                    "files".format(len(self.labels))
                )
                data = Read(priors[0])
                for i in self.labels:
                    prior_dict["samples"][i] = data.samples_dict
            elif len(priors) != len(self.labels):
                raise InputError(
                    "Please provide a prior file for each result file"
                )
            else:
                for num, i in enumerate(priors):
                    data = Read(priors[0])
                    prior_dict["samples"][self.labels[num]] = data.samples_dict
        return prior_dict

    def grab_data_from_input(self, file, label, config=None, injection=None):
        """Wrapper function for the grab_data_from_metafile and
        grab_data_from_file functions

        Parameters
        ----------
        file: str
            path to the result file
        label: str
            label that you wish to use for the result file
        config: str, optional
            path to a configuration file used in the analysis
        injection: str, optional
            path to an injection file used in the analysis
        """
        if self.is_pesummary_metafile(file):
            existing_data = self.grab_data_from_metafile(
                file, self.webdir, compare=self.compare_results
            )
            return existing_data
        else:
            data = self.grab_data_from_file(
                file, label, config=config, injection=injection
            )
            return data

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, email):
        if email is not None and "@" not in email:
            raise InputError("Please provide a valid email address")
        self._email = email

    @property
    def dump(self):
        return self._dump

    @dump.setter
    def dump(self, dump):
        self._dump = dump

    @property
    def palette(self):
        return self._palette

    @palette.setter
    def palette(self, palette):
        self._palette = palette
        if palette is not conf.palette:
            import seaborn

            try:
                seaborn.color_palette(palette, n_colors=1)
                logger.info(
                    conf.overwrite.format("palette", conf.palette, palette)
                )
                conf.palette = palette
            except Exception as e:
                raise InputError(
                    "Unrecognised palette. Please choose from one of the "
                    "following {}".format(
                        ", ".join(seaborn.palettes.SEABORN_PALETTES.keys())
                    )
                )

    @property
    def include_prior(self):
        return self._include_prior

    @include_prior.setter
    def include_prior(self, include_prior):
        self._include_prior = include_prior
        if include_prior != conf.include_prior:
            conf.overwrite.format("prior", conf.include_prior, include_prior)
            conf.include_prior = include_prior

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        import seaborn

        number = len(self.labels)
        if self.existing:
            number += len(self.existing_labels)
        self._colors = seaborn.color_palette(
            palette=conf.palette, n_colors=number
        ).as_hex()

    @property
    def notes(self):
        return self._notes

    @notes.setter
    def notes(self, notes):
        self._notes = notes
        if notes is not None:
            if not os.path.isfile(notes):
                raise InputError(
                    "No such file or directory called {}".format(notes)
                )
            try:
                with open(notes, "r") as f:
                    self._notes = f.read()
            except Exception as e:
                logger.warn(
                    "Failed to read {}. Unable to put notes on "
                    "summarypages".format(notes)
                )

    def make_directories(self):
        """Make the directories to store the information
        """
        dirs = [
            "samples", "plots", "js", "html", "css", "plots/corner", "config"
        ]
        if self.publication:
            dirs.append("plots/publication")
        for i in dirs:
            if not os.path.isdir(i):
                make_dir(os.path.join(self.webdir, i))

    def copy_files(self):
        """Copy the relevant file to the web directory
        """
        import shutil

        path = pesummary.__file__[:-12]
        scripts = glob(os.path.join(path, "core/js/*.js"))
        for i in scripts:
            shutil.copyfile(
                i, os.path.join(
                    self.webdir, "js", os.path.basename(i)
                )
            )
        scripts = glob(path + "/core/css/*.css")
        for i in scripts:
            shutil.copyfile(
                i, os.path.join(
                    self.webdir, "css", os.path.basename(i)
                )
            )
        if not all(i is None for i in self.config):
            for num, i in enumerate(self.config):
                if self.webdir not in i:
                    filename = "_".join(
                        [self.labels[num], "config.ini"]
                    )
                    shutil.copyfile(
                        i, os.path.join(
                            self.webdir, "config", filename
                        )
                    )

    def default_labels(self):
        """Return a list of default labels.
        """
        from time import time

        label_list = []
        if self.result_files is None or len(self.result_files) == 0:
            raise InputError("Please provide a results file")
        for num, i in enumerate(self.result_files):
            file_name = os.path.splitext(os.path.basename(i))[0]
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
    """Super class to post process the input data

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files

    Attributes
    ----------
    result_files: list
        list of result files passed
    compare_results: list
        list of labels stored in the metafile that you wish to compare
    add_to_existing: Bool
        True if we are adding to an existing web directory
    existing_samples: dict
        dictionary of samples stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_injection_data: dict
        dictionary of injection data stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_file_version: dict
        dictionary of file versions stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_config: list
        list of configuration files stored in an existing metafile. None if
        `self.add_to_existing` is False
    existing_labels: list
        list of labels stored in an existing metafile. None if
        `self.add_to_existing` is False
    user: str
        the user who submitted the job
    webdir: str
        the directory to store the webpages, plots and metafile produced
    baseurl: str
        the base url of the webpages
    labels: list
        list of labels used to distinguish the result files
    config: list
        list of configuration files for each result file
    injection_file: list
        list of injection files for each result file
    publication: Bool
        if true, publication quality plots are generated. Default False
    kde_plot: Bool
        if true, kde plots are generated instead of histograms. Default False
    samples: dict
        dictionary of posterior samples stored in the result files
    priors: dict
        dictionary of prior samples stored in the result files
    custom_plotting: list
        list containing the directory and name of python file which contains
        custom plotting functions. Default None
    email: str
        the email address of the user
    dump: Bool
        if True, all plots will be dumped onto a single html page. Default False
    hdf5: Bool
        if True, the metafile is stored in hdf5 format. Default False
    same_parameters: dict
        list of parameters that are common in all result files
    """
    def __init__(self, inputs, colors="default"):
        self.inputs = inputs
        self.result_files = self.inputs.result_files
        self.existing = self.inputs.existing
        self.compare_results = self.inputs.compare_results
        self.add_to_existing = False
        if self.existing is not None:
            self.add_to_existing = True
            self.existing_metafile = self.inputs.existing_metafile
            self.existing_samples = self.inputs.existing_samples
            self.existing_injection_data = self.inputs.existing_injection_data
            self.existing_file_version = self.inputs.existing_file_version
            self.existing_file_kwargs = self.inputs.existing_file_kwargs
            self.existing_priors = self.inputs.existing_priors
            self.existing_config = self.inputs.existing_config
            self.existing_labels = self.inputs.existing_labels
        else:
            self.existing_metafile = None
            self.existing_labels = None
            self.existing_samples = None
            self.existing_file_version = None
            self.existing_file_kwargs = None
            self.existing_priors = None
            self.existing_config = None
            self.existing_injection_data = None
        self.user = self.inputs.user
        self.host = self.inputs.host
        self.webdir = self.inputs.webdir
        self.baseurl = self.inputs.baseurl
        self.labels = self.inputs.labels
        self.weights = self.inputs.weights
        self.config = self.inputs.config
        self.injection_file = self.inputs.injection_file
        self.injection_data = self.inputs.injection_data
        self.publication = self.inputs.publication
        self.kde_plot = self.inputs.kde_plot
        self.samples = self.inputs.samples
        self.priors = self.inputs.priors
        self.custom_plotting = self.inputs.custom_plotting
        self.email = self.inputs.email
        self.dump = self.inputs.dump
        self.hdf5 = self.inputs.hdf5
        self.file_version = self.inputs.file_version
        self.file_kwargs = self.inputs.file_kwargs
        self.palette = self.inputs.palette
        self.colors = self.inputs.colors
        self.include_prior = self.inputs.include_prior
        self.notes = self.inputs.notes
        self.same_parameters = []

    @property
    def same_parameters(self):
        return self._same_parameters

    @same_parameters.setter
    def same_parameters(self, same_parameters):
        parameters = [list(self.samples[key]) for key in self.samples.keys()]
        params = list(set.intersection(*[set(l) for l in parameters]))
        self._same_parameters = params
