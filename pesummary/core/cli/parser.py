# Licensed under an MIT style license -- see LICENSE.md

import sys
import argparse
from ... import conf
from ...utils.utils import logger
from .actions import (
    ConfigAction, CheckFilesExistAction, DictionaryAction,
    DeprecatedStoreFalseAction
)

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class ArgumentParser(argparse.ArgumentParser):
    """Extension of the argparse.ArgumentParser object to handle pesummary
    command line arguments.

    Properties
    ----------
    fallback_options: dict
        dictionary of default kwargs
    pesummary_options: dict
        dictionary giving pesummary options
    command_line: str
        command line run
    command_line_arguments: list
        list giving command line arguments
    dynamic_argparse: list
        list of dynamic argparse functions that you wish to add to the
        argparse.Namespace object
    """
    @property
    def fallback_options(self):
        return {
            "default": None,
            "nargs": None,
            "dest": None,
            "type": str,
            "action": argparse._StoreAction,
            "choices": None
        }

    @property
    def pesummary_options(self):
        return self._pesummary_options()

    def update_default_pesummary_option(self, option, key, value):
        """Update the default dictionary of pesummary option

        Parameters
        ----------
        option: str
            Name of pesummary option to change.
        key: str
            Name of kwarg you wish to change.
        value: str/float/func
            str/float/func you with to change kwarg to
        """
        if option not in self._pesummary_options().keys():
            raise ValueError("Unknown option '{}'".format(option))
        self._pesummary_options()[option][key] = value
        return self._pesummary_options()

    def _pesummary_options(self):
        return {
            "pesummary": {
                "nargs": "?",
                "action": ConfigAction,
                "help": "Configuration file containing command line arguments",
                "key": "core",
            },
            "--webdir": {
                "dest": "webdir",
                "metavar": "DIR",
                "help": "make page and plots in DIR",
                "short": "-w",
                "key": "core",
            },
            "--baseurl": {
                "dest": "baseurl",
                "metavar": "DIR",
                "help": "make the page at this url",
                "short": "-b",
                "key": "core",
            },
            "--labels": {
                "dest": "labels",
                "help": "labels used to distinguish runs",
                "nargs": "+",
                "key": "core",
            },
            "--samples": {
                "dest": "samples",
                "nargs": "+",
                "action": CheckFilesExistAction,
                "help": (
                    "Path to posterior samples file(s). See documentation for "
                    "allowed formats. If path is on a remote server, add "
                    "username and servername in the form "
                    "{username}@{servername}:{path}. If path is on a public "
                    "webpage, ensure the path starts with https://. You may "
                    "also pass a string such as posterior_samples*.dat and "
                    "all matching files will be used"
                ),
                "short": "-s",
                "key": "core",
            },
            "--config": {
                "dest": "config",
                "nargs": "+",
                "action": CheckFilesExistAction,
                "help": "configuration file associcated with each samples file.",
                "short": "-c",
                "key": "core",
            },
            "--email": {
                "action": "store",
                "help": (
                    "send an e-mail to the given address with a link to the "
                    "finished page."
                ),
                "key": "core",
            },
            "--inj_file": {
                "dest": "inj_file",
                "nargs": "+",
                "action": CheckFilesExistAction,
                "help": "path to injetcion file",
                "short": "-i",
                "key": "core",
            },
            "--user": {
                "dest": "user",
                "help": argparse.SUPPRESS,
                "default": conf.user,
                "key": "core",
            },
            "--testing": {
                "action": "store_true",
                "help": argparse.SUPPRESS,
                "default": False,
                "key": "core",
            },
            "--add_to_existing": {
                "action": "store_true",
                "default": False,
                "help": "add new results to an existing html page",
                "key": "core",
            },
            "--existing_webdir": {
                "dest": "existing",
                "help": "web directory of existing output",
                "short": "-e",
                "key": "core",
            },
            "--seed": {
                "dest": "seed",
                "default": 123456789,
                "type": int,
                "help": "Random seed to used through the analysis.",
                "key": "core",
            },
            "--verbose": {
                "action": "store_true",
                "help": "print useful information for debugging purposes",
                "short": "-v",
                "key": "core",
            },
            "--preferred": {
                "dest": "preferred",
                "help": (
                    "label of the preferred run. If only one result file is "
                    "passed this label is the preferred analysis by default"
                ),
                "key": "core",
            },
            "--ignore_parameters": {
                "dest": "ignore_parameters",
                "nargs": "+",
                "help": (
                    "Parameters that you wish to not include in the "
                    "summarypages. You may list them or use wildcards "
                    "('recalib*')"
                ),
                "key": "samples",
            },
            "--nsamples": {
                "dest": "nsamples",
                "help": (
                    "The number of samples to use and store in the PESummary "
                    "metafile. These samples will be randomly drawn from the "
                    "posterior distributions"
                ),
                "key": "samples",
            },
            "--keep_nan_likelihood_samples": {
                "dest": "keep_nan_likelihood_samples",
                "action": "store_true",
                "default": False,
                "help": (
                    "Do not remove posterior samples where the likelihood="
                    "'nan'. Without this option, posterior samples where the "
                    "likelihood='nan' are removed by default."
                ),
                "key": "samples",
            },
            "--burnin": {
                "dest": "burnin",
                "help": "Number of samples to remove as burnin",
                "key": "samples",
            },
            "--burnin_method": {
                "dest": "burnin_method",
                "help": (
                    "The algorithm to use to remove mcmc samples as burnin. "
                    "This is only used when `--mcmc_samples` also used"
                ),
                "key": "samples",
            },
            "--regenerate": {
                "dest": "regenerate",
                "nargs": "+",
                "help": (
                    "List of posterior distributions that you wish to "
                    "regenerate if possible"
                ),
                "key": "samples",
            },
            "--mcmc_samples": {
                "action": "store_true",
                "default": False,
                "help": (
                    "treat the passed samples as seperate mcmc chains for the "
                    "same analysis"
                ),
                "key": "samples",
            },
            "--path_to_samples": {
                "nargs": "+",
                "help": (
                    "Path to the posterior samples stored in the result file. "
                    "If None, pesummary will search for a 'posterior' or "
                    "'posterior_samples' group. If more than one result file "
                    "is passed, and only the third file requires a "
                    "path_to_samples provide --path_to_samples None None "
                    "path/to/samples"
                ),
                "key": "samples",
            },
            "--pe_algorithm": {
                "nargs": "+",
                "help": "Name of the algorithm used to generate the result file",
                "key": "samples",
            },
            "--reweight_samples": {
                "default": False,
                "help": (
                    "Method to use when reweighting posterior and/or prior "
                    "samples. Default do not reweight samples."
                ),
                "key": "samples",
            },
            "--descriptions": {
                "default": {},
                "action": DictionaryAction,
                "nargs": "+",
                "help": (
                    "JSON file containing a set of descriptions for each "
                    "analysis or dictionary giving descriptions for each "
                    "analysis directly from the command line (e.g. "
                    "`--descriptions label1:'description'`). These "
                    "descriptions are then saved in the output."
                ),
                "key": "samples",
            },
            "--custom_plotting": {
                "dest": "custom_plotting",
                "help": "Python file containing functions for custom plotting",
                "key": "plot",
            },
            "--publication": {
                "action": "store_true",
                "help": "generate production quality plots",
                "key": "plot",
            },
            "--publication_kwargs": {
                "action": DictionaryAction,
                "nargs": "+",
                "default": {},
                "help": "Optional kwargs for publication plots",
                "key": "plot",
            },
            "--kde_plot": {
                "action": "store_true",
                "default": False,
                "help": "plot a kde rather than a histogram",
                "key": "plot",
            },
            "--colors": {
                "dest": "colors",
                "nargs": "+",
                "help": "Colors you wish to use to distinguish result files",
                "key": "plot",
            },
            "--palette": {
                "dest": "palette",
                "default": "colorblind",
                "help": "Color palette to use to distinguish result files",
                "key": "plot",
            },
            "--linestyles": {
                "dest": "linestyles",
                "nargs": "+",
                "help": "Linestyles you wish to use to distinguish result files",
                "key": "plot",
            },
            "--include_prior": {
                "action": "store_true",
                "default": False,
                "help": "Plot the prior on the same plot as the posterior",
                "key": "plot",
            },
            "--style_file": {
                "dest": "style_file",
                "action": CheckFilesExistAction,
                "help": "Style file you wish to use when generating plots",
                "key": "plot",
            },
            "--add_to_corner": {
                "dest": "add_to_corner",
                "nargs": "+",
                "help": "Parameters you wish to include in the corner plot",
                "key": "plot",
            },
            "--add_existing_plot": {
                "dest": "existing_plot",
                "nargs": "+",
                "action": DictionaryAction,
                "help": (
                    "Path(s) to existing plots that you wish to add to the "
                    "summarypages. Should be of the form {label}:{path}"
                ),
                "key": "plot",
            },
            "--dump": {
                "action": "store_true",
                "default": False,
                "help": "dump all information onto a single html page",
                "key": "webpage",
            },
            "--notes": {
                "dest": "notes",
                "help": (
                    "Single file containing notes that you wish to put on "
                    "summarypages"
                ),
                "key": "webpage",
            },
            "--prior_file": {
                "dest": "prior_file",
                "nargs": "+",
                "action": CheckFilesExistAction,
                "help": "File containing for the prior samples for a given label",
                "key": "prior",
            },
            "--nsamples_for_prior": {
                "dest": "nsamples_for_prior",
                "default": 5000,
                "type": int,
                "help": (
                    "The number of prior samples to extract from a bilby prior "
                    "file or a bilby result file"
                ),
                "key": "prior",
            },
            "--disable_prior_sampling": {
                "action": "store_true",
                "default": False,
                "help": "Skip generating prior samples using bilby",
                "key": "prior"
            },
            "--disable_comparison": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Whether to make a comparison webpage if multple results "
                    "are present"
                ),
                "key": "performance",
            },
            "--disable_interactive": {
                "action": "store_true",
                "default": False,
                "help": "Whether to make interactive plots or not",
                "key": "performance",
            },
            "--disable_corner": {
                "action": "store_true",
                "default": False,
                "help": "Whether to make a corner plot or not",
                "key": "performance",
            },
            "--enable_expert": {
                "action": "store_true",
                "default": False,
                "help": "Whether to generate extra diagnostic plots or not",
                "key": "performance",
            },
            "--disable_expert": {
                "action": DeprecatedStoreFalseAction(
                    new_option="--enable_expert"
                ),
                "dest": "enable_expert",
                "default": False,
                "help": "Whether to generate extra diagnostic plots or not",
                "key": "performance",
            },
            "--multi_process": {
                "dest": "multi_process",
                "default": 1,
                "help": "The number of cores to use when generating plots",
                "key": "performance",
            },
            "--file_format": {
                "dest": "file_format",
                "nargs": "+",
                "help": "The file format of each result file.",
                "key": "performance",
            },
            "--restart_from_checkpoint": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Restart from checkpoint if a checkpoint file can be found "
                    "in webdir"
                ),
                "key": "performance",
            },
            "--compare_results": {
                "dest": "compare_results",
                "nargs": "+",
                "help": (
                    "labels for events stored in the posterior_samples.json "
                    "that you wish to compare"
                ),
                "key": "metafile",
            },
            "--save_to_json": {
                "action": "store_true",
                "default": False,
                "help": "save the meta file in json format",
                "key": "metafile",
            },
            "--posterior_samples_filename": {
                "dest": "filename",
                "help": "name of the posterior samples metafile that is produced",
                "key": "metafile",
            },
            "--external_hdf5_links": {
                "action": "store_true",
                "default": False,
                "help": (
                    "save each analysis as a seperate hdf5 file and connect "
                    "them to the meta file through external links"
                ),
                "key": "metafile",
            },
            "--hdf5_compression": {
                "dest": "hdf5_compression",
                "type": int,
                "help": (
                    "compress each dataset with a particular compression "
                    "filter. Filter must be integer between 0 and 9. Only "
                    "applies to meta files stored in hdf5 format. Default, no "
                    "compression applied"
                ),
                "key": "metafile",
            },
            "--disable_injection": {
                "action": "store_true",
                "default": False,
                "help": (
                    "whether or not to extract stored injection data from the "
                    "meta file"
                ),
                "key": "metafile",
            },
        }

    def add_known_options_to_parser_from_key(self, parser, key):
        options = [
            option for option, _dict in self.pesummary_options.items() if
            _dict["key"] == key
        ]
        return self.add_known_options_to_parser(options, parser=parser)

    def add_core_group(self):
        core_group = self.add_argument_group(
            "Core command line options\n"
            "-------------------------"
        )
        return self.add_known_options_to_parser_from_key(core_group, "core")

    def add_samples_group(self):
        sample_group = self.add_argument_group(
            "Options specific to the samples you wish to input\n"
            "-------------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(sample_group, "samples")

    def add_plot_group(self):
        plot_group = self.add_argument_group(
            "Options specific to the plots you wish to make\n"
            "----------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(plot_group, "plot")

    def add_webpage_group(self):
        webpage_group = self.add_argument_group(
            "Options specific to the webpages you wish to produce\n"
            "----------------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(webpage_group, "webpage")

    def add_prior_group(self):
        prior_group = self.add_argument_group(
            "Options specific for passing prior files\n"
            "----------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(prior_group, "prior")

    def add_performance_group(self):
        performance_group = self.add_argument_group(
            "Options specific for enhancing the performance of the code\n"
            "----------------------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(
            performance_group, "performance"
        )

    def add_metafile_group(self):
        metafile_group = self.add_argument_group(
            "Options specific for reading and manipulating pesummary metafiles\n"
            "-----------------------------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(metafile_group, "metafile")

    def add_all_groups_to_parser(self):
        self.add_core_group()
        self.add_samples_group()
        self.add_plot_group()
        self.add_webpage_group()
        self.add_prior_group()
        self.add_performance_group()
        self.add_metafile_group()

    def add_all_known_options_to_parser(self):
        """Add all known pesummary options to the parser
        """
        return self.add_known_options_to_parser(self.pesummary_options.keys())

    def add_additional_options_to_parser(self, options, parser=None):
        """Add additional options to the parser

        Parameters
        ----------
        options: dict
            dictionary containing options. Key should be the option name
            and value should be option kwargs.
        parser: ArgumentParser, optional
            parser to use when adding additional options. Default self
        """
        import inspect
        if parser is None:
            parser = self
        for name, kwargs in options.items():
            _kwargs = self.fallback_options.copy()
            _kwargs.update(kwargs)
            if "short" in _kwargs.keys():
                args = [_kwargs["short"], name]
            else:
                args = [name]
            if "key" in _kwargs.keys():
                _kwargs.pop("key")
            if "-" not in name:
                # must be a positional argument
                _kwargs.pop("dest")

            action = _kwargs.get('action', None)
            action_class = self._registry_get('action', action, action)
            arglist = inspect.getfullargspec(action_class).args
            keys = _kwargs.copy().keys()
            for key in keys:
                if (key not in arglist) and (key != "action"):
                    _kwargs.pop(key)
            parser.add_argument(*args, **_kwargs)
        return parser

    def add_known_options_to_parser(self, options, parser=None):
        """Add a list of known options to the parser

        Parameters
        ----------
        options: list
            list of option names that you wish to add to the parser. Option
            names must be known to the pesummary ArgumentParser
        parser: ArgumentParser, optional
            parser to use when adding additional options. Default self
        """
        _options = {}
        for option in options:
            if option not in self.pesummary_options.keys():
                raise ValueError("Unknown option '{}'".format(option))
            _options[option] = self.pesummary_options[option]
        return self.add_additional_options_to_parser(_options, parser=parser)

    @property
    def dynamic_argparse(self):
        return []

    @property
    def command_line_arguments(self):
        return sys.argv[1:]

    @property
    def command_line(self):
        return "{} {}".format(self.prog, " ".join(self.command_line_arguments))

    @staticmethod
    def intersection(a, b):
        """Return the common entries between two lists
        """
        return list(set(a).intersection(set(b)))

    def parse_known_args(
        self, *args, logger_level={"known": "info", "unknown": "warn"}, **kwargs
    ):
        """Parse known command line arguments and return unknown quantities

        Parameters
        ----------
        args: list, optional
            optional list of command line arguments you wish to pass
        logger_level: dict, optional
            dictionary containing the logger levels to use when printing the
            known and unknown quantities to stdout. Default
            {"known": "info", "unknown": "warn"}
        """
        opts, unknown = super(ArgumentParser, self).parse_known_args(
            *args, **kwargs
        )
        for dynamic in self.dynamic_argparse:
            _, _unknown = dynamic(opts, command_line=kwargs.get("args", None))
            unknown = self.intersection(unknown, _unknown)
        getattr(logger, logger_level["known"])(
            "Command line arguments: %s" % (opts)
        )
        if len(unknown):
            getattr(logger, logger_level["unknown"])(
                "Unknown command line arguments: {}".format(
                    " ".join([cmd for cmd in unknown if "--" in cmd])
                )
            )
        return opts, unknown


def convert_dict_to_namespace(dictionary, add_defaults=None):
    """Convert a dictionary to an argparse.Namespace object

    Parameters
    ----------
    dictionary: dict
        dictionary you wish to convert to an argparse.Namespace object
    """
    from argparse import Namespace

    _namespace = Namespace()
    for key, item in dictionary.items():
        setattr(_namespace, key, item)
    if add_defaults is not None:
        _opts = add_defaults.parse_args(args="")
        for key in vars(_opts):
            if key not in dictionary.keys():
                setattr(_namespace, key, add_defaults.get_default(key))
    return _namespace
