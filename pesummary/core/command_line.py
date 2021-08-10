# Licensed under an MIT style license -- see LICENSE.md

import copy
import os

import argparse
import configparser
import numpy as np
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class CheckFilesExistAction(argparse.Action):
    """Class to extend the argparse.Action to identify if files exist
    """
    def __init__(self, *args, **kwargs):
        super(CheckFilesExistAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.check_input(values)

    def check_input(self, value):
        """Check that all files provided exist

        Parameters
        ----------
        value: str, list, dict
            data structure that you wish to check
        """
        if isinstance(value, list):
            for ff in value:
                _ = self.check_input(ff)
        elif isinstance(value, str):
            _ = self._is_file(value)
        elif isinstance(value, dict):
            for _value in value.values():
                _ = self.check_input(_value)
        else:
            _ = self._is_file(value)

    def _is_file(self, ff):
        """Return True if the file exists else raise a FileNotFoundError
        exception

        Parameters
        ----------
        ff: str
            path to file you wish to check
        """
        cond = any(_str in ff for _str in ["*", "@", "https://"])
        cond2 = isinstance(ff, str) and ff.lower() == "none"
        if not os.path.isfile(ff) and not cond and not cond2:
            raise FileNotFoundError(
                "The file '{}' provided for '{}' does not exist".format(
                    ff, self.dest
                )
            )
        return True


class DeprecatedStoreTrueAction(object):
    """Class to handle deprecated argparse options
    """
    class _DeprecatedStoreTrueAction(argparse._StoreTrueAction):
        def __init__(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            import warnings
            msg = (
                "The option '{}' is out-of-date and may not be supported in future "
                "releases.".format(self.option_strings[0])
            )
            if _new_option is not None:
                msg += " Please use '{}'".format(_new_option)
            warnings.warn(msg)
            return super(self.__class__, self).__call__(*args, **kwargs)

    def __new__(cls, *args, new_option=None, **kwargs):
        global _new_option
        _new_option = new_option
        return cls._DeprecatedStoreTrueAction


class ConfigAction(argparse.Action):
    """Class to extend the argparse.Action to handle dictionaries as input
    """
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(ConfigAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=nargs,
            const=const, default=default, type=str, choices=choices,
            required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        items = {}
        config = configparser.ConfigParser()
        config.optionxform = str
        try:
            config.read(values)
            for key, value in config.items("pesummary"):
                if value == "True" or value == "true":
                    items[key] = True
                else:
                    if ":" in value:
                        items[key] = self.dict_from_str(value)
                    elif "," in value:
                        items[key] = self.list_from_str(value)
                    else:
                        items[key] = value
        except Exception:
            pass
        for i in vars(namespace).keys():
            if i in items.keys():
                setattr(namespace, i, items[i])

    @staticmethod
    def dict_from_str(string, delimiter=":"):
        """Reformat the string into a dictionary

        Parameters
        ----------
        string: str
            string that you would like reformatted into a dictionary
        """
        mydict = {}
        if "{" in string:
            string = string.replace("{", "")
        if "}" in string:
            string = string.replace("}", "")

        if " " in string and "," not in string:
            string = string.split(" ")
        elif "," in string and ", " not in string:
            string = string.split(",")
        elif ", " in string:
            string = string.split(", ")

        for i in string:
            value = i.split(delimiter)
            if " " in value[0]:
                value[0] = value[0].replace(" ", "")
            if " " in value[1]:
                value[1] = value[1].replace(" ", "")
            if value[0] in mydict.keys():
                mydict[value[0]].append(value[1])
            else:
                mydict[value[0]] = [value[1]]
        return mydict

    @staticmethod
    def list_from_str(string):
        """Reformat the string into a list

        Parameters
        ----------
        string: str
            string that you would like reformatted into a list
        """
        list = []
        if "[" in string:
            string.replace("[", "")
        if "]" in string:
            string.replace("]", "")
        if ", " in string:
            list = string.split(", ")
        elif " " in string:
            list = string.split(" ")
        elif "," in string:
            list = string.split(",")
        return list


class DictionaryAction(argparse.Action):
    """Class to extend the argparse.Action to handle dictionaries as input
    """
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(DictionaryAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=nargs,
            const=const, default=default, type=str, choices=choices,
            required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        bool = [True if ':' in value else False for value in values]
        if all(i is True for i in bool):
            setattr(namespace, self.dest, {})
        elif all(i is False for i in bool):
            setattr(namespace, self.dest, [])
        else:
            raise ValueError("Did not understand input")

        items = getattr(namespace, self.dest)
        items = copy.copy(items)
        for value in values:
            value = value.split(':')
            if len(value) > 2:
                value = [":".join(value[:-1]), value[-1]]
            if len(value) == 2:
                if value[0] in items.keys():
                    if not isinstance(items[value[0]], list):
                        items[value[0]] = [items[value[0]]]
                    items[value[0]].append(value[1])
                else:
                    items[value[0]] = value[1]
            elif len(value) == 1:
                items.append(value[0])
            else:
                raise ValueError("Did not understand input")
        setattr(namespace, self.dest, items)


class DelimiterSplitAction(argparse.Action):
    """Class to extend the argparse.Action to handle inputs which need to be split with
    with a provided delimiter
    """
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(DelimiterSplitAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=nargs,
            const=const, default=default, type=str, choices=choices,
            required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        import sys

        args = np.array(sys.argv[1:])
        cond1 = "--delimiter" in args
        cond2 = False
        if cond1:
            cond2 = (
                float(np.argwhere(args == "--delimiter"))
                > float(np.argwhere(args == self.option_strings[0]))
            )
        if cond1 and cond2:
            raise ValueError(
                "Please provide the '--delimiter' command line argument "
                "before the '{}' argument".format(self.option_strings[0])
            )
        delimiter = namespace.delimiter
        items = {}
        for value in values:
            value = value.split(delimiter)
            if len(value) > 2:
                raise ValueError(
                    "'{}' appears multiple times. Please choose a different "
                    "delimiter".format(delimiter)
                )
            if value[0] in items.keys() and not isinstance(items[value[0]], list):
                items[value[0]] = [items[value[0]]]
            if value[0] in items.keys():
                items[value[0]].append(value[1])
            elif len(value) == 1 and len(values) == 1:
                items = [value[0]]
            else:
                items[value[0]] = value[1]
        setattr(namespace, self.dest, items)


def _core_command_line_arguments(parser):
    """Add core command line options to the Argument Parser

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    core_group = parser.add_argument_group(
        "Core command line options\n"
        "-------------------------"
    )
    core_group.add_argument(
        "pesummary", nargs='?', action=ConfigAction,
        help="configuration file containing the command line arguments"
    )
    core_group.add_argument(
        "-w", "--webdir", dest="webdir", default=None, metavar="DIR",
        help="make page and plots in DIR"
    )
    core_group.add_argument(
        "-b", "--baseurl", dest="baseurl", metavar="DIR", default=None,
        help="make the page at this url"
    )
    core_group.add_argument(
        "--labels", dest="labels", help="labels used to distinguish runs",
        nargs='+', default=None
    )
    core_group.add_argument(
        "-s", "--samples", dest="samples", default=None, nargs='+',
        action=CheckFilesExistAction, help=(
            "Path to posterior samples file(s). See documentation for allowed "
            "formats. If path is on a remote server, add username and "
            "servername in the form {username}@{servername}:{path}. If path "
            "is on a public webpage, ensure the path starts with https://. "
            "You may also pass a string such as posterior_samples*.dat and "
            "all matching files will be used"
        )
    )
    core_group.add_argument(
        "-c", "--config", dest="config", nargs='+', default=None,
        action=CheckFilesExistAction, help=(
            "configuration file associcated with each samples file."
        )
    )
    core_group.add_argument(
        "--email", action="store", default=None,
        help=(
            "send an e-mail to the given address with a link to the finished "
            "page."
        )
    )
    core_group.add_argument(
        "-i", "--inj_file", dest="inj_file", nargs='+', default=None,
        action=CheckFilesExistAction, help="path to injetcion file"
    )
    core_group.add_argument(
        "--user", dest="user", help=argparse.SUPPRESS, default=conf.user
    )
    core_group.add_argument(
        "--testing", action="store_true", help=argparse.SUPPRESS, default=False
    )
    core_group.add_argument(
        "--add_to_existing", action="store_true", default=False,
        help="add new results to an existing html page"
    )
    core_group.add_argument(
        "-e", "--existing_webdir", dest="existing", default=None,
        help="web directory of existing output"
    )
    core_group.add_argument(
        "--seed", dest="seed", default=123456789, type=int,
        help="Random seed to used through the analysis. Default 123456789"
    )
    core_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="print useful information for debugging purposes"
    )
    return core_group


def _samples_command_line_arguments(parser):
    """Add sample specific command line options to the Argument Parser

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    sample_group = parser.add_argument_group(
        "Options specific to the samples you wish to input\n"
        "-------------------------------------------------"
    )
    sample_group.add_argument(
        "--ignore_parameters", dest="ignore_parameters", nargs='+', default=None,
        help=(
            "Parameters that you wish to not include in the summarypages. You "
            "may list them or use wildcards ('recalib*')"
        )
    )
    sample_group.add_argument(
        "--nsamples", dest="nsamples", default=None, help=(
            "The number of samples to use and store in the PESummary metafile. "
            "These samples will be randomly drawn from the posterior "
            "distributions"
        )
    )
    sample_group.add_argument(
        "--burnin", dest="burnin", default=None, help=(
            "Number of samples to remove as burnin"
        )
    )
    sample_group.add_argument(
        "--burnin_method", dest="burnin_method", default=None, help=(
            "The algorithm to use to remove mcmc samples as burnin. This is "
            "only used when `--mcmc_samples` also used"
        )
    )
    sample_group.add_argument(
        "--regenerate", dest="regenerate", default=None, nargs="+", help=(
            "List of posterior distributions that you wish to regenerate if "
            "possible"
        )
    )
    sample_group.add_argument(
        "--mcmc_samples", action="store_true", default=False, help=(
            "treat the passed samples as seperate mcmc chains for the same "
            "analysis"
        )
    )
    sample_group.add_argument(
        "--path_to_samples", default=None, nargs="+", help=(
            "Path to the posterior samples stored in the result file. If "
            "None, pesummary will search for a 'posterior' or "
            "'posterior_samples' group. If more than one result file is "
            "passed, and only the third file requires a path_to_samples "
            "provide --path_to_samples None None path/to/samples"
        )
    )
    sample_group.add_argument(
        "--pe_algorithm", default=None, nargs="+", help=(
            "Name of the algorithm used to generate the result file"
        )
    )
    sample_group.add_argument(
        "--reweight_samples", default=False, help=(
            "Method to use when reweighting posterior and/or prior samples. "
            "Default do not reweight samples."
        )
    )
    sample_group.add_argument(
        "--descriptions", default={}, action=DictionaryAction, nargs="+",
        help=(
            "JSON file containing a set of descriptions for each analysis or "
            "dictionary giving descriptions for each analysis directly from "
            "the command line (e.g. `--descriptions label1:'description'`). "
            "These descriptions are then saved in the output."
        )
    )
    return sample_group


def _plotting_command_line_arguments(parser):
    """Add specific command line options for plotting options

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    plot_group = parser.add_argument_group(
        "Options specific to the plots you wish to make\n"
        "----------------------------------------------"
    )
    plot_group.add_argument(
        "--custom_plotting", dest="custom_plotting", default=None,
        help="Python file containing functions for custom plotting"
    )
    plot_group.add_argument(
        "--publication", action="store_true", default=None, help=(
            "generate production quality plots"
        )
    )
    plot_group.add_argument(
        "--publication_kwargs", action=DictionaryAction, nargs="+", default={},
        help="Optional kwargs for publication plots",
    )
    plot_group.add_argument(
        "--kde_plot", action="store_true", default=False, help=(
            "plot a kde rather than a histogram"
        )
    )
    plot_group.add_argument(
        "--colors", dest="colors", nargs='+', default=None,
        help="Colors you wish to use to distinguish result files",
    )
    plot_group.add_argument(
        "--palette", dest="palette", default="colorblind",
        help="Color palette to use to distinguish result files",
    )
    plot_group.add_argument(
        "--linestyles", dest="linestyles", nargs='+', default=None,
        help="Linestyles you wish to use to distinguish result files"
    )
    plot_group.add_argument(
        "--include_prior", action="store_true", default=False,
        help="Plot the prior on the same plot as the posterior",
    )
    plot_group.add_argument(
        "--style_file", dest="style_file", default=None,
        action=CheckFilesExistAction, help=(
            "Style file you wish to use when generating plots"
        )
    )
    plot_group.add_argument(
        "--add_to_corner", dest="add_to_corner", default=None,
        nargs="+", help="Parameters you wish to include in the corner plot"
    )
    plot_group.add_argument(
        "--add_existing_plot", dest="existing_plot", nargs="+", default=None,
        action=DictionaryAction, help=(
            "Path(s) to existing plots that you wish to add to the "
            "summarypages. Should be of the form {label}:{path}"
        )
    )
    return plot_group


def _webpage_command_line_arguments(parser):
    """Add specific command line options for the webpage generation

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    webpage_group = parser.add_argument_group(
        "Options specific to the webpages you wish to produce\n"
        "----------------------------------------------------"
    )
    webpage_group.add_argument(
        "--dump", action="store_true", default=False,
        help="dump all information onto a single html page",
    )
    webpage_group.add_argument(
        "--notes", dest="notes", default=None,
        help="Single file containing notes that you wish to put on summarypages"
    )
    return webpage_group


def _prior_command_line_arguments(parser):
    """Add specific command line options for prior files

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    prior_group = parser.add_argument_group(
        "Options specific for passing prior files\n"
        "----------------------------------------"
    )
    prior_group.add_argument(
        "--prior_file", dest="prior_file", nargs='+', default=None,
        action=CheckFilesExistAction, help=(
            "File containing for the prior samples for a given label"
        )
    )
    prior_group.add_argument(
        "--nsamples_for_prior", dest="nsamples_for_prior", default=5000,
        type=int, help=(
            "The number of prior samples to extract from a bilby prior file "
            "or a bilby result file"
        )
    )
    prior_group.add_argument(
        "--disable_prior_sampling", action="store_true",
        help="Skip generating prior samples using bilby", default=False
    )
    return prior_group


def _performance_command_line_options(parser):
    """Add command line options which enhance the performance of the code

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    performance_group = parser.add_argument_group(
        "Options specific for enhancing the performance of the code\n"
        "----------------------------------------------------------"
    )
    performance_group.add_argument(
        "--disable_comparison", action="store_true", default=False,
        help=(
            "Whether to make a comparison webpage if multple results are "
            "present"
        )
    )
    performance_group.add_argument(
        "--disable_interactive", action="store_true", default=False,
        help="Whether to make interactive plots or not"
    )
    performance_group.add_argument(
        "--disable_corner", action="store_true", default=False,
        help="Whether to make a corner plot or not"
    )
    performance_group.add_argument(
        "--disable_expert", action="store_true", default=False,
        help="Whether to generate extra diagnostic plots or not"
    )
    performance_group.add_argument(
        "--multi_process", dest="multi_process", default=1,
        help="The number of cores to use when generating plots"
    )
    performance_group.add_argument(
        "--file_format", dest="file_format", nargs='+', default=None,
        help="The file format of each result file."
    )
    performance_group.add_argument(
        "--restart_from_checkpoint", action="store_true", default=False,
        help=(
            "Restart from checkpoint if a checkpoint file can be found in "
            "webdir"
        )
    )
    return performance_group


def _pesummary_metafile_command_line_options(parser):
    """Add command line options which are specific for reading and
    manipulating pesummary metafiles

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    pesummary_group = parser.add_argument_group(
        "Options specific for reading and manipulating pesummary metafiles\n"
        "-----------------------------------------------------------------"
    )
    pesummary_group.add_argument(
        "--compare_results", dest="compare_results", nargs='+', default=None,
        help=(
            "labels for events stored in the posterior_samples.json that you "
            "wish to compare"
        )
    )
    pesummary_group.add_argument(
        "--save_to_json", action="store_true", default=False,
        help="save the meta file in json format"
    )
    pesummary_group.add_argument(
        "--posterior_samples_filename", dest="filename", default=None,
        help="name of the posterior samples metafile that is produced"
    )
    pesummary_group.add_argument(
        "--external_hdf5_links", action="store_true", default=False,
        help=(
            "save each analysis as a seperate hdf5 file and connect them to "
            "the meta file through external links"
        )
    )
    pesummary_group.add_argument(
        "--hdf5_compression", dest="hdf5_compression", default=None, type=int,
        help=(
            "compress each dataset with a particular compression filter. "
            "Filter must be integer between 0 and 9. Only applies to meta "
            "files stored in hdf5 format. Default, no compression applied"
        )
    )
    pesummary_group.add_argument(
        "--disable_injection", action="store_true", default=False,
        help=(
            "whether or not to extract stored injection data from the meta file"
        )
    )
    return pesummary_group


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    core_group = _core_command_line_arguments(parser)
    sample_group = _samples_command_line_arguments(parser)
    plot_group = _plotting_command_line_arguments(parser)
    webpage_group = _webpage_command_line_arguments(parser)
    prior_group = _prior_command_line_arguments(parser)
    performance_group = _performance_command_line_options(parser)
    pesummary_group = _pesummary_metafile_command_line_options(parser)
    return parser
