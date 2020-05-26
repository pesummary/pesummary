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

import copy

import argparse
import configparser
import numpy as np
from pesummary import conf


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
    def dict_from_str(string):
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
            value = i.split(":")
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
        help="Posterior samples hdf5 file"
    )
    core_group.add_argument(
        "-c", "--config", dest="config", nargs='+', default=None,
        help="configuration file associcated with each samples file."
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
        help="path to injetcion file"
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
        "--seed", dest="seed", default=None,
        help="Random seed to used through the analysis"
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
        help="Style file you wish to use when generating plots"
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
        help="File containing for the prior samples for a given label"
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
        "--multi_process", dest="multi_process", default=1,
        help="The number of cores to use when generating plots"
    )
    performance_group.add_argument(
        "--file_format", dest="file_format", nargs='+', default=None,
        help="The file format of each result file."
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
