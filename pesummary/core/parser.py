# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.command_line import command_line
from pesummary.utils.utils import logger, command_line_arguments

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class parser(object):
    """Class to handle parsing command line arguments

    Parameters
    ----------
    existing_parser: argparse.ArgumentParser
        existing argument parser to use

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """
    def __init__(self, existing_parser=None):
        if existing_parser is None:
            self._parser = command_line()
        else:
            self._parser = existing_parser

    @property
    def dynamic_argparse(self):
        return []

    @property
    def prog(self):
        return self._parser.prog

    @property
    def command_line(self):
        return self.prog + " " + " ".join(command_line_arguments())

    @staticmethod
    def intersection(a, b):
        """Return the common entries between two lists
        """
        return list(set(a).intersection(set(b)))

    def parse_known_args(
        self, args=None, logger_level={"known": "info", "unknown": "warn"}
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
        opts, unknown = self._parser.parse_known_args(args=args)
        for dynamic in self.dynamic_argparse:
            _, _unknown = dynamic(opts, command_line=args)
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
