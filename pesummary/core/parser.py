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

from pesummary.core.command_line import command_line
from pesummary.utils.utils import logger


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
            _, _unknown = dynamic(opts)
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
