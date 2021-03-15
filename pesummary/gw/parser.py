# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.parser import parser as core_parser
from pesummary.gw.command_line import (
    insert_gwspecific_option_group, add_dynamic_PSD_to_namespace,
    add_dynamic_calibration_to_namespace, add_dynamic_tgr_kwargs_to_namespace
)

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class parser(core_parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """
    def __init__(self, existing_parser=None):
        super(parser, self).__init__(existing_parser=existing_parser)
        if existing_parser is None:
            insert_gwspecific_option_group(self._parser)

    @property
    def dynamic_argparse(self):
        return [
            add_dynamic_PSD_to_namespace,
            add_dynamic_calibration_to_namespace
        ]


class TGRparser(parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """

    def __init__(self, existing_parser=None):
        super(parser, self).__init__(existing_parser=existing_parser)

    @property
    def dynamic_argparse(self):
        return [add_dynamic_tgr_kwargs_to_namespace]
