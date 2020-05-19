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

from pesummary.core.parser import parser as core_parser
from pesummary.gw.command_line import (
    insert_gwspecific_option_group, add_dynamic_PSD_to_namespace,
    add_dynamic_calibration_to_namespace
)


class parser(core_parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """
    def __init__(self):
        super(parser, self).__init__()
        insert_gwspecific_option_group(self._parser)

    @property
    def dynamic_argparse(self):
        return [
            add_dynamic_PSD_to_namespace,
            add_dynamic_calibration_to_namespace
        ]
