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

from pesummary.core.command_line import command_line
from pesummary.gw.command_line import (
    insert_gwspecific_option_group, add_dynamic_PSD_to_namespace,
    add_dynamic_calibration_to_namespace
)
from pesummary.utils import functions


__doc__ = """This executable is used to combine multiple result files into a
single PESummary metafile"""


def main(args=None):
    """Top level interface for `summarycombine`
    """
    parser = command_line()
    insert_gwspecific_option_group(parser)
    opts, unknown = parser.parse_known_args(args=args)
    add_dynamic_PSD_to_namespace(opts)
    add_dynamic_calibration_to_namespace(opts)
    func = functions(opts)
    args = func["input"](opts)
    func["MetaFile"](args)


if __name__ == "__main__":
    main()
