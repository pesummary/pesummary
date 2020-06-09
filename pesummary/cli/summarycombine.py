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

from pesummary.gw.parser import parser
from pesummary.utils import functions, history_dictionary


__doc__ = """This executable is used to combine multiple result files into a
single PESummary metafile"""


def main(args=None):
    """Top level interface for `summarycombine`
    """
    _parser = parser()
    opts, unknown = _parser.parse_known_args(args=args)
    func = functions(opts)
    args = func["input"](opts)
    _history = history_dictionary(
        program=_parser.prog, creator=args.user,
        command_line=_parser.command_line
    )
    func["MetaFile"](args, history=_history)


if __name__ == "__main__":
    main()
