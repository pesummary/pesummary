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

import argparse


def command_line():
    """
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", default=None, required=True,
        help="The GraceDB id of the event you are interested in"
    )
    parser.add_argument(
        "--info", default=None, nargs='+',
        help="Specific information you wish to retrieve"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output json file you wish to save the data to"
    )
    return parser


def main(args=None):
    """
    """
    from pesummary.gw.parser import parser
    from pesummary.gw.gracedb import get_gracedb_data

    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    data = get_gracedb_data(opts.id, info=opts.info)
    if opts.output is not None:
        import json

        with open(opts.output, "w") as f:
            json.dump(data, f)
        return
    print(data)


if __name__ == "__main__":
    main()
