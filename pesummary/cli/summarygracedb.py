#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import argparse

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
