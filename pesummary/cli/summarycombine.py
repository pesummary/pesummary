#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.parser import parser
from pesummary.utils import functions, history_dictionary

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
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
