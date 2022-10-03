#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.cli.parser import ArgumentParser
from pesummary.utils import history_dictionary
from pesummary.utils.utils import gw_results_file

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to combine multiple result files into a
single PESummary metafile"""


def main(args=None):
    """Top level interface for `summarycombine`
    """
    _parser = ArgumentParser()
    _parser.add_all_groups_to_parser()
    opts, unknown = _parser.parse_known_args(args=args)
    if opts.gw or gw_results_file(opts):
        from pesummary.gw.cli.inputs import MetaFileInput
        from pesummary.gw.file.meta_file import GWMetaFile
        input_cls = MetaFileInput
        meta_file_cls = GWMetaFile
    else:
        from pesummary.core.cli.inputs import MetaFileInput
        from pesummary.core.file.meta_file import MetaFile
        input_cls = MetaFileInput
        meta_file_cls = MetaFile
    args = input_cls(opts)
    _history = history_dictionary(
        program=_parser.prog, creator=args.user,
        command_line=_parser.command_line
    )
    meta_file_cls(args, history=_history)


if __name__ == "__main__":
    main()
