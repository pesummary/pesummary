#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import argparse

from pesummary.core.command_line import CheckFilesExistAction
from pesummary.core.parser import parser
from pesummary.utils.utils import logger
from pesummary.io import read, available_formats

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to extract a set of posterior samples
from a file containing more than one set of analyses, for instance a PESummary
metafile"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", dest="label", default=None, required=True,
        help="Analysis that you wish to extract from the file"
    )
    parser.add_argument(
        "-s", "--samples", dest="samples", default=None, required=True,
        action=CheckFilesExistAction, help=(
            "Path to posterior samples file containing more than one analysis"
        )
    )
    parser.add_argument(
        "--file_format", dest="file_format", type=str, default="dat",
        help="Format of output file", choices=available_formats()[1]
    )
    parser.add_argument(
        "--filename", dest="filename", type=str, default=None,
        help="Name of the output file"
    )
    parser.add_argument(
        "--outdir", dest="outdir", type=str, default="./",
        help="Directory to save the file"
    )
    return parser


def main(args=None):
    """Top level interface for `summaryextract`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    logger.info("Loading file: '{}'".format(opts.samples))
    f = read(
        opts.samples, disable_prior=True, disable_injection_conversion=True
    )
    posterior_samples = f.samples_dict
    logger.info("Writing analysis: '{}' to file".format(opts.label))
    posterior_samples.write(
        file_format=opts.file_format, labels=[opts.label], outdir=opts.outdir,
        filename=opts.filename
    )


if __name__ == "__main__":
    main()
