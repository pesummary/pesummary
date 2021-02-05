#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import argparse

from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.core.command_line import CheckFilesExistAction
from pesummary.core.parser import parser
from pesummary.io import write

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to combine posterior samples. This is
different from 'summarycombine' as 'summarycombine' combines N files into a single
metafile containing N analyses while 'summarycombine_posteriors' combines N
posterior samples and creates a single file containing a single analysis"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", dest="labels", nargs='+', default=None, required=True,
        help=(
            "Analyses you wish to combine. If only a file containing more than "
            "one analysis is provided, please pass the labels in that file that "
            "you wish to combine. If multiple single analysis files are "
            "provided, please pass unique labels to distinguish each analysis. "
            "If a file containing more than one analysis is provided alongside "
            "a single analysis file, or multiple files containing more than one "
            "analysis each, only a single analysis can be combined from each "
            "file"
        )
    )
    parser.add_argument(
        "-s", "--samples", dest="samples", default=None, nargs='+',
        required=True, action=CheckFilesExistAction, help=(
            "Path to posterior samples file(s). See documentation for allowed "
            "formats."
        )
    )
    parser.add_argument(
        "--weights", dest="weights", nargs="+", default=None, type=float,
        help="Weights to assign to each analysis. Must be same length as labels"
    )
    parser.add_argument(
        "--use_all", dest="use_all", action="store_true", default=False,
        help="Use all posterior samples (do not weight)"
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", action="store_true", default=False,
        help="Shuffle the combined samples"
    )
    parser.add_argument(
        "--file_format", dest="file_format", type=str, default="dat",
        help="Format of output file"
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
    """Top level interface for `summarycombine_posteriors`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    samples = {
        label: samples for label, samples in zip(opts.labels, opts.samples)
    }
    mydict = MultiAnalysisSamplesDict.from_files(
        samples, disable_prior=True, disable_injection_conversion=True
    )
    combined = mydict.combine(
        use_all=opts.use_all, weights=opts.weights, labels=opts.labels,
        shuffle=opts.shuffle, logger_level="info"
    )
    combined.write(
        file_format=opts.file_format, filename=opts.filename, outdir=opts.outdir
    )


if __name__ == "__main__":
    main()
