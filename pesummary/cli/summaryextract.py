#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.cli.actions import CheckFilesExistAction
from pesummary.core.cli.parser import ArgumentParser as _ArgumentParser
from pesummary.utils.utils import logger
from pesummary.io import read, available_formats

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to extract a set of posterior samples
from a file containing more than one set of analyses, for instance a PESummary
metafile"""


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--label": {
                    "required": True,
                    "help": "Analysis that you wish to extract from the file"
                },
                "--samples": {
                    "required": True,
                    "short": "-s",
                    "action": CheckFilesExistAction,
                    "help": (
                        "Path to posterior samples file containing more than "
                        "one analysis"
                    )
                },
                "--file_format": {
                    "type": str,
                    "default": "dat",
                    "help": "Format of output file",
                    "choices": available_formats()[1]
                },
                "--filename": {
                    "type": str,
                    "help": "Name of the output file"
                },
                "--outdir": {
                    "type": str,
                    "default": "./",
                    "help": "Directory to save the file",
                },
            }
        )
        return options


def main(args=None):
    """Top level interface for `summaryextract`
    """
    _parser = ArgumentParser(description=__doc__)
    _parser.add_known_options_to_parser(
        ["--label", "--samples", "--file_format", "--filename", "--outdir"]
    )
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
