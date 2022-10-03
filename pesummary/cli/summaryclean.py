#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.read import read as GWRead
from pesummary.core.cli.parser import ArgumentParser as _ArgumentParser

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable returns a cleaned data file"""


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--file_format": {
                    "default": "dat",
                    "help": "Save the cleaned data in this format",
                    "choices": [
                        "dat", "lalinference", "bilby", "lalinference_dat"
                    ],
                }
            }
        )
        return options


def clean_data_file(path):
    """Clean the data file and return a PESummary result file object

    Parameters
    ----------
    path: str
        path to the result file
    """
    f = GWRead(path)
    f.generate_all_posterior_samples()
    return f


def save(pesummary_object, file_format, webdir=None, label=None):
    """Save the pesummary_object to a given format

    Parameters
    ----------
    pesummary_object: pesummary.gw.file.formats
        pesummary results file object
    file_format: str
        the file format that you wish to save the file as
    webdir: str
        directory to save the cleaned data file
    """
    if file_format == "dat":
        pesummary_object.to_dat(outdir=webdir, label=label)
    elif file_format == "lalinference":
        pesummary_object.to_lalinference(outdir=webdir, label=label)
    elif file_format == "lalinference_dat":
        pesummary_object.to_lalinference(outdir=webdir, label=label, dat=True)
    elif file_format == "bilby":
        pesummary_object.to_bilby()


def main(args=None):
    """Top level interface for `summaryclean`
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_known_options_to_parser(
        ["--webdir", "--samples", "--labels", "--file_format"]
    )
    opts = parser.parse_args(args=args)
    if opts.labels:
        if len(opts.labels) != len(opts.samples):
            raise Exception("Please provide labels for all result files")
    for num, i in enumerate(opts.samples):
        f = clean_data_file(i)
        label = None
        if opts.labels:
            label = opts.labels[num]
        save(f, opts.file_format, webdir=opts.webdir, label=label)
