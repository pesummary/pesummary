#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import argparse

from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.core.command_line import CheckFilesExistAction
from pesummary.core.parser import parser
from pesummary.core.inputs import _Input
from pesummary.io import read, write

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
            "formats. If path is on a remote server, add username and "
            "servername in the form {username}@{servername}:{path}. If path "
            "is on a public webpage, ensure the path starts with https://. "
            "You may also pass a string such as posterior_samples*.dat and "
            "all matching files will be used"
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
    parser.add_argument(
        "--seed", dest="seed", default=123456789, type=int,
        help="Random seed to used through the analysis. Default 123456789"
    )
    parser.add_argument(
        "--add_to_existing", action="store_true", default=False, help=(
            "Add the combined samples to an existing metafile. Only used when "
            "a PESummary metafile is provided via the `--samples` option. If "
            "this option is provided, the `--file_format` and `--filename` "
            "options are ignored"
        )
    )
    return parser


class Input(_Input):
    """Class to handle the core command line arguments

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options

    Attributes
    ----------
    result_files: list
        list of result files passed
    labels: list
        list of labels used to distinguish the result files
    """
    def __init__(self, opts):
        self.opts = opts
        self.seed = self.opts.seed
        self.result_files = self.opts.samples
        self.mcmc_samples = False
        self.add_to_existing = False
        cond = np.sum([self.is_pesummary_metafile(f) for f in self.result_files])
        if cond > 1:
            raise ValueError(
                "Can only combine analyses from a single PESummary metafile"
            )
        elif cond == 1 and len(self.result_files) > 1:
            raise ValueError(
                "Can only combine analyses from a single PESummary metafile "
                "or multiple non-PESummary metafiles"
            )
        self.pesummary = False
        if self.is_pesummary_metafile(self.result_files[0]):
            self.pesummary = True
            self._labels = self.opts.labels
        else:
            self.labels = self.opts.labels


def main(args=None):
    """Top level interface for `summarycombine_posteriors`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    args = Input(opts)
    if not args.pesummary:
        samples = {
            label: samples for label, samples in
            zip(args.labels, args.result_files)
        }
        mydict = MultiAnalysisSamplesDict.from_files(
            samples, disable_prior=True, disable_injection_conversion=True
        )
    else:
        mydict = read(
            args.result_files[0], disable_prior=True,
            disable_injection_conversion=True
        ).samples_dict
    combined = mydict.combine(
        use_all=opts.use_all, weights=opts.weights, labels=args.labels,
        shuffle=opts.shuffle, logger_level="info"
    )
    if opts.add_to_existing and args.pesummary:
        from .summarymodify import _Input as _ModifyInput
        from pesummary.gw.file.meta_file import _GWMetaFile
        from pathlib import Path
        import os

        class PESummaryInput(_ModifyInput):
            def __init__(self, samples):
                self.samples = samples
                self.data = None

        _args = PESummaryInput(args.result_files)
        data = _args.data
        data["{}_combined".format("_".join(args.labels))] = {
            "posterior_samples": combined.to_structured_array(),
            "meta_data": {
                "sampler": {"nsamples": combined.number_of_samples},
                "meta_data": {"combined": ", ".join(args.labels)}
            }
        }
        input_file = Path(args.result_files[0])
        _metafile = os.path.join(
            opts.outdir, input_file.stem + "_combined" + input_file.suffix
        )
        if input_file.suffix in [".h5", ".hdf5"]:
            _GWMetaFile.save_to_hdf5(
                data, list(data.keys()), None, _metafile, no_convert=True
            )
        else:
            _GWMetaFile.save_to_json(data, _metafile)
        return
    elif opts.add_to_existing:
        from pesummary.utils.utils import logger
        logger.warn(
            "Can only use the `--add_to_existing` option when a PESummary "
            "metafile is provided via the `--samples` option. Writing "
            "combined samples to a `dat` file"
        )
    combined.write(
        file_format=opts.file_format, filename=opts.filename, outdir=opts.outdir
    )


if __name__ == "__main__":
    main()
