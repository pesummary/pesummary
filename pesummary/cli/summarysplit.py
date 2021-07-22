#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import argparse
import numpy as np
import multiprocessing

from pesummary.core.command_line import CheckFilesExistAction
from pesummary.core.parser import parser
from pesummary.utils.utils import iterator, logger, make_dir
from pesummary.io import read, write, available_formats

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to split the posterior samples contained
within a file into N separate files. If the input file has more than one
analysis, the posterior samples for each analysis is split into N separate
files. This is useful for submitting thousands of summarypages to condor"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--samples", dest="samples", default=None, required=True,
        action=CheckFilesExistAction, help=(
            "Path to the posterior samples file you wish to split"
        )
    )
    parser.add_argument(
        "--file_format", dest="file_format", type=str, default="dat",
        help="Format of each output file", choices=available_formats()[1]
    )
    parser.add_argument(
        "--outdir", dest="outdir", type=str, default="./",
        help="Directory to save each file"
    )
    parser.add_argument(
        "-N", "--N_files", dest="N_files", type=int, default=0,
        help=(
            "Number of files to split the posterior samples into. Default 0 "
            "meaning N_files=n_samples where n_samples is the number "
            "of posterior samples"
        )
    )
    parser.add_argument(
        "--multi_process", dest="multi_process", type=int, default=1,
        help="The number of cores to use when writing split posterior samples"
    )
    return parser


def _write_posterior_samples(
    posterior_samples, split_idxs, file_format, outdir, filename
):
    """Split a set of posterior samples and write them to file

    Parameters
    ----------
    posterior_samples: pesummary.utils.samples_dict.SamplesDict
        set of posterior samples you wish to split and write to file
    split_idxs: np.ndarray
        2D array giving indices for each split, e.g. [[1,2,3], [4,5,6], [7,8]]
    file_format: str
        format to write the posterior samples
    outdir: str
        directory to store each file
    filename: str
        filename to use for each file
    """
    _parameters = posterior_samples.parameters
    _samples = posterior_samples.samples.T[split_idxs[0]:split_idxs[-1] + 1]
    write(
        _parameters, _samples, file_format=file_format, outdir=outdir,
        filename=filename
    )
    return


def _wrapper_for_write_posterior_samples(args):
    """Wrapper function for _write_posterior_samples for a pool of workers

    Parameters
    ----------
    args: tuple
        All args passed to _write_posterior_samples
    """
    return _write_posterior_samples(*args)


def _split_posterior_samples(
    posterior_samples, N_files, file_format="dat", outdir="./",
    filename=None, multi_process=1
):
    """Split a set of posterior samples and write each split to file

    Parameters
    ----------
    posterior_samples: pesummary.utils.samples_dict.SamplesDict
        set of posterior samples you wish to split and write to file
    N_files: int
        number of times to split the posterior samples
    file_format: str, optional
        file format to write split posterior samples. Default 'dat'
    outdir: str, optional
        directory to write split posterior samples. Default './'
    filename: str, optional
        filename to use when writing split posterior samples. Should be of
        the form 'filename_{}.file_format'; '{}' will be replaced by the
        split num. Default 'None' which leads to
        'split_posterior_samples_{}.dat'
    multi_process: int, optional
        number of cpus to use when writing the split posterior samples.
        Default 1
    """
    n_samples = posterior_samples.number_of_samples
    if N_files > n_samples:
        logger.warn(
            "Number of requested files '{}' greater than number of samples "
            "'{}'. Reducing the number of files to '{}'".format(
                N_files, n_samples, n_samples
            )
        )
        N_files = n_samples
    elif not N_files:
        N_files = n_samples
    if filename is None:
        filename = "split_posterior_samples_{}.%s" % (file_format)
    make_dir(outdir)

    logger.info(
        "Splitting posterior samples into {} files".format(N_files)
    )
    idxs = np.arange(n_samples)
    split_idxs = np.array_split(idxs, N_files)
    filenames = [
        filename.format(num) for num in np.arange(len(split_idxs))
    ]
    args = np.array(
        [
            [posterior_samples] * len(split_idxs), split_idxs,
            [file_format] * len(split_idxs), [outdir] * len(split_idxs),
            filenames
        ], dtype=object
    ).T
    with multiprocessing.Pool(multi_process) as pool:
        _ = np.array(
            list(
                iterator(
                    pool.imap(_wrapper_for_write_posterior_samples, args),
                    tqdm=True, desc="Saving posterior samples to file",
                    logger=logger, total=len(split_idxs)
                )
            )
        )


def main(args=None):
    """Top level interface for `summarysplit`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    logger.info("Loading file: '{}'".format(opts.samples))
    f = read(
        opts.samples, disable_prior=True, disable_injection_conversion=True
    )
    posterior_samples = f.samples_dict
    if hasattr(f, "labels") and f.labels is not None and len(f.labels) > 1:
        for label in f.labels:
            _split_posterior_samples(
                posterior_samples[label], opts.N_files,
                outdir=os.path.join(opts.outdir, label),
                file_format=opts.file_format, multi_process=opts.multi_process
            )
    else:
        _split_posterior_samples(
            posterior_samples, opts.N_files, outdir=opts.outdir,
            file_format=opts.file_format, multi_process=opts.multi_process
        )


if __name__ == "__main__":
    main()
