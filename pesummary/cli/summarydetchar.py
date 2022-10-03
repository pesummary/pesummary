#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
from pesummary.gw.file.read import read
from pesummary.gw.plots import detchar
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import make_dir
from pesummary.gw.cli.parser import ArgumentParser as _ArgumentParser

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to generate plots associated with the
detectors"""


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--plot": {
                    "choices": ["spectrogram", "omegascan"],
                    "default": "2d_contour",
                    "help": "name of the publication plot you wish to produce",
                },
                "--gps": {
                    "help": "GPS time to centre the omegascan around",
                },
                "--vmin": {
                    "default": 0,
                    "help": "minimum for the omegascan colormap"
                },
                "--vmax": {
                    "default": 25,
                    "help": "maximum for the omegascan colormap"
                },
                "--window": {
                    "default": 4,
                    "help": "window around gps time to generate omegascan for",
                }
            }
        )
        return options


def get_maxL_time(samples):
    """Return the maxL time stored in the samples

    Parameters
    ----------
    samples: str
        path to a samples file
    """
    f = read(samples)
    samples_dict = f.samples_dict
    return samples_dict["geocent_time"].maxL


def read_strain(dictionary):
    """Read the gwdata strain and return a gwpy.timeseries.TimeSeries object

    Parameters
    ----------
    dictionary: dict
        dictionary of channels and cache files
    """
    from pesummary.gw.file.strain import StrainDataDict

    for i in dictionary.keys():
        if not os.path.isfile(dictionary[i]):
            raise InputError(
                "The file {} does not exist. Please check the path to "
                "your strain file".format(dictionary[i])
            )
    timeseries = StrainDataDict.read(dictionary)
    return timeseries


def make_spectrogram_plot(opts):
    """Make a spectrogram plot
    """
    gwdata = read_strain(opts.gwdata)
    figs = detchar.spectrogram(gwdata)
    for det, fig in figs.items():
        fig.savefig(
            os.path.join(
                opts.webdir, "spectrogram_{}.png".format(det)
            )
        )
        fig.close()


def make_omegascan_plot(opts):
    """Make an omegascan plot. If gps is None, centre around maxL from samples
    """
    if opts.gps is None:
        opts.gps = get_maxL_time(opts.samples[0])
    gwdata = read_strain(opts.gwdata)
    figs = detchar.omegascan(
        gwdata, float(opts.gps), window=float(opts.window),
        vmin=float(opts.vmin), vmax=float(opts.vmax)
    )
    for det, fig in figs.items():
        fig.savefig(
            os.path.join(
                opts.webdir, "omegascan_{}.png".format(det)
            )
        )
        fig.close()


def main(args=None):
    """Top level interface for `summarydetchar`
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_known_options_to_parser(
        [
            "--webdir", "--samples", "--gwdata", "--plot", "--gps", "--vmin",
            "--vmax", "--window"
        ]
    )
    opts = parser.parse_args(args=args)
    make_dir(opts.webdir)
    func_map = {"spectrogram": make_spectrogram_plot,
                "omegascan": make_omegascan_plot}
    func_map[opts.plot](opts)
