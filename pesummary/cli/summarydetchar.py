#! /usr/bin/env python

# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os
import pesummary
from pesummary.gw.file.read import read
from pesummary.gw.plots import detchar
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import make_dir, logger
from pesummary.gw.command_line import DictionaryAction
import matplotlib.pyplot as plt
import argparse


__doc__ = """This executable is used to generate plots associated with the
detectors"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default=None)
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Posterior samples hdf5 file", nargs='+',
                        default=None)
    parser.add_argument("--gwdata", dest="gwdata",
                        help="channels and paths to strain cache files",
                        action=DictionaryAction, metavar="CHANNEL:CACHEFILE",
                        nargs="+", default=None)
    parser.add_argument("--plot", dest="plot",
                        help=("name of the publication plot you wish to "
                              "produce"), default="2d_contour",
                        choices=["spectrogram", "omegascan"])
    parser.add_argument("--gps", dest="gps", default=None,
                        help="GPS time to centre the omegascan around")
    parser.add_argument("--vmin", dest="vmin", default=0,
                        help="minimum for the omegascan colormap")
    parser.add_argument("--vmax", dest="vmax", default=0,
                        help="maximum for the omegascan colormap")
    parser.add_argument("--window", dest="window", default=4,
                        help="window around gps time to generate omegascan for")
    return parser


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
    from pesummary.gw.file.formats.base_read import GWRead

    for i in dictionary.keys():
        if not os.path.isfile(dictionary[i]):
            raise InputError(
                "The file {} does not exist. Please check the path to "
                "your strain file".format(dictionary[i])
            )
    timeseries = GWRead.load_strain_data(dictionary)
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
        plt.close()


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
        plt.close()


def main():
    """Top level interface for `summarydetchar`
    """
    parser = command_line()
    opts = parser.parse_args()
    make_dir(opts.webdir)
    func_map = {"spectrogram": make_spectrogram_plot,
                "omegascan": make_omegascan_plot}
    func_map[opts.plot](opts)
