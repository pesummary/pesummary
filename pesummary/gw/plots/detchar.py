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

from pesummary.utils.utils import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spectrogram(
    strain, vmin=1e-23, vmax=1e-19, cmap="viridis", ylim=[40, 2000], **kwargs
):
    """Generate a spectrogram from the timeseries

    Parameters
    ----------
    strain: dict
        dictionary of gw.py timeseries objects containing the strain data for
        each IFO
    vmin: float, optional
        minimum for the colormap
    vmax: float, optional
        maximum for the colormap
    cmap: str, optional
        cmap for the plot. See `matplotlib.pyplot.colormaps()` for options
    ylim: list, optional
        list to give the lower and upper bound of the plot
    """
    figs = {}
    for num, key in enumerate(list(strain.keys())):
        logger.debug("Generating a spectrogram for {}".format(key))
        figs[key] = plt.figure(figsize=(12, 6))
        try:
            try:
                specgram = strain[key].spectrogram(
                    20, fftlength=8, overlap=4
                ) ** (1 / 2.)
            except Exception as e:
                specgram = strain[key].spectrogram(strain[key].duration / 2.)
            plt.pcolormesh(specgram, vmin=vmin, vmax=vmax, norm='log', cmap=cmap)
            plt.ylim(ylim)
            plt.ylabel(r'Frequency [$Hz$]')
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_xscale('minutes', epoch=strain[key].times[0])
            cbar = plt.colorbar()
            cbar.set_label(r"ASD [strain/$\sqrt{Hz}$]")
        except Exception as e:
            logger.info(
                "Failed to generate an spectrogram for {} because {}".format(key, e)
            )
    return figs


def omegascan(
    strain, gps, window=4, vmin=0, vmax=25, cmap="viridis", ylim=[40, 2000],
    **kwargs
):
    """Generate an omegascan from the timeseries

    Parameters
    ----------
    strain: dict
        dictionary of gw.py timeseries objects containing the strain data for
        each IFO
    gps: float
        gps time you wish to center your omegascan around
    window: float, optional
        window around gps time to generate omagescan for. Default 4s
    vmin: float, optional
        minimum for the colormap
    vmax: float, optional
        maximum for the colormap
    cmap: str, optional
        cmap for the plot. See `matplotlib.pyplot.colormaps()` for options
    ylim: list, optional
        list to give the lower and upper bound of the plot
    """
    detectors = list(strain.keys())
    figs = {}
    for num, key in enumerate(detectors):
        logger.debug("Generating an omegascan for {}".format(key))
        try:
            try:
                cropped_data = strain[key].crop(gps - window, gps + window)
                qtransform = cropped_data.q_transform(
                    gps=gps, outseg=(gps - 0.5 * window, gps + 0.5 * window),
                    logf=True
                )
            except Exception as e:
                qtransform = strain[key].q_transform(gps=gps, logf=True)
            figs[key] = plt.figure(figsize=(12, 6))
            plt.pcolormesh(qtransform, vmin=vmin, vmax=vmax, cmap=cmap)
            plt.ylim(ylim)
            plt.ylabel(r'Frequency [$Hz$]')
            ax = plt.gca()
            ax.set_xscale('seconds', epoch=gps)
            ax.set_yscale('log')
            cbar = plt.colorbar()
            cbar.set_label("Signal-to-noise ratio")
        except Exception as e:
            logger.info(
                "Failed to generate an omegascan for {} because {}".format(key, e)
            )
            figs[key] = plt.figure(figsize=(12, 6))
    return figs
