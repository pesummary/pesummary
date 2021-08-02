# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.plots.figure import figure
from pesummary.utils.utils import (
    logger, get_matplotlib_style_file, _check_latex_install
)
from pesummary import conf
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
import matplotlib.style
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_check_latex_install()


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
        figs[key], ax = figure(figsize=(12, 6), gca=True)
        try:
            try:
                specgram = strain[key].spectrogram(
                    20, fftlength=8, overlap=4
                ) ** (1 / 2.)
            except Exception as e:
                specgram = strain[key].spectrogram(strain[key].duration / 2.)
            im = ax.pcolormesh(
                specgram, vmin=vmin, vmax=vmax, norm='log', cmap=cmap
            )
            ax.set_ylim(ylim)
            ax.set_ylabel(r'Frequency [$Hz$]')
            ax.set_yscale('log')
            ax.set_xscale('minutes', epoch=strain[key].times[0])
            cbar = figs[key].colorbar(im)
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
            figs[key], ax = figure(figsize=(12, 6), gca=True)
            im = ax.pcolormesh(qtransform, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_ylim(ylim)
            ax.set_ylabel(r'Frequency [$Hz$]')
            ax.set_xscale('seconds', epoch=gps)
            ax.set_yscale('log')
            cbar = figs[key].colorbar(im)
            cbar.set_label("Signal-to-noise ratio")
        except Exception as e:
            logger.info(
                "Failed to generate an omegascan for {} because {}".format(key, e)
            )
            figs[key] = figure(figsize=(12, 6), gca=False)
    return figs


def time_domain_strain_data(
    strain, bandpass_frequencies=[50, 250], notches=[60., 120., 180.],
    window=None, merger_time=None, template=None, grid=False,
    xlabel="UTC", UTC_format="%B %d %Y, %H:%M:%S"
):
    """Plot the strain data in the time domain. Code based on the GW150914
    tutorial provided by gwpy:
    https://gwpy.github.io/docs/latest/examples/signal/gw150914.html

    Parameters
    ----------
    """
    from gwpy.signal import filter_design
    import matplotlib.patheffects as pe

    detectors = list(strain.keys())
    figs = {}
    for num, key in enumerate(detectors):
        if bandpass_frequencies is not None and notches is not None:
            bp = filter_design.bandpass(*bandpass_frequencies, strain[key].sample_rate)
            zpks = [
                filter_design.notch(line, strain[key].sample_rate) for line in
                notches
            ]
            zpk = filter_design.concatenate_zpks(bp, *zpks)
            hfilt = strain[key].filter(zpk, filtfilt=True)
            _strain = hfilt.crop(*hfilt.span.contract(1))
        else:
            _strain = strain[key]
        figs[key], ax = figure(figsize=(12, 6), gca=True)
        if merger_time is not None:
            x = _strain.times.value - merger_time
            _xlabel = "Time (seconds) from {} {}"
            if xlabel == "UTC":
                from lal import gpstime

                _merger_time = gpstime.gps_to_str(merger_time, form=UTC_format)
                xlabel = _xlabel.format(_merger_time, "UTC")
            else:
                xlabel = _xlabel.format(merger_time, "GPS")
        else:
            x = _strain.times
            xlabel = "GPS [$s$]"
        if template is not None:
            if not isinstance(template[key], dict):
                template[key] = {"template": template[key]}
            _x = template[key]["template"].times.value
            if merger_time is not None:
                _x -= merger_time
            ax.plot(
                _x, template[key], color='gray', linewidth=3.,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='k'), pe.Normal()],
                label="Template"
            )
            _bounds = ["upper", "lower", "bound_times"]
            if all(bound in template[key].keys() for bound in _bounds):
                _x = template[key]["bound_times"]
                if merger_time is not None:
                    _x -= merger_time
                ax.fill_between(
                    _x, template[key]["upper"], template[key]["lower"],
                    color='lightgray', label="Uncertainty"
                )
        ax.plot(
            x, _strain, color=GW_OBSERVATORY_COLORS[key], linewidth=3.,
            label="Detector data"
        )
        if window is not None:
            ax.set_xlim(*window)
        ax.set_xlabel(xlabel)
        ax.grid(b=grid)
        ax.legend()
    return figs


def frequency_domain_strain_data(
    strain, window=True, window_kwargs={"roll_off": 0.2}, resolution=1. / 512,
    fmin=-np.inf, fmax=np.inf, asd={}
):
    """Plot the strain data in the frequency domain

    Parameters
    ----------
    strain: dict
        dictionary of gw.py timeseries objects containing the strain data for
        each IFO
    window: Bool, optional
        if True, apply a window to the data before applying FFT to the data.
        Default True
    window_kwargs: dict, optional
        optional kwargs for the window function
    resolution: float, optional
        resolution to downsample the frequency domain data. Default 1./512
    fmin: float, optional
        lowest frequency to start plotting the data
    fmax: float, optional
        highest frequency to stop plotting the data
    asd: dict, optional
        dictionary containing the ASDs for each detector to plot ontop of the
        detector data
    """
    detectors = list(strain.keys())
    figs = {}
    if not isinstance(asd, dict):
        raise ValueError(
            "Please provide the asd as a dictionary keyed by the detector"
        )
    elif not all(ifo in asd.keys() for ifo in detectors):
        logger.info(
            ""
        )
    for num, key in enumerate(detectors):
        logger.debug("Plotting strain data in frequency domain")
        if window:
            from scipy.signal.windows import tukey

            if "alpha" in window_kwargs.keys():
                alpha = window_kwargs["alpha"]
            elif "roll_off" and "duration" in window_kwargs.keys():
                alpha = 2 * window_kwargs["roll_off"] / window_kwargs["duration"]
            elif "roll_off" in window_kwargs.keys():
                alpha = 2 * window_kwargs["roll_off"] / strain[key].duration.value
            else:
                raise ValueError(
                    "Please either provide 'alpha' (the shape parameter of the "
                    "Tukey window) or the 'roll_off' for the Tukey window."
                )
            _window = tukey(len(strain[key].value), alpha=alpha)
        else:
            _window = None
        freq = strain[key].average_fft(window=_window)
        freq = freq.interpolate(resolution)
        freq = np.absolute(freq) / freq.df.value**0.5
        figs[key], ax = figure(figsize=(12, 6), gca=True)
        inds = np.where(
            (freq.frequencies.value > fmin) & (freq.frequencies.value < fmax)
        )
        ax.loglog(
            freq.frequencies[inds], freq[inds], label=key,
            color=GW_OBSERVATORY_COLORS[key], alpha=0.2
        )
        if key in asd.keys():
            inds = np.where((asd[key][:, 0] > fmin) & (asd[key][:, 0] < fmax))
            ax.loglog(
                asd[key][:, 0][inds], asd[key][:, 1][inds],
                label="%s ASD" % (key), color=GW_OBSERVATORY_COLORS[key]
            )
        ax.set_xlabel(r"Frequency [$Hz$]")
        ax.set_ylabel(r"Strain [strain/$\sqrt{Hz}$]")
        ax.legend()
        figs[key].tight_layout()
    return figs
