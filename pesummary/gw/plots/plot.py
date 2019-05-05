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
import corner

import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.time import Time

from lal import MSUN_SI, PC_SI
try:
    import lalsimulation as lalsim
    LALSIMULATION = True
except ImportError:
    LALSIMULATION = None

PSD_COLORS = {"H1": "#1b9e77", "L1": "#d95f02", "V1": "#7570b3"}


def _make_corner_plot(samples, params, latex_labels, **kwargs):
    """Generate the corner plots for a given approximant

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from the command line
    samples: nd list
        nd list of samples for each parameter for a given approximant
    params: list
        list of parameters associated with each element in samples
    approximant: str
        name of approximant that was used to generate the samples
    latex_labels: dict
        dictionary of latex labels for each parameter
    """
    logger.debug("Generating the corner plot")
    # set the default kwargs
    default_kwargs = dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3)
    corner_parameters = [
        "luminosity_distance", "dec", "a_2", "a_1", "geocent_time", "phi_jl",
        "psi", "ra", "phase", "mass_2", "mass_1", "phi_12", "tilt_2", "iota",
        "tilt_1", "chi_p", "chirp_mass", "mass_ratio", "symmetric_mass_ratio",
        "total_mass", "chi_eff", "redshift", "mass_1_source", "mass_2_source",
        "total_mass_source", "chirp_mass_source"]
    included_parameters = [i for i in params if i in corner_parameters]
    xs = np.zeros([len(included_parameters), len(samples)])
    for num, i in enumerate(included_parameters):
        xs[num] = [j[params.index("%s" % (i))] for j in samples]
    default_kwargs['range'] = [1.0] * len(included_parameters)
    default_kwargs["labels"] = [latex_labels[i] for i in included_parameters]
    figure = corner.corner(xs.T, **default_kwargs)
    # grab the axes of the subplots
    axes = figure.get_axes()
    extent = axes[0].get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    width, height = extent.width, extent.height
    width *= figure.dpi
    height *= figure.dpi
    return figure, included_parameters


def __antenna_response(name, ra, dec, psi, time_gps):
    """Calculate the antenna response function

    Parameters
    ----------
    name: str
        name of the detector you wish to calculate the antenna response
        function for
    ra: float
        right ascension of the source
    dec: float
        declination of the source
    psi: float
        polarisation of the source
    time_gps: float
        gps time of merger
    """
    gmst = Time(time_gps, format='gps', location=(0, 0))
    corrected_ra = gmst.sidereal_time('mean').rad - ra
    if not LALSIMULATION:
        raise Exception("lalsimulation could not be imported. please install "
                        "lalsuite to be able to use all features")
    detector = lalsim.DetectorPrefixToLALDetector(str(name))

    x0 = -np.cos(psi) * np.sin(corrected_ra) - \
        np.sin(psi) * np.cos(corrected_ra) * np.sin(dec)
    x1 = -np.cos(psi) * np.cos(corrected_ra) + \
        np.sin(psi) * np.sin(corrected_ra) * np.sin(dec)
    x2 = np.sin(psi) * np.cos(dec)
    x = np.array([x0, x1, x2])
    dx = detector.response.dot(x)

    y0 = np.sin(psi) * np.sin(corrected_ra) - \
        np.cos(psi) * np.cos(corrected_ra) * np.sin(dec)
    y1 = np.sin(psi) * np.cos(corrected_ra) + \
        np.cos(psi) * np.sin(corrected_ra) * np.sin(dec)
    y2 = np.cos(psi) * np.cos(dec)
    y = np.array([y0, y1, y2])
    dy = detector.response.dot(y)

    fplus = (x * dx - y * dy).sum()
    fcross = (x * dy + y * dx).sum()

    return fplus, fcross


def _waveform_plot(detectors, maxL_params, **kwargs):
    """Plot the maximum likelihood waveform for a given approximant.

    Parameters
    ----------
    detectors: list
        list of detectors that you want to generate waveforms for
    maxL_params: dict
        dictionary of maximum likelihood parameter values
    kwargs: dict
        dictionary of optional keyword arguments
    """
    logger.debug("Generating the maximum likelihood waveform plot")
    if not LALSIMULATION:
        raise Exception("lalsimulation could not be imported. please install "
                        "lalsuite to be able to use all features")
    delta_frequency = kwargs.get("delta_f", 1. / 256)
    minimum_frequency = kwargs.get("f_min", 5.)
    maximum_frequency = kwargs.get("f_max", 1000.)
    frequency_array = np.arange(minimum_frequency, maximum_frequency,
                                delta_frequency)

    approx = lalsim.GetApproximantFromString(maxL_params["approximant"])
    mass_1 = maxL_params["mass_1"] * MSUN_SI
    mass_2 = maxL_params["mass_2"] * MSUN_SI
    luminosity_distance = maxL_params["luminosity_distance"] * PC_SI * 10**6
    if "phi_jl" in maxL_params.keys():
        iota, S1x, S1y, S1z, S2x, S2y, S2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                maxL_params["theta_jn"], maxL_params["phi_jl"], maxL_params["tilt_1"],
                maxL_params["tilt_2"], maxL_params["phi_12"], maxL_params["a_1"],
                maxL_params["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
                maxL_params["phase"])
    else:
        iota, S1x, S1y, S1z, S2x, S2y, S2z = maxL_params["iota"], 0., 0., 0., \
            0., 0., 0.
    phase = maxL_params["phase"] if "phase" in maxL_params.keys() else 0.0
    h_plus, h_cross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance, iota,
        phase, 0.0, 0.0, 0.0, delta_frequency, minimum_frequency,
        maximum_frequency, kwargs.get("f_ref", 10.), None, approx)
    h_plus = h_plus.data.data
    h_cross = h_cross.data.data
    h_plus = h_plus[:len(frequency_array)]
    h_cross = h_cross[:len(frequency_array)]
    fig = plt.figure()
    colors = [PSD_COLORS[i] for i in detectors]
    for num, i in enumerate(detectors):
        ar = __antenna_response(i, maxL_params["ra"], maxL_params["dec"],
                                maxL_params["psi"], maxL_params["geocent_time"])
        plt.plot(frequency_array, abs(h_plus * ar[0] + h_cross * ar[1]),
                 color=colors[num], linewidth=1.0, label=i)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Frequency $[Hz]$", fontsize=16)
    plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    return fig


def _waveform_comparison_plot(maxL_params_list, colors, labels,
                              **kwargs):
    """Generate a plot which compares the maximum likelihood waveforms for
    each approximant.

    Parameters
    ----------
    maxL_params_list: list
        list of dictionaries containing the maximum likelihood parameter
        values for each approximant
    colors: list
        list of colors to be used to differentiate the different approximants
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    kwargs: dict
        dictionary of optional keyword arguments
    """
    logger.debug("Generating the maximum likelihood waveform comparison plot "
                 "for H1")
    if not LALSIMULATION:
        raise Exception("LALSimulation could not be imported. Please install "
                        "LALSuite to be able to use all features")
    delta_frequency = kwargs.get("delta_f", 1. / 256)
    minimum_frequency = kwargs.get("f_min", 5.)
    maximum_frequency = kwargs.get("f_max", 1000.)
    frequency_array = np.arange(minimum_frequency, maximum_frequency,
                                delta_frequency)

    fig = plt.figure()
    for num, i in enumerate(maxL_params_list):
        approx = lalsim.GetApproximantFromString(i["approximant"])
        mass_1 = i["mass_1"] * MSUN_SI
        mass_2 = i["mass_2"] * MSUN_SI
        luminosity_distance = i["luminosity_distance"] * PC_SI * 10**6
        if "phi_jl" in i.keys():
            iota, S1x, S1y, S1z, S2x, S2y, S2z = \
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    i["theta_jn"], i["phi_jl"], i["tilt_1"],
                    i["tilt_2"], i["phi_12"], i["a_1"],
                    i["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
                    i["phase"])
        else:
            iota, S1x, S1y, S1z, S2x, S2y, S2z = i["iota"], 0., 0., 0., \
                0., 0., 0.
        phase = i["phase"] if "phase" in i.keys() else 0.0
        h_plus, h_cross = lalsim.SimInspiralChooseFDWaveform(
            mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance,
            iota, phase, 0.0, 0.0, 0.0, delta_frequency, minimum_frequency,
            maximum_frequency, kwargs.get("f_ref", 10.), None, approx)
        h_plus = h_plus.data.data
        h_cross = h_cross.data.data
        h_plus = h_plus[:len(frequency_array)]
        h_cross = h_cross[:len(frequency_array)]
        ar = __antenna_response("H1", i["ra"], i["dec"], i["psi"],
                                i["geocent_time"])
        plt.plot(frequency_array, abs(h_plus * ar[0] + h_cross * ar[1]),
                 color=colors[num], label=labels[num], linewidth=2.0)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel(r"Frequency $[Hz]$", fontsize=16)
    plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
    plt.tight_layout()
    return fig


def _sky_map_plot(ra, dec, **kwargs):
    """Plot the sky location of the source for a given approximant

    Parameters
    ----------
    ra: list
        list of samples for right ascension
    dec: list
        list of samples for declination
    kwargs: dict
        optional keyword arguments
    """
    ra = [-i + np.pi for i in ra]
    logger.debug("Generating the sky map plot")
    fig = plt.figure()
    ax = plt.subplot(111, projection="hammer")
    ax.cla()
    ax.grid()
    ax.set_xticklabels([
        r"$2^{h}$", r"$4^{h}$", r"$6^{h}$", r"$8^{h}$", r"$10^{h}$",
        r"$12^{h}$", r"$14^{h}$", r"$16^{h}$", r"$18^{h}$", r"$20^{h}$",
        r"$22^{h}$"])
    levels = [1.0 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9. / 2.)]

    H, X, Y = np.histogram2d(ra, dec, bins=50)
    H = gaussian_filter(H, kwargs.get("smooth", 0.9))
    Hflat = H.flatten()
    indicies = np.argsort(Hflat)[::-1]
    Hflat = Hflat[indicies]

    CF = np.cumsum(Hflat)
    CF /= CF[-1]

    V = np.empty(len(levels))
    for num, i in enumerate(levels):
        try:
            V[num] = Hflat[CF <= i][-1]
        except Exception:
            V[num] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]), ])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]), ])

    plt.contour(X2, Y2, H2.T, V, colors=["#AED6F1", "#3498DB", "#21618C"],
                linewidths=2.0)

    xticks = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    ax.set_xticks(xticks)
    labels = [r"$%s^{h}$" % (np.round((i + np.pi) * 3.82, 1)) for i in xticks]
    ax.set_xticklabels(labels[::-1])
    return fig


def _sky_map_comparison_plot(ra_list, dec_list, labels, colors, **kwargs):
    """Generate a plot that compares the sky location for multiple approximants

    Parameters
    ----------
    ra_list: 2d list
        list of samples for right ascension for each approximant
    dec_list: 2d list
        list of samples for declination for each approximant
    approximants: list
        list of approximants used to generate the samples
    colors: list
        list of colors to be used to differentiate the different approximants
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    kwargs: dict
        optional keyword arguments
    """
    ra_list = [[-i + np.pi for i in j] for j in ra_list]
    logger.debug("Generating the sky map comparison plot")
    fig = plt.figure()
    ax = plt.subplot(111, projection="hammer")
    ax.cla()
    ax.grid()
    ax.set_xticklabels([
        r"$2^{h}$", r"$4^{h}$", r"$6^{h}$", r"$8^{h}$", r"$10^{h}$",
        r"$12^{h}$", r"$14^{h}$", r"$16^{h}$", r"$18^{h}$", r"$20^{h}$",
        r"$22^{h}$"])
    levels = [1.0 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9. / 2.)]
    for num, i in enumerate(ra_list):
        H, X, Y = np.histogram2d(i, dec_list[num], bins=50)
        H = gaussian_filter(H, kwargs.get("smooth", 0.9))
        Hflat = H.flatten()
        indicies = np.argsort(Hflat)[::-1]
        Hflat = Hflat[indicies]

        CF = np.cumsum(Hflat)
        CF /= CF[-1]

        V = np.empty(len(levels))
        for num2, j in enumerate(levels):
            try:
                V[num2] = Hflat[CF <= j][-1]
            except Exception:
                V[num2] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                             X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]), ])
        Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                             Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]), ])
        CS = plt.contour(X2, Y2, H2.T, V, colors=colors[num], linewidths=2.0)
        CS.collections[0].set_label(labels[num])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.,
               mode="expand", ncol=2)
    xticks = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    ax.set_xticks(xticks)
    labels = [r"$%s^{h}$" % (np.round((i + np.pi) * 3.82, 1)) for i in xticks]
    ax.set_xticklabels(labels[::-1])
    return fig


def __get_cutoff_indices(flow, fhigh, df, N):
    """
    Gets the indices of a frequency series at which to stop an overlap
    calculation.

    Parameters
    ----------
    flow: float
        The frequency (in Hz) of the lower index.
    fhigh: float
        The frequency (in Hz) of the upper index.
    df: float
        The frequency step (in Hz) of the frequency series.
    N: int
        The number of points in the **time** series. Can be odd
        or even.

    Returns
    -------
    kmin: int
    kmax: int
    """
    if flow:
        kmin = int(flow / df)
    else:
        kmin = 1
    if fhigh:
        kmax = int(fhigh / df)
    else:
        kmax = int((N + 1) / 2.)
    return kmin, kmax


def _sky_sensitivity(network, resolution, maxL_params, **kwargs):
    """Generate the sky sensitivity for a given network

    Parameters
    ----------
    network: list
        list of detectors you want included in your sky sensitivity plot
    resolution: float
        resolution of the skymap
    maxL_params: dict
        dictionary of waveform parameters for the maximum likelihood waveform
    """
    logger.debug("Generating the sky sensitivity for %s" % (network))
    if not LALSIMULATION:
        raise Exception("LALSimulation could not be imported. Please install "
                        "LALSuite to be able to use all features")
    delta_frequency = kwargs.get("delta_f", 1. / 256)
    minimum_frequency = kwargs.get("f_min", 20.)
    maximum_frequency = kwargs.get("f_max", 1000.)
    frequency_array = np.arange(minimum_frequency, maximum_frequency,
                                delta_frequency)

    approx = lalsim.GetApproximantFromString(maxL_params["approximant"])
    mass_1 = maxL_params["mass_1"] * MSUN_SI
    mass_2 = maxL_params["mass_2"] * MSUN_SI
    luminosity_distance = maxL_params["luminosity_distance"] * PC_SI * 10**6
    iota, S1x, S1y, S1z, S2x, S2y, S2z = \
        lalsim.SimInspiralTransformPrecessingNewInitialConditions(
            maxL_params["iota"], maxL_params["phi_jl"], maxL_params["tilt_1"],
            maxL_params["tilt_2"], maxL_params["phi_12"], maxL_params["a_1"],
            maxL_params["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
            maxL_params["phase"])
    h_plus, h_cross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance, iota,
        maxL_params["phase"], 0.0, 0.0, 0.0, delta_frequency, minimum_frequency,
        maximum_frequency, kwargs.get("f_ref", 10.), None, approx)
    h_plus = h_plus.data.data
    h_cross = h_cross.data.data
    h_plus = h_plus[:len(frequency_array)]
    h_cross = h_cross[:len(frequency_array)]
    psd = {}
    psd["H1"] = psd["L1"] = np.array([
        lalsim.SimNoisePSDaLIGOZeroDetHighPower(i) for i in frequency_array])
    psd["V1"] = np.array([lalsim.SimNoisePSDVirgo(i) for i in frequency_array])
    kmin, kmax = __get_cutoff_indices(minimum_frequency, maximum_frequency,
                                      delta_frequency, (len(h_plus) - 1) * 2)
    ra = np.arange(-np.pi, np.pi, resolution)
    dec = np.arange(-np.pi, np.pi, resolution)
    X, Y = np.meshgrid(ra, dec)
    N = np.zeros([len(dec), len(ra)])

    indices = np.ndindex(len(ra), len(dec))
    for ind in indices:
        ar = {}
        SNR = {}
        for i in network:
            ard = __antenna_response(i, ra[ind[0]], dec[ind[1]],
                                     maxL_params["psi"], maxL_params["geocent_time"])
            ar[i] = [ard[0], ard[1]]
            strain = np.array(h_plus * ar[i][0] + h_cross * ar[i][1])
            integrand = np.conj(strain[kmin:kmax]) * strain[kmin:kmax] / psd[i][kmin:kmax]
            integrand = integrand[:-1]
            SNR[i] = np.sqrt(4 * delta_frequency * np.sum(integrand).real)
            ar[i][0] *= SNR[i]
            ar[i][1] *= SNR[i]
        numerator = 0.0
        denominator = 0.0
        for i in network:
            numerator += sum(i**2 for i in ar[i])
            denominator += SNR[i]**2
        N[ind[1]][ind[0]] = (((numerator / denominator)**0.5))
    fig = plt.figure()
    ax = plt.subplot(111, projection="hammer")
    ax.cla()
    ax.grid()
    plt.pcolormesh(X, Y, N)
    ax.set_xticklabels([
        r"$22^{h}$", r"$20^{h}$", r"$18^{h}$", r"$16^{h}$", r"$14^{h}$",
        r"$12^{h}$", r"$10^{h}$", r"$8^{h}$", r"$6^{h}$", r"$4^{h}$",
        r"$2^{h}$"])
    return fig


def _time_domain_waveform(detectors, maxL_params, **kwargs):
    """
    Plot the maximum likelihood waveform for a given approximant
    in the time domain.

    Parameters
    ----------
    detectors: list
        list of detectors that you want to generate waveforms for
    maxL_params: dict
        dictionary of maximum likelihood parameter values
    kwargs: dict
        dictionary of optional keyword arguments
    """
    logger.debug("Generating the maximum likelihood waveform time domain plot")
    if not LALSIMULATION:
        raise Exception("lalsimulation could not be imported. please install "
                        "lalsuite to be able to use all features")
    delta_t = 1. / 4096.
    minimum_frequency = kwargs.get("f_min", 5.)
    t_start = maxL_params['geocent_time']
    t_finish = maxL_params['geocent_time'] + 4.
    time_array = np.arange(t_start, t_finish, delta_t)

    approx = lalsim.GetApproximantFromString(maxL_params["approximant"])
    mass_1 = maxL_params["mass_1"] * MSUN_SI
    mass_2 = maxL_params["mass_2"] * MSUN_SI
    luminosity_distance = maxL_params["luminosity_distance"] * PC_SI * 10**6
    if "phi_jl" in maxL_params.keys():
        iota, S1x, S1y, S1z, S2x, S2y, S2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                maxL_params["theta_jn"], maxL_params["phi_jl"], maxL_params["tilt_1"],
                maxL_params["tilt_2"], maxL_params["phi_12"], maxL_params["a_1"],
                maxL_params["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
                maxL_params["phase"])
    else:
        iota, S1x, S1y, S1z, S2x, S2y, S2z = maxL_params["iota"], 0., 0., 0., \
            0., 0., 0.
    phase = maxL_params["phase"] if "phase" in maxL_params.keys() else 0.0
    h_plus, h_cross = lalsim.SimInspiralChooseTDWaveform(
        mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance, iota,
        phase, 0.0, 0.0, 0.0, delta_t, minimum_frequency,
        kwargs.get("f_ref", 10.), None, approx)

    h_plus = h_plus.data.data
    h_cross = h_cross.data.data
    h_plus = h_plus[:len(time_array)]
    h_cross = h_cross[:len(time_array)]
    fig = plt.figure()
    colors = [PSD_COLORS[i] for i in detectors]
    for num, i in enumerate(detectors):
        ar = __antenna_response(i, maxL_params["ra"], maxL_params["dec"],
                                maxL_params["psi"], maxL_params["geocent_time"])
        plt.plot(time_array, (h_plus * ar[0] + h_cross * ar[1]),
                 color=colors[num], linewidth=1.0, label=i)
    plt.xlabel(r"Time $[s]$", fontsize=16)
    plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    return fig


def _time_domain_waveform_comparison_plot(maxL_params_list, colors, labels,
                                          **kwargs):
    """Generate a plot which compares the maximum likelihood waveforms for
    each approximant.

    Parameters
    ----------
    maxL_params_list: list
        list of dictionaries containing the maximum likelihood parameter
        values for each approximant
    colors: list
        list of colors to be used to differentiate the different approximants
    approximant_labels: list, optional
        label to prepend the approximant in the legend
    kwargs: dict
        dictionary of optional keyword arguments
    """
    logger.debug("Generating the maximum likelihood time domain waveform "
                 "comparison plot for H1")
    if not LALSIMULATION:
        raise Exception("LALSimulation could not be imported. Please install "
                        "LALSuite to be able to use all features")
    delta_t = 1. / 4096.
    minimum_frequency = kwargs.get("f_min", 5.)

    fig = plt.figure()
    for num, i in enumerate(maxL_params_list):
        t_start = i['geocent_time']
        t_finish = i['geocent_time'] + 4.
        time_array = np.arange(t_start, t_finish, delta_t)

        approx = lalsim.GetApproximantFromString(i["approximant"])
        mass_1 = i["mass_1"] * MSUN_SI
        mass_2 = i["mass_2"] * MSUN_SI
        luminosity_distance = i["luminosity_distance"] * PC_SI * 10**6
        if "phi_jl" in i.keys():
            iota, S1x, S1y, S1z, S2x, S2y, S2z = \
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    i["theta_jn"], i["phi_jl"], i["tilt_1"],
                    i["tilt_2"], i["phi_12"], i["a_1"],
                    i["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
                    i["phase"])
        else:
            iota, S1x, S1y, S1z, S2x, S2y, S2z = i["iota"], 0., 0., 0., \
                0., 0., 0.
        phase = i["phase"] if "phase" in i.keys() else 0.0
        h_plus, h_cross = lalsim.SimInspiralChooseTDWaveform(
            mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance,
            iota, phase, 0.0, 0.0, 0.0, delta_t, minimum_frequency,
            kwargs.get("f_ref", 10.), None, approx)

        h_plus = h_plus.data.data
        h_cross = h_cross.data.data
        h_plus = h_plus[:len(time_array)]
        h_cross = h_cross[:len(time_array)]
        ar = __antenna_response("H1", i["ra"], i["dec"], i["psi"],
                                i["geocent_time"])
        plt.plot(time_array, abs(h_plus * ar[0] + h_cross * ar[1]),
                 color=colors[num], label=labels[num], linewidth=2.0)
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel(r"Time $[s]$", fontsize=16)
    plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
    plt.tight_layout()
    return fig


def _psd_plot(frequencies, strains, colors=None, labels=None):
    """Superimpose all PSD plots onto a single figure.

    Parameters
    ----------
    frequencies: nd list
        list of all frequencies used for each psd file
    strains: nd list
        list of all strains used for each psd file
    colors: optional, list
        list of colors to be used to differentiate the different PSDs
    labels: optional, list
        list of lavels for each PSD
    """
    fig = plt.figure()
    if not colors and labels in list(PSD_COLORS.keys()):
        colors = [PSD_COLORS[i] for i in labels]
    elif not colors:
        colors = ['r', 'b', 'orange', 'c', 'g', 'purple']
        while len(colors) <= len(labels):
            colors += colors
    for num, i in enumerate(frequencies):
        plt.loglog(i, strains[num], color=colors[num], label=labels[num])
    plt.xlabel(r"Frequency $[Hz]$", fontsize=16)
    plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
    plt.legend(loc="best")
    plt.tight_layout()
    return fig


def _calibration_envelope_plot(frequency, calibration_envelopes, ifos,
                               colors=None):
    """Generate a plot showing the calibration envelope

    Parameters
    ----------
    frequency: array
        frequency bandwidth that you would like to use
    calibration_envelopes: nd list
        list containing the calibration envelope data for different IFOs
    ifos: list
        list of IFOs that are associated with the calibration envelopes
    colors: list, optional
        list of colors to be used to differentiate the different calibration
        envelopes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    if not colors and ifos in list(PSD_COLORS.keys()):
        colors = [PSD_COLORS[i] for i in ifos]
    elif not colors:
        colors = ['r', 'b', 'orange', 'c', 'g', 'purple']
        while len(colors) <= len(ifos):
            colors += colors

    for num, i in enumerate(calibration_envelopes):
        interp = [np.interp(
            frequency, i[:, 0], i[:, j], left=k, right=k) for j, k in zip(
                range(1, 7), [1, 0, 1, 0, 1, 0])]
        amp_median = (1 - interp[0]) * 100
        phase_median = interp[1] * 180. / np.pi
        amp_lower_sigma = (1 - interp[2]) * 100
        phase_lower_sigma = interp[3] * 180. / np.pi
        amp_upper_sigma = (1 - interp[4]) * 100
        phase_upper_sigma = interp[5] * 180. / np.pi
        ax1.plot(frequency, amp_median, color=colors[num], label=ifos[num])
        ax1.plot(frequency, amp_upper_sigma, color=colors[num], linestyle="--")
        ax1.plot(frequency, amp_lower_sigma, color=colors[num], linestyle="--")
        ax1.fill_between(
            frequency, amp_upper_sigma, amp_lower_sigma, color=colors[num],
            alpha=0.4)
        ax1.set_ylabel(r"Amplitude deviation $[\%]$")
        ax1.legend(loc="best")
        ax2.plot(frequency, phase_median, color=colors[num], label=ifos[num])
        ax2.plot(frequency, phase_upper_sigma, color=colors[num],
                 linestyle="--")
        ax2.plot(frequency, phase_lower_sigma, color=colors[num],
                 linestyle="--")
        ax2.fill_between(
            frequency, phase_upper_sigma, phase_lower_sigma, color=colors[num],
            alpha=0.4)
        ax2.set_ylabel(r"Phase deviation $[\degree]$")
    plt.xscale('log')
    plt.xlabel(r"Frequency $[Hz]$", fontsize=16)
    plt.tight_layout()
    return fig
