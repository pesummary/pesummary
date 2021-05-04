# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import (
    logger, number_of_columns_for_legend, _check_latex_install,
    get_matplotlib_style_file
)
from pesummary.utils.decorators import no_latex_plot
from pesummary.gw.plots.bounds import default_bounds
from pesummary.core.plots.seaborn.kde import kdeplot
from pesummary.core.plots.figure import figure, subplots, ExistingFigure
from pesummary.core.plots.plot import _default_legend_kwargs
from pesummary import conf

import os
import matplotlib.style
import matplotlib.lines as mlines
import corner
import numpy as np
import math
from scipy.ndimage import gaussian_filter
from astropy.time import Time

_check_latex_install()

from lal import MSUN_SI, PC_SI

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
try:
    import lalsimulation as lalsim
    LALSIMULATION = True
except ImportError:
    LALSIMULATION = None


def _return_bounds(param, samples, comparison=False):
    """Return the bounds for a given param

    Parameters
    ----------
    param: str
        name of the parameter you wish to get bounds for
    samples: list/np.ndarray
        array or list of array of posterior samples for param
    comparison: Bool, optional
        True if samples is a list of array's of posterior samples
    """
    xlow, xhigh = None, None
    if param in default_bounds.keys():
        bounds = default_bounds[param]
        if "low" in bounds.keys():
            xlow = bounds["low"]
        if "high" in bounds.keys():
            if isinstance(bounds["high"], str) and "mass_1" in bounds["high"]:
                if comparison:
                    xhigh = np.max([np.max(i) for i in samples])
                else:
                    xhigh = np.max(samples)
            else:
                xhigh = bounds["high"]
    return xlow, xhigh


def _add_default_bounds_to_kde_kwargs_dict(
    kde_kwargs, param, samples, comparison=False
):
    """Add default kde bounds to the a dictionary of kwargs

    Parameters
    ----------
    kde_kwargs: dict
        dictionary of kwargs to pass to the kde class
    param: str
        name of the parameter you wish to plot
    samples: list
        list of samples for param
    """
    from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde

    xlow, xhigh = _return_bounds(param, samples, comparison=comparison)
    kde_kwargs["xlow"] = xlow
    kde_kwargs["xhigh"] = xhigh
    kde_kwargs["kde_kernel"] = bounded_1d_kde
    return kde_kwargs


def _1d_histogram_plot(
    param, samples, *args, kde_kwargs={}, bounded=True, **kwargs
):
    """Generate the 1d histogram plot for a given parameter for a given
    approximant.

    Parameters
    ----------
    *args: tuple
        all args passed directly to pesummary.core.plots.plot._1d_histogram_plot
        function
    kde_kwargs: dict, optional
        optional kwargs passed to the kde class
    bounded: Bool, optional
        if True, pass default 'xlow' and 'xhigh' arguments to the kde class
    **kwargs: dict, optional
        all additional kwargs passed to the
        pesummary.core.plots.plot._1d_histogram_plot function
    """
    from pesummary.core.plots.plot import _1d_histogram_plot

    if bounded:
        kde_kwargs = _add_default_bounds_to_kde_kwargs_dict(
            kde_kwargs, param, samples
        )
    return _1d_histogram_plot(
        param, samples, *args, kde_kwargs=kde_kwargs, **kwargs
    )


def _1d_histogram_plot_mcmc(
    param, samples, *args, kde_kwargs={}, bounded=True, **kwargs
):
    """Generate the 1d histogram plot for a given parameter for set of
    mcmc chains

    Parameters
    ----------
    *args: tuple
        all args passed directly to
        pesummary.core.plots.plot._1d_histogram_plot_mcmc function
    kde_kwargs: dict, optional
        optional kwargs passed to the kde class
    bounded: Bool, optional
        if True, pass default 'xlow' and 'xhigh' arguments to the kde class
    **kwargs: dict, optional
        all additional kwargs passed to the
        pesummary.core.plots.plot._1d_histogram_plot_mcmc function
    """
    from pesummary.core.plots.plot import _1d_histogram_plot_mcmc

    if bounded:
        kde_kwargs = _add_default_bounds_to_kde_kwargs_dict(
            kde_kwargs, param, samples, comparison=True
        )
    return _1d_histogram_plot_mcmc(
        param, samples, *args, kde_kwargs=kde_kwargs, **kwargs
    )


def _1d_histogram_plot_bootstrap(
    param, samples, *args, kde_kwargs={}, bounded=True, **kwargs
):
    """Generate a bootstrapped 1d histogram plot for a given parameter

    Parameters
    ----------
    param: str
        name of the parameter that you wish to plot
    samples: np.ndarray
        array of samples for param
    args: tuple
        all args passed to
        pesummary.core.plots.plot._1d_histogram_plot_bootstrap function
    kde_kwargs: dict, optional
        optional kwargs passed to the kde class
    bounded: Bool, optional
        if True, pass default 'xlow' and 'xhigh' arguments to the kde class
    **kwargs: dict, optional
        all additional kwargs passed to the
        pesummary.core.plots.plot._1d_histogram_plot_bootstrap function
    """
    from pesummary.core.plots.plot import _1d_histogram_plot_bootstrap

    if bounded:
        kde_kwargs = _add_default_bounds_to_kde_kwargs_dict(
            kde_kwargs, param, samples
        )
    return _1d_histogram_plot_bootstrap(
        param, samples, *args, kde_kwargs=kde_kwargs, **kwargs
    )


def _1d_comparison_histogram_plot(
    param, samples, *args, kde_kwargs={}, bounded=True, max_vline=2,
    legend_kwargs=_default_legend_kwargs, **kwargs
):
    """Generate the a plot to compare the 1d_histogram plots for a given
    parameter for different approximants.

    Parameters
    ----------
    *args: tuple
        all args passed directly to
        pesummary.core.plots.plot._1d_comparisonhistogram_plot function
    kde_kwargs: dict, optional
        optional kwargs passed to the kde class
    bounded: Bool, optional
        if True, pass default 'xlow' and 'xhigh' arguments to the kde class
    max_vline: int, optional
        if number of peaks < max_vline draw peaks as vertical lines rather
        than histogramming the data
    **kwargs: dict, optional
        all additional kwargs passed to the
        pesummary.core.plots.plot._1d_comparison_histogram_plot function
    """
    from pesummary.core.plots.plot import _1d_comparison_histogram_plot

    if bounded:
        kde_kwargs = _add_default_bounds_to_kde_kwargs_dict(
            kde_kwargs, param, samples, comparison=True
        )
    return _1d_comparison_histogram_plot(
        param, samples, *args, kde_kwargs=kde_kwargs, max_vline=max_vline,
        legend_kwargs=legend_kwargs, **kwargs
    )


def _make_corner_plot(samples, latex_labels, corner_parameters=None, **kwargs):
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
    from pesummary.core.plots.plot import _make_corner_plot

    if corner_parameters is None:
        corner_parameters = conf.gw_corner_parameters

    return _make_corner_plot(
        samples, latex_labels, corner_parameters=corner_parameters, **kwargs
    )


def _make_source_corner_plot(samples, latex_labels, **kwargs):
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
    from pesummary.core.plots.plot import _make_corner_plot

    return _make_corner_plot(
        samples, latex_labels,
        corner_parameters=conf.gw_source_frame_corner_parameters, **kwargs
    )[0]


def _make_extrinsic_corner_plot(samples, latex_labels, **kwargs):
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
    from pesummary.core.plots.plot import _make_corner_plot

    return _make_corner_plot(
        samples, latex_labels,
        corner_parameters=conf.gw_extrinsic_corner_parameters, **kwargs
    )[0]


def _make_comparison_corner_plot(
    samples, latex_labels, corner_parameters=None, colors=conf.corner_colors,
    **kwargs
):
    """Generate a corner plot which contains multiple datasets

    Parameters
    ----------
    samples: dict
        nested dictionary containing the label as key and SamplesDict as item
        for each dataset you wish to plot
    latex_labels: dict
        dictionary of latex labels for each parameter
    corner_parameters: list, optional
        corner parameters you wish to include in the plot
    colors: list, optional
        unique colors for each dataset
    **kwargs: dict
        all kwargs are passed to `corner.corner`
    """
    from pesummary.core.plots.plot import _make_comparison_corner_plot

    if corner_parameters is None:
        corner_parameters = conf.gw_corner_parameters

    return _make_comparison_corner_plot(
        samples, latex_labels, corner_parameters=corner_parameters,
        colors=colors, **kwargs
    )


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
    # Following 8 lines taken from pycbc.detector.Detector
    from astropy.units.si import sday
    reference_time = 1126259462.0
    gmst_reference = Time(
        reference_time, format='gps', scale='utc', location=(0, 0)
    ).sidereal_time('mean').rad
    dphase = (time_gps - reference_time) / float(sday.si.scale) * (2.0 * np.pi)
    gmst = (gmst_reference + dphase) % (2.0 * np.pi)
    corrected_ra = gmst - ra
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

    if hasattr(dx, "shape"):
        fplus = (x * dx - y * dy).sum(axis=0)
        fcross = (x * dy + y * dx).sum(axis=0)
    else:
        fplus = (x * dx - y * dy).sum()
        fcross = (x * dy + y * dx).sum()

    return fplus, fcross


@no_latex_plot
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
    from gwpy.plot.colors import GW_OBSERVATORY_COLORS
    if math.isnan(maxL_params["mass_1"]):
        return
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
    fig, ax = figure(gca=True)
    colors = [GW_OBSERVATORY_COLORS[i] for i in detectors]
    for num, i in enumerate(detectors):
        ar = __antenna_response(i, maxL_params["ra"], maxL_params["dec"],
                                maxL_params["psi"], maxL_params["geocent_time"])
        ax.plot(frequency_array, abs(h_plus * ar[0] + h_cross * ar[1]),
                color=colors[num], linewidth=1.0, label=i)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frequency $[Hz]$")
    ax.set_ylabel(r"Strain")
    ax.grid(b=True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@no_latex_plot
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

    fig, ax = figure(gca=True)
    for num, i in enumerate(maxL_params_list):
        if math.isnan(i["mass_1"]):
            continue
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
        ax.plot(frequency_array, abs(h_plus * ar[0] + h_cross * ar[1]),
                color=colors[num], label=labels[num], linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(b=True)
    ax.legend(loc="best")
    ax.set_xlabel(r"Frequency $[Hz]$")
    ax.set_ylabel(r"Strain")
    fig.tight_layout()
    return fig


def _ligo_skymap_plot(ra, dec, dist=None, savedir="./", nprocess=1,
                      downsampled=False, label="pesummary", time=None,
                      distance_map=True, multi_resolution=True,
                      injection=None, **kwargs):
    """Plot the sky location of the source for a given approximant using the
    ligo.skymap package

    Parameters
    ----------
    ra: list
        list of samples for right ascension
    dec: list
        list of samples for declination
    dist: list
        list of samples for the luminosity distance
    savedir: str
        path to the directory where you would like to save the output files
    nprocess: Bool
        Boolean for whether to use multithreading or not
    downsampled: Bool
        Boolean for whether the samples have been downsampled or not
    distance_map: Bool
        Boolean for whether or not to produce a distance map
    multi_resolution: Bool
        Boolean for whether or not to generate a multiresolution HEALPix map
    injection: list, optional
        List containing RA and DEC of the injection. Both must be in radians
    kwargs: dict
        optional keyword arguments
    """
    import healpy as hp
    from ligo.skymap import plot, postprocess, io
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.kde import Clustered2DSkyKDE, Clustered2Plus1DSkyKDE
    from astropy.time import Time

    if dist is not None and distance_map:
        pts = np.column_stack((ra, dec, dist))
        cls = Clustered2Plus1DSkyKDE
    else:
        pts = np.column_stack((ra, dec))
        cls = Clustered2DSkyKDE
    skypost = cls(pts, trials=5, jobs=nprocess)
    hpmap = skypost.as_healpix()
    if not multi_resolution:
        hpmap = rasterize(hpmap)
    hpmap.meta['creator'] = "pesummary"
    hpmap.meta['origin'] = 'LIGO/Virgo'
    hpmap.meta['gps_creation_time'] = Time.now().gps
    if dist is not None:
        hpmap.meta["distmean"] = float(np.mean(dist))
        hpmap.meta["diststd"] = float(np.std(dist))
    if time is not None:
        if isinstance(time, (float, int)):
            _time = time
        else:
            _time = np.mean(time)
        hpmap.meta["gps_time"] = _time

    io.write_sky_map(
        os.path.join(savedir, "%s_skymap.fits" % (label)), hpmap, nest=True
    )
    skymap, metadata = io.fits.read_sky_map(
        os.path.join(savedir, "%s_skymap.fits" % (label)), nest=None
    )
    return _ligo_skymap_plot_from_array(
        skymap, nsamples=len(ra), downsampled=downsampled, injection=injection
    )[0]


def _ligo_skymap_plot_from_array(
    skymap, nsamples=None, downsampled=False, contour=[50, 90],
    annotate=True, ax=None, colors="k", injection=None
):
    """Generate a skymap with `ligo.skymap` based on an array of probabilities

    Parameters
    ----------
    skymap: np.array
        array of probabilities
    nsamples: int, optional
        number of samples used
    downsampled: Bool, optional
        If True, add a header to the skymap saying that this plot is downsampled
    contour: list, optional
        list of contours to be plotted on the skymap. Default 50, 90
    annotate: Bool, optional
        If True, annotate the figure by adding the 90% and 50% sky areas
        by default
    ax: matplotlib.axes._subplots.AxesSubplot, optional
        Existing axis to add the plot to
    colors: str/list
        colors to use for the contours
    injection: list, optional
        List containing RA and DEC of the injection. Both must be in radians
    """
    import healpy as hp
    from ligo.skymap import plot, io
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.kde import Clustered2DSkyKDE, Clustered2Plus1DSkyKDE
    from astropy.time import Time

    if ax is None:
        fig = figure(gca=False)
        ax = fig.add_subplot(111, projection='astro hours mollweide')
        ax.grid(b=True)

    nside = hp.npix2nside(len(skymap))
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap / deg2perpix

    if downsampled:
        ax.set_title("Downsampled to %s" % (nsamples), fontdict={'fontsize': 11})

    vmax = probperdeg2.max()
    ax.imshow_hpx((probperdeg2, 'ICRS'), nested=True, vmin=0.,
                  vmax=vmax, cmap="cylon")
    cls, cs = _ligo_skymap_contours(ax, skymap, contour=contour, colors=colors)
    if annotate:
        text = []
        pp = np.round(contour).astype(int)
        ii = np.round(
            np.searchsorted(np.sort(cls), contour) * deg2perpix).astype(int)
        for i, p in zip(ii, pp):
            text.append(u'{:d}% area: {:d} deg²'.format(p, i, grouping=True))
        ax.text(1, 1.05, '\n'.join(text), transform=ax.transAxes, ha='right',
                fontsize=10)
    plot.outline_text(ax)
    if injection is not None and len(injection) == 2:
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        _inj = SkyCoord(*injection, unit=u.rad)
        ax.scatter(
            _inj.ra.value, _inj.dec.value, marker="*", color="orange",
            edgecolors='k', linewidth=1.75, s=100, zorder=100,
            transform=ax.get_transform('world')
        )
    return ExistingFigure(fig), ax


def _ligo_skymap_comparion_plot_from_array(
    skymaps, colors, labels, contour=[50, 90], show_probability_map=False,
    injection=None
):
    """Generate a skymap with `ligo.skymap` based which compares arrays of
    probabilities

    Parameters
    ----------
    skymaps: list
        list of skymap probabilities
    colors: list
        list of colors to use for each skymap
    labels: list
        list of labels associated with each skymap
    contour: list, optional
        contours you wish to display on the comparison plot
    show_probability_map: int, optional
        the index of the skymap you wish to show the probability
        map for. Default False
    injection: list, optional
        List containing RA and DEC of the injection. Both must be in radians
    """
    from ligo.skymap import plot

    ncols = number_of_columns_for_legend(labels)
    fig = figure(gca=False)
    ax = fig.add_subplot(111, projection='astro hours mollweide')
    ax.grid(b=True)
    for num, skymap in enumerate(skymaps):
        if isinstance(show_probability_map, int) and show_probability_map == num:
            _, ax = _ligo_skymap_plot_from_array(
                skymap, nsamples=None, downsampled=False, contour=contour,
                annotate=False, ax=ax, colors=colors[num], injection=injection,
            )
        cls, cs = _ligo_skymap_contours(
            ax, skymap, contour=contour, colors=colors[num]
        )
        cs.collections[0].set_label(labels[num])
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.,
              mode="expand", ncol=ncols)
    return fig


def _ligo_skymap_contours(ax, skymap, contour=[50, 90], colors='k'):
    """Plot contours on a ligo.skymap skymap

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot, optional
        Existing axis to add the plot to
    skymap: np.array
        array of probabilities
    contour: list
        list contours you wish to plot
    colors: str/list
        colors to use for the contours
    """
    from ligo.skymap import postprocess

    cls = 100 * postprocess.find_greedy_credible_levels(skymap)
    cs = ax.contour_hpx((cls, 'ICRS'), nested=True, colors=colors,
                        linewidths=0.5, levels=contour)
    ax.clabel(cs, fmt=r'%g\%%', fontsize=6, inline=True)
    return cls, cs


def _default_skymap_plot(ra, dec, weights=None, injection=None, **kwargs):
    """Plot the default sky location of the source for a given approximant

    Parameters
    ----------
    ra: list
        list of samples for right ascension
    dec: list
        list of samples for declination
    injection: list, optional
        list containing the injected value of ra and dec
    kwargs: dict
        optional keyword arguments
    """
    ra = [-i + np.pi for i in ra]
    logger.debug("Generating the sky map plot")
    fig, ax = figure(gca=True)
    ax = fig.add_subplot(
        111, projection="mollweide",
        facecolor=(1.0, 0.939165516411, 0.880255669068)
    )
    ax.cla()
    ax.set_title("Preliminary", fontdict={'fontsize': 11})
    ax.grid(b=True)
    ax.set_xticklabels([
        r"$2^{h}$", r"$4^{h}$", r"$6^{h}$", r"$8^{h}$", r"$10^{h}$",
        r"$12^{h}$", r"$14^{h}$", r"$16^{h}$", r"$18^{h}$", r"$20^{h}$",
        r"$22^{h}$"])
    levels = [0.9, 0.5]

    if weights is None:
        H, X, Y = np.histogram2d(ra, dec, bins=50)
    else:
        H, X, Y = np.histogram2d(ra, dec, bins=50, weights=weights)
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

    ax.pcolormesh(X2, Y2, H2.T, vmin=0., vmax=H2.T.max(), cmap="cylon")
    cs = ax.contour(X2, Y2, H2.T, V, colors="k", linewidths=0.5)
    if injection is not None:
        ax.scatter(
            -injection[0] + np.pi, injection[1], marker="*",
            color=conf.injection_color, edgecolors='k', linewidth=1.75, s=100
        )
    fmt = {l: s for l, s in zip(cs.levels, [r"$90\%$", r"$50\%$"])}
    ax.clabel(cs, fmt=fmt, fontsize=8, inline=True)
    text = []
    for i, j in zip(cs.collections, [90, 50]):
        area = 0.
        for k in i.get_paths():
            x = k.vertices[:, 0]
            y = k.vertices[:, 1]
            area += 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
        area = int(np.abs(area) * (180 / np.pi) * (180 / np.pi))
        text.append(u'{:d}% area: {:d} deg²'.format(
            int(j), area, grouping=True))
    ax.text(1, 1.05, '\n'.join(text[::-1]), transform=ax.transAxes, ha='right',
            fontsize=10)
    xticks = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 4)
    ax.set_xticks(xticks)
    ax.set_yticks([-np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3])
    labels = [r"$%s^{h}$" % (int(np.round((i + np.pi) * 3.82, 1))) for i in xticks]
    ax.set_xticklabels(labels[::-1], fontsize=10)
    ax.set_yticklabels([r"$-60^{\circ}$", r"$-30^{\circ}$", r"$0^{\circ}$",
                        r"$30^{\circ}$", r"$60^{\circ}$"], fontsize=10)
    ax.grid(b=True)
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
    fig = figure(gca=False)
    ax = fig.add_subplot(
        111, projection="mollweide",
        facecolor=(1.0, 0.939165516411, 0.880255669068)
    )
    ax.cla()
    ax.grid(b=True)
    ax.set_xticklabels([
        r"$2^{h}$", r"$4^{h}$", r"$6^{h}$", r"$8^{h}$", r"$10^{h}$",
        r"$12^{h}$", r"$14^{h}$", r"$16^{h}$", r"$18^{h}$", r"$20^{h}$",
        r"$22^{h}$"])
    levels = [0.9, 0.5]
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
        CS = ax.contour(X2, Y2, H2.T, V, colors=colors[num], linewidths=2.0)
        CS.collections[0].set_label(labels[num])
    ncols = number_of_columns_for_legend(labels)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.,
              mode="expand", ncol=ncols)
    xticks = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 4)
    ax.set_xticks(xticks)
    ax.set_yticks([-np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3])
    labels = [r"$%s^{h}$" % (int(np.round((i + np.pi) * 3.82, 1))) for i in xticks]
    ax.set_xticklabels(labels[::-1], fontsize=10)
    ax.set_yticklabels([r"$-60^\degree$", r"$-30^\degree$", r"$0^\degree$",
                        r"$30^\degree$", r"$60^\degree$"], fontsize=10)
    ax.grid(b=True)
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


@no_latex_plot
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
    fig = figure(gca=False)
    ax = fig.add_subplot(111, projection="hammer")
    ax.cla()
    ax.grid(b=True)
    ax.pcolormesh(X, Y, N)
    ax.set_xticklabels([
        r"$22^{h}$", r"$20^{h}$", r"$18^{h}$", r"$16^{h}$", r"$14^{h}$",
        r"$12^{h}$", r"$10^{h}$", r"$8^{h}$", r"$6^{h}$", r"$4^{h}$",
        r"$2^{h}$"])
    return fig


@no_latex_plot
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
    from gwpy.timeseries import TimeSeries
    from gwpy.plot.colors import GW_OBSERVATORY_COLORS
    if math.isnan(maxL_params["mass_1"]):
        return
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

    fig, ax = figure(gca=True)
    colors = [GW_OBSERVATORY_COLORS[i] for i in detectors]
    for num, i in enumerate(detectors):
        ar = __antenna_response(i, maxL_params["ra"], maxL_params["dec"],
                                maxL_params["psi"], maxL_params["geocent_time"])
        h_t = h_plus.data.data * ar[0] + h_cross.data.data * ar[1]
        h_t = TimeSeries(h_t[:], dt=h_plus.deltaT, t0=h_plus.epoch)
        h_t.times = [float(np.array(i)) + t_start for i in h_t.times]
        ax.plot(h_t.times, h_t,
                color=colors[num], linewidth=1.0, label=i)
        ax.set_xlim([t_start - 3, t_start + 0.5])
    ax.set_xlabel(r"Time $[s]$")
    ax.set_ylabel(r"Strain")
    ax.grid(b=True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@no_latex_plot
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
    from gwpy.timeseries import TimeSeries
    logger.debug("Generating the maximum likelihood time domain waveform "
                 "comparison plot for H1")
    if not LALSIMULATION:
        raise Exception("LALSimulation could not be imported. Please install "
                        "LALSuite to be able to use all features")
    delta_t = 1. / 4096.
    minimum_frequency = kwargs.get("f_min", 5.)

    fig, ax = figure(gca=True)
    for num, i in enumerate(maxL_params_list):
        if math.isnan(i["mass_1"]):
            continue
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

        ar = __antenna_response("H1", i["ra"], i["dec"], i["psi"],
                                i["geocent_time"])
        h_t = h_plus.data.data * ar[0] + h_cross.data.data * ar[1]
        h_t = TimeSeries(h_t[:], dt=h_plus.deltaT, t0=h_plus.epoch)
        h_t.times = [float(np.array(i)) + t_start for i in h_t.times]

        ax.plot(h_t.times, h_t,
                color=colors[num], label=labels[num], linewidth=2.0)
    ax.set_xlabel(r"Time $[s]$")
    ax.set_ylabel(r"Strain")
    ax.set_xlim([t_start - 3, t_start + 0.5])
    ax.grid(b=True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _psd_plot(frequencies, strains, colors=None, labels=None, fmin=None):
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
    fmin: optional, float
        starting frequency of the plot
    """
    from gwpy.plot.colors import GW_OBSERVATORY_COLORS
    fig, ax = figure(gca=True)
    if not colors and all(i in GW_OBSERVATORY_COLORS.keys() for i in labels):
        colors = [GW_OBSERVATORY_COLORS[i] for i in labels]
    elif not colors:
        colors = ['r', 'b', 'orange', 'c', 'g', 'purple']
        while len(colors) <= len(labels):
            colors += colors
    for num, i in enumerate(frequencies):
        if fmin is not None:
            ff = np.array(i)
            ss = np.array(strains[num])
            ind = np.argwhere(ff >= fmin)
            i = ff[ind]
            strains[num] = ss[ind]
        ax.loglog(i, strains[num], color=colors[num], label=labels[num])
    ax.tick_params(which="both", bottom=True, length=3, width=1)
    ax.set_xlabel(r"Frequency $[\mathrm{Hz}]$")
    ax.set_ylabel(r"Power Spectral Density [$\mathrm{strain}^{2}/\mathrm{Hz}$]")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@no_latex_plot
def _calibration_envelope_plot(frequency, calibration_envelopes, ifos,
                               colors=None, prior=[]):
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
    prior: list, optional
        list containing the prior calibration envelope data for different IFOs
    """
    from gwpy.plot.colors import GW_OBSERVATORY_COLORS

    def interpolate_calibration(data):
        """Interpolate the calibration data using spline

        Parameters
        ----------
        data: np.ndarray
            array containing the calibration data
        """
        interp = [
            np.interp(frequency, data[:, 0], data[:, j], left=k, right=k)
            for j, k in zip(range(1, 7), [1, 0, 1, 0, 1, 0])
        ]
        amp_median = (interp[0] - 1) * 100
        phase_median = interp[1] * 180. / np.pi
        amp_lower_sigma = (interp[2] - 1) * 100
        phase_lower_sigma = interp[3] * 180. / np.pi
        amp_upper_sigma = (interp[4] - 1) * 100
        phase_upper_sigma = interp[5] * 180. / np.pi
        data_dict = {
            "amplitude": {
                "median": amp_median,
                "lower": amp_lower_sigma,
                "upper": amp_upper_sigma
            },
            "phase": {
                "median": phase_median,
                "lower": phase_lower_sigma,
                "upper": phase_upper_sigma
            }
        }
        return data_dict

    fig, (ax1, ax2) = subplots(2, 1, sharex=True, gca=False)
    if not colors and all(i in GW_OBSERVATORY_COLORS.keys() for i in ifos):
        colors = [GW_OBSERVATORY_COLORS[i] for i in ifos]
    elif not colors:
        colors = ['r', 'b', 'orange', 'c', 'g', 'purple']
        while len(colors) <= len(ifos):
            colors += colors

    for num, i in enumerate(calibration_envelopes):
        calibration_envelopes[num] = np.array(calibration_envelopes[num])

    for num, i in enumerate(calibration_envelopes):
        calibration_data = interpolate_calibration(i)
        if prior != []:
            prior_data = interpolate_calibration(prior[num])
        ax1.plot(
            frequency, calibration_data["amplitude"]["upper"], color=colors[num],
            linestyle="-", label=ifos[num]
        )
        ax1.plot(
            frequency, calibration_data["amplitude"]["lower"], color=colors[num],
            linestyle="-"
        )
        ax1.set_ylabel(r"Amplitude deviation $[\%]$", fontsize=10)
        ax1.legend(loc="best")
        ax2.plot(
            frequency, calibration_data["phase"]["upper"], color=colors[num],
            linestyle="-", label=ifos[num]
        )
        ax2.plot(
            frequency, calibration_data["phase"]["lower"], color=colors[num],
            linestyle="-"
        )
        ax2.set_ylabel(r"Phase deviation $[\degree]$", fontsize=10)
        if prior != []:
            ax1.fill_between(
                frequency, prior_data["amplitude"]["upper"],
                prior_data["amplitude"]["lower"], color=colors[num], alpha=0.2
            )
            ax2.fill_between(
                frequency, prior_data["phase"]["upper"],
                prior_data["phase"]["lower"], color=colors[num], alpha=0.2
            )

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Frequency $[Hz]$")
    fig.tight_layout()
    return fig


def _strain_plot(strain, maxL_params, **kwargs):
    """Generate a plot showing the strain data and the maxL waveform

    Parameters
    ----------
    strain: gwpy.timeseries
        timeseries containing the strain data
    maxL_samples: dict
        dictionary of maximum likelihood parameter values
    """
    logger.debug("Generating the strain plot")
    from pesummary.gw.conversions import time_in_each_ifo
    from gwpy.timeseries import TimeSeries

    fig, axs = subplots(nrows=len(strain.keys()), sharex=True)
    time = maxL_params["geocent_time"]
    delta_t = 1. / 4096.
    minimum_frequency = kwargs.get("f_min", 5.)
    t_start = time - 15.0
    t_finish = time + 0.06
    time_array = np.arange(t_start, t_finish, delta_t)

    approx = lalsim.GetApproximantFromString(maxL_params["approximant"])
    mass_1 = maxL_params["mass_1"] * MSUN_SI
    mass_2 = maxL_params["mass_2"] * MSUN_SI
    luminosity_distance = maxL_params["luminosity_distance"] * PC_SI * 10**6
    phase = maxL_params["phase"] if "phase" in maxL_params.keys() else 0.0
    cartesian = [
        "iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"
    ]
    if not all(param in maxL_params.keys() for param in cartesian):
        if "phi_jl" in maxL_params.keys():
            iota, S1x, S1y, S1z, S2x, S2y, S2z = \
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    maxL_params["theta_jn"], maxL_params["phi_jl"],
                    maxL_params["tilt_1"], maxL_params["tilt_2"],
                    maxL_params["phi_12"], maxL_params["a_1"],
                    maxL_params["a_2"], mass_1, mass_2, kwargs.get("f_ref", 10.),
                    phase
                )
        else:
            iota, S1x, S1y, S1z, S2x, S2y, S2z = maxL_params["iota"], 0., 0., \
                0., 0., 0., 0.
    else:
        iota, S1x, S1y, S1z, S2x, S2y, S2z = [
            maxL_params[param] for param in cartesian
        ]
    h_plus, h_cross = lalsim.SimInspiralChooseTDWaveform(
        mass_1, mass_2, S1x, S1y, S1z, S2x, S2y, S2z, luminosity_distance, iota,
        phase, 0.0, 0.0, 0.0, delta_t, minimum_frequency,
        kwargs.get("f_ref", 10.), None, approx)
    h_plus = TimeSeries(
        h_plus.data.data[:], dt=h_plus.deltaT, t0=h_plus.epoch
    )
    h_cross = TimeSeries(
        h_cross.data.data[:], dt=h_cross.deltaT, t0=h_cross.epoch
    )

    for num, key in enumerate(list(strain.keys())):
        ifo_time = time_in_each_ifo(key, maxL_params["ra"], maxL_params["dec"],
                                    maxL_params["geocent_time"])

        asd = strain[key].asd(8, 4, method="median")
        strain_data_frequency = strain[key].fft()
        asd_interp = asd.interpolate(float(np.array(strain_data_frequency.df)))
        asd_interp = asd_interp[:len(strain_data_frequency)]
        strain_data_time = (strain_data_frequency / asd_interp).ifft()
        strain_data_time = strain_data_time.highpass(30)
        strain_data_time = strain_data_time.lowpass(300)

        ar = __antenna_response(key, maxL_params["ra"], maxL_params["dec"],
                                maxL_params["psi"], maxL_params["geocent_time"])

        h_t = ar[0] * h_plus + ar[1] * h_cross
        h_t_frequency = h_t.fft()
        asd_interp = asd.interpolate(float(np.array(h_t_frequency.df)))
        asd_interp = asd_interp[:len(h_t_frequency)]
        h_t_time = (h_t_frequency / asd_interp).ifft()
        h_t_time = h_t_time.highpass(30)
        h_t_time = h_t_time.lowpass(300)
        h_t_time.times = [float(np.array(i)) + ifo_time for i in h_t.times]

        strain_data_crop = strain_data_time.crop(ifo_time - 0.2, ifo_time + 0.06)
        try:
            h_t_time = h_t_time.crop(ifo_time - 0.2, ifo_time + 0.06)
        except Exception:
            pass
        max_strain = np.max(strain_data_crop).value

        axs[num].plot(strain_data_crop, color='grey', alpha=0.75, label="data")
        axs[num].plot(h_t_time, color='orange', label="template")
        axs[num].set_xlim([ifo_time - 0.2, ifo_time + 0.06])
        if not math.isnan(max_strain):
            axs[num].set_ylim([-max_strain * 1.5, max_strain * 1.5])
        axs[num].set_ylabel("Whitened %s strain" % (key), fontsize=8)
        axs[num].grid(False)
        axs[num].legend(loc="best", prop={'size': 8})
    axs[-1].set_xlabel("Time $[s]$", fontsize=16)
    fig.tight_layout()
    return fig


def _format_prob(prob):
    """Format the probabilities for use with _classification_plot
    """
    if prob >= 1:
        return '100%'
    elif prob <= 0:
        return '0%'
    elif prob > 0.99:
        return '>99%'
    elif prob < 0.01:
        return '<1%'
    else:
        return '{}%'.format(int(np.round(100 * prob)))


@no_latex_plot
def _classification_plot(classification):
    """Generate a bar chart showing the source classifications probabilities

    Parameters
    ----------
    classification: dict
        dictionary of source classifications
    """
    from matplotlib import rcParams

    original_fontsize = rcParams["font.size"]
    original_ylabel = rcParams["ytick.labelsize"]
    rcParams["font.size"] = 12
    rcParams["ytick.labelsize"] = 12
    probs, names = zip(
        *sorted(zip(classification.values(), classification.keys())))
    with matplotlib.style.context('seaborn-white'):
        fig, ax = figure(figsize=(2.5, 2), gca=True)
        ax.barh(names, probs)
        for i, prob in enumerate(probs):
            ax.annotate(_format_prob(prob), (0, i), (4, 0),
                        textcoords='offset points', ha='left', va='center')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.tick_params(left=False)
        for side in ['top', 'bottom', 'right']:
            ax.spines[side].set_visible(False)
        fig.tight_layout()
    rcParams["font.size"] = original_fontsize
    rcParams["ytick.labelsize"] = original_ylabel
    return fig
