# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.plots.bounded_2d_kde import Bounded_2d_kde
from pesummary.utils.utils import logger
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp2d
import multiprocessing
from pesummary.utils.probability_dict import ProbabilityDict2D

__author__ = [
    "Aditya Vijaykumar <aditya.vijaykumar@ligo.org>",
    "Charlie Hoy <charlie.hoy@ligo.org>"
]


def _wrapper_for_multiprocessing_kde(kde, *args):
    """Wrapper to evaluate a KDE on multiple cpus

    Parameters
    ----------
    kde: func
        KDE you wish to evaluate
    *args: tuple
        all args are passed to the KDE
    """
    _reshape = (len(args[0]), len(args[1]))
    yy, xx = np.meshgrid(args[1], args[0])
    _args = [xx.ravel(), yy.ravel()]
    return kde(_args).reshape(_reshape)


def _wrapper_for_multiprocessing_interp(interp, *args):
    """Wrapper to evaluate an interpolant on multiple cpus

    Parameters
    ----------
    interp: func
        interpolant you wish to use
    *args: tuple
        all args are passed to the interpolant
    """
    return interp(*args)


def _imrct_deviation_parameters_integrand_vectorized(
    final_mass,
    final_spin,
    v1,
    v2,
    P_final_mass_final_spin_i_interp_object,
    P_final_mass_final_spin_r_interp_object,
    multi_process=4,
    wrapper_function_for_multiprocess=None,
):
    """Compute the integrand of P(delta_final_mass/final_mass_bar,
    delta_final_spin/final_spin_bar).

    Parameters
    ----------
    final_mass: np.ndarray
        vector of values of final mass
    final_spin: np.ndarray
        vector of values of final spin
    v1: np.ndarray
        array of delta_final_mass/final_mass_bar values
    v2: np.ndarray
        array of delta_final_spin/final_spin_bar values
    P_final_mass_final_spin_i_interp_object:
        interpolated P_i(final_mass, final_spin)
    P_final_mass_final_spin_r_interp_object:
        interpolated P_r(final_mass, final_spin)
    multi_process: int
        Number of parallel processes. Default: 4
    wrapper_function_for_multiprocess: method
        Wrapper function for the multiprocessing. Default: None

    Returns
    -------
    np.array
        integrand of P(delta_final_mass/final_mass_bar,
        delta_final_spin/final_spin_bar)
    """
    final_mass_mat, final_spin_mat = np.meshgrid(final_mass, final_spin)
    _abs = np.abs(final_mass_mat * final_spin_mat)
    _reshape = (len(v1), len(v1))
    _v2, _v1 = np.meshgrid(v2, v1)
    v1, v2 = _v1.ravel(), _v2.ravel()
    v1, v2 = v1.reshape(len(v1), 1), v2.reshape(len(v2), 1)

    # Create delta_final_mass and delta_final_spin vectors corresponding
    # to the given v1 and v2.

    # These vectors have to be monotonically increasing in order to
    # evaluate the interpolated prob densities. Hence, for v1, v2 < 0,
    # flip them, evaluate the prob density (in column or row) and flip it back

    # The definition of the delta_* parameters is taken from eq A1 of
    # Ghosh et al 2018, arXiv:1704.06784.

    delta_final_mass_i = ((1.0 + v1 / 2.0)) * final_mass
    delta_final_spin_i = ((1.0 + v2 / 2.0)) * final_spin
    delta_final_mass_r = ((1.0 - v1 / 2.0)) * final_mass
    delta_final_spin_r = ((1.0 - v2 / 2.0)) * final_spin

    for num in range(len(v1)):
        if (1.0 + v1[num] / 2.0) < 0.0:
            delta_final_mass_i[num] = np.flipud(delta_final_mass_i[num])
        if (1.0 + v2[num] / 2.0) < 0.0:
            delta_final_spin_i[num] = np.flipud(delta_final_spin_i[num])
        if (1.0 - v1[num] / 2.0) < 0.0:
            delta_final_mass_r[num] = np.flipud(delta_final_mass_r[num])
        if (1.0 - v2[num] / 2.0) < 0.0:
            delta_final_spin_r[num] = np.flipud(delta_final_spin_r[num])

    if multi_process > 1:
        with multiprocessing.Pool(multi_process) as pool:
            _length = len(delta_final_mass_i)
            _P_i = pool.starmap(
                wrapper_function_for_multiprocess,
                zip(
                    [P_final_mass_final_spin_i_interp_object] * _length,
                    delta_final_mass_i,
                    delta_final_spin_i,
                ),
            )
            _P_r = pool.starmap(
                wrapper_function_for_multiprocess,
                zip(
                    [P_final_mass_final_spin_r_interp_object] * _length,
                    delta_final_mass_r,
                    delta_final_spin_r,
                ),
            )
        P_i = np.array([i for i in _P_i])
        P_r = np.array([i for i in _P_r])
    else:
        P_i, P_r = [], []
        for num in range(len(delta_final_mass_i)):
            P_i.append(
                wrapper_function_for_multiprocess(
                    P_final_mass_final_spin_i_interp_object,
                    delta_final_mass_i[num],
                    delta_final_spin_i[num],
                )
            )
            P_r.append(
                wrapper_function_for_multiprocess(
                    P_final_mass_final_spin_r_interp_object,
                    delta_final_mass_r[num],
                    delta_final_spin_r[num],
                )
            )

    for num in range(len(v1)):
        if (1.0 + v1[num] / 2.0) < 0.0:
            P_i[num] = np.fliplr(P_i[num])
        if (1.0 + v2[num] / 2.0) < 0.0:
            P_i[num] = np.flipud(P_i[num])
        if (1.0 - v1[num] / 2.0) < 0.0:
            P_r[num] = np.fliplr(P_r[num])
        if (1.0 - v2[num] / 2.0) < 0.0:
            P_r[num] = np.flipud(P_r[num])

    # The integration is performed according to eq A2 of Ghosh et al,
    # arXiv:1704.06784

    _prod = np.array(
        [np.sum(_P_i * _P_r * _abs) for _P_i, _P_r in zip(P_i, P_r)]
    ).reshape(_reshape)
    return _prod


def _apply_args_and_kwargs(function, args, kwargs):
    """Apply a tuple of args and a dictionary of kwargs to a function

    Parameters
    ----------
    function: func
        function you wish to use
    args: tuple
        all args passed to function
    kwargs: dict
        all kwargs passed to function
    """
    return function(*args, **kwargs)


def _imrct_deviation_parameters_integrand_series(
    final_mass,
    final_spin,
    v1,
    v2,
    P_final_mass_final_spin_i_interp_object,
    P_final_mass_final_spin_r_interp_object,
    multi_process=4,
    **kwargs
):
    """
    Creates the over the deviation parameter space.

    Parameters
    ----------
    final_mass: np.ndarray
        vector of values of final mass
    final_spin: np.ndarray
        vector of values of final spin
    v1: np.ndarray
        array of delta_final_mass/final_mass_bar values
    v2: np.ndarray
        array of delta_final_spin/final_spin_bar values
    P_final_mass_final_spin_i_interp_object:
        interpolated P_i(final_mass, final_spin)
    P_final_massfinal_spin_r_interp_object:
        interpolated P_r(final_mass, final_spin)
    """
    P = np.zeros(shape=(len(v1), len(v2)))
    if multi_process == 1:
        logger.debug("Performing calculation on a single cpu. This may take some time")
        for i, _v2 in enumerate(v2):
            for j, _v1 in enumerate(v1):
                P[i, j] = _imrct_deviation_parameters_integrand_vectorized(
                    final_mass,
                    final_spin,
                    [_v1],
                    [_v2],
                    P_final_mass_final_spin_i_interp_object,
                    P_final_mass_final_spin_r_interp_object,
                    multi_process=1,
                    **kwargs
                )
    else:
        logger.debug("Splitting the calculation across {} cpus".format(multi_process))
        _v1, _v2 = np.meshgrid(v1, v2)
        _v1, _v2 = _v1.ravel(), _v2.ravel()
        with multiprocessing.Pool(multi_process) as pool:
            args = [
                [final_mass] * len(_v1),
                [final_spin] * len(_v1),
                np.atleast_2d(_v1).T.tolist(),
                np.atleast_2d(_v2).T.tolist(),
                [P_final_mass_final_spin_i_interp_object] * len(_v1),
                [P_final_mass_final_spin_r_interp_object] * len(_v1),
            ]
            kwargs["multi_process"] = 1
            _args = np.array(args, dtype=object).T
            _P = pool.starmap(
                _apply_args_and_kwargs,
                zip(
                    [_imrct_deviation_parameters_integrand_vectorized] * len(_args),
                    _args,
                    [kwargs] * len(_args),
                ),
            )
        P = np.array([i[0] for i in _P]).reshape(len(v1), len(v2))
    return P


def imrct_deviation_parameters_integrand(*args, vectorize=False, **kwargs):
    """Compute the final mass and final spin deviation parameters

    Parameters
    ----------
    *args: tuple
        all args passed to either
        _imrct_deviation_parameters_integrand_vectorized or
        _imrct_deviation_parameters_integrand_series
    vectorize: bool
        Vectorize the calculation. Note that vectorize=True uses up a lot
        of memory
    kwargs: dict, optional
        all kwargs passed to either
        _imrct_deviation_parameters_integrand_vectorized or
        _imrct_deviation_parameters_integrand_series
    """
    if vectorize:
        return _imrct_deviation_parameters_integrand_vectorized(*args, **kwargs)
    return _imrct_deviation_parameters_integrand_series(*args, **kwargs)


def imrct_deviation_parameters_from_final_mass_final_spin(
    final_mass_inspiral,
    final_spin_inspiral,
    final_mass_postinspiral,
    final_spin_postinspiral,
    N_bins=101,
    final_mass_deviation_lim=1,
    final_spin_deviation_lim=1,
    multi_process=4,
    use_kde=False,
    kde=gaussian_kde,
    kde_kwargs=dict(),
    interp_method=interp2d,
    interp_kwargs=dict(fill_value=0.0, bounds_error=False),
    vectorize=False,
):
    """Compute the IMR Consistency Test deviation parameters.
    Code borrows from the implementation in lalsuite:
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/python/lalinference/imrtgr/imrtgrutils.py
    and
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/bin/imrtgr_imr_consistency_test.py

    Parameters
    ----------
    final_mass_inspiral: np.ndarray
        values of final mass calculated from the inspiral part
    final_spin_inspiral: np.ndarray
        values of final spin calculated from the inspiral part
    final_mass_postinspiral: np.ndarray
        values of final mass calculated from the post-inspiral part
    final_spin_postinspiral: np.ndarray
        values of final spin calculated from the post-inspiral part
    final_mass_deviation_lim: float
        Maximum absolute value of the final mass deviation parameter. Default 1.
    final_spin_deviation_lim: float
        Maximum absolute value of the final spin deviation parameter. Default 1.
    N_bins: int, optional
        Number of equally spaced bins between [-final_mass_deviation_lim,
        final_mass_deviation_lim] and [-final_spin_deviation_lim,
        final_spin_deviation_lim]. Default is 101.
    multi_process: int
        Number of parallel processes. Default is 4.
    use_kde: bool
        If `True`, uses kde instead of interpolation. Default is False.
    kde: method
        KDE method to use. Default is scipy.stats.gaussian_kde
    kde_kwargs: dict
        Arguments to be passed to the KDE method
    interp_method: method
        Interpolation method to use. Default is scipy.interpolate.interp2d
    interp_kwargs: dict, optional
        Arguments to be passed to the interpolation method
        Default is `dict(fill_value=0.0, bounds_error=False)`
    vectorize: bool
        if True, use vectorized imrct_deviation_parameters_integrand
        function. This is quicker but does consume more memory. Default: False

    Returns
    -------
    imrct_deviations: ProbabilityDict2d
        Contains the 2d pdf of the IMRCT deviation parameters
    """
    # Find the maximum values
    final_mass_lim = np.max(np.append(final_mass_inspiral, final_mass_postinspiral))
    final_spin_lim = np.max(
        np.abs(np.append(final_spin_inspiral, final_spin_postinspiral))
    )

    # bin the data
    final_mass_bins = np.linspace(-final_mass_lim, final_mass_lim, N_bins)
    diff_final_mass = final_mass_bins[1] - final_mass_bins[0]
    final_spin_bins = np.linspace(-final_spin_lim, final_spin_lim, N_bins)
    diff_final_spin = final_spin_bins[1] - final_spin_bins[0]
    final_mass_intp = (final_mass_bins[:-1] + final_mass_bins[1:]) * 0.5
    final_spin_intp = (final_spin_bins[:-1] + final_spin_bins[1:]) * 0.5
    if use_kde:
        logger.debug("Using KDE to interpolate data")
        # kde the samples for final mass and final spin
        final_mass_intp = np.append(
            final_mass_intp, final_mass_bins[-1] + diff_final_mass
        )
        final_spin_intp = np.append(
            final_spin_intp, final_spin_bins[-1] + diff_final_spin
        )
        inspiral_interp = kde(
            np.array([final_mass_inspiral, final_spin_inspiral]), **kde_kwargs
        )
        postinspiral_interp = kde(
            np.array([final_mass_postinspiral, final_spin_postinspiral]), **kde_kwargs
        )
        _wrapper_function = _wrapper_for_multiprocessing_kde
    else:
        logger.debug("Interpolating 2d histogram data")
        _inspiral_2d_histogram, _, _ = np.histogram2d(
            final_mass_inspiral,
            final_spin_inspiral,
            bins=(final_mass_bins, final_spin_bins),
            density=True,
        )
        _postinspiral_2d_histogram, _, _ = np.histogram2d(
            final_mass_postinspiral,
            final_spin_postinspiral,
            bins=(final_mass_bins, final_spin_bins),
            density=True,
        )
        # transpose density to go from (X,Y) indexing returned by
        # np.histogram2d() to array (i,j) indexing for further computations.
        # From now onwards, different rows (i) correspond to different values
        # of final mass and different columns (j) correspond to different
        # values of final_spin
        _inspiral_2d_histogram = _inspiral_2d_histogram.T
        _postinspiral_2d_histogram = _postinspiral_2d_histogram.T
        inspiral_interp = interp_method(
            final_mass_intp, final_spin_intp, _inspiral_2d_histogram, **interp_kwargs
        )
        postinspiral_interp = interp_method(
            final_mass_intp, final_spin_intp, _postinspiral_2d_histogram, **interp_kwargs
        )
        _wrapper_function = _wrapper_for_multiprocessing_interp

    final_mass_deviation_vec = np.linspace(
        -final_mass_deviation_lim, final_mass_deviation_lim, N_bins
    )
    final_spin_deviation_vec = np.linspace(
        -final_spin_deviation_lim, final_spin_deviation_lim, N_bins
    )

    diff_final_mass_deviation = final_mass_deviation_vec[1] - final_mass_deviation_vec[0]
    diff_final_spin_deviation = final_spin_deviation_vec[1] - final_spin_deviation_vec[0]

    P_final_mass_deviation_final_spin_deviation = imrct_deviation_parameters_integrand(
        final_mass_intp,
        final_spin_intp,
        final_mass_deviation_vec,
        final_spin_deviation_vec,
        inspiral_interp,
        postinspiral_interp,
        multi_process=multi_process,
        vectorize=vectorize,
        wrapper_function_for_multiprocess=_wrapper_function,
    )

    imrct_deviations = ProbabilityDict2D(
        {
            "final_mass_final_spin_deviations": [
                final_mass_deviation_vec,
                final_spin_deviation_vec,
                P_final_mass_deviation_final_spin_deviation
                / np.sum(P_final_mass_deviation_final_spin_deviation),
            ]
        }
    )
    return imrct_deviations


def generate_imrct_deviation_parameters(
    samples, evolve_spins_forward=True, inspiral_string="inspiral",
    postinspiral_string="postinspiral", approximant=None, f_low=None,
    return_samples_used=False, **kwargs
):
    """Generate deviation parameter pdfs for the IMR Consistency Test

    Parameters
    ----------
    samples: MultiAnalysisSamplesDict
        Dictionary containing inspiral and postinspiral samples
    evolve_spins_forward: bool
        If `True`, evolve spins to the ISCO frequency. Default: True.
    inspiral_string: string
        Identifier for the inspiral samples
    postinspiral_string: string
        Identifier for the post-inspiral samples
    approximant: dict, optional
        The approximant used for the inspiral and postinspiral analyses.
        Keys of the dictionary must be the same as the inspiral_string and
        postinspiral_string. Default None
    f_low: dict, optional
        The low frequency cut-off used for the inspiral and postinspiral
        analyses. Keys of the dictionary must be the same as the inspiral_string
        and postinspiral_string. Default None
    return_samples_used: Bool, optional
        if True, return the samples which were used to generate the IMRCT deviation
        parameters. These samples will match the input but may include remnant
        samples if they were not previously present
    kwargs: dict, optional
        Keywords to be passed to imrct_deviation_parameters_from_final_mass_final_spin

    Returns
    -------
    imrct_deviations: ProbabilityDict2d
        2d pdf of the IMRCT deviation parameters
    data: dict
        Metadata
    """
    import time

    remnant_condition = lambda _dictionary, _suffix: all(
        "{}{}".format(param, _suffix) not in _dictionary.keys() for
        param in ["final_mass", "final_spin"]
    )
    evolved = np.ones(2, dtype=bool)
    suffix = [""]
    evolve_spins = ["ISCO"]
    if not evolve_spins_forward:
        suffix = ["_non_evolved"] + suffix
        evolve_spins = [False, False]
    fits_data = {}
    for idx, (key, sample) in enumerate(samples.items()):
        zipped = zip(suffix, evolve_spins)
        for num, (_suffix, _evolve_spins) in enumerate(zipped):
            cond = remnant_condition(sample, _suffix)
            _found_msg = (
                "Found {} remnant properties in the posterior table  "
                "for {}. Using these for calculation."
            )
            if not cond:
                logger.info(
                    _found_msg.format(
                        "evolved" if not len(_suffix) else "non-evolved",
                        key
                    )
                )
                if len(_suffix):
                    evolved[idx] = False
                break
            elif not remnant_condition(sample, ""):
                logger.info(_found_msg.format("evolved", key))
                evolved[idx] = True
                break
            else:
                logger.warning(
                    "{} remnant properties not found in the posterior "
                    "table for {}. Trying to calculate them.".format(
                        "Evolved" if not len(_suffix) else "Non-evolved",
                        key
                    )
                )
                returned_extra_kwargs = sample.generate_all_posterior_samples(
                    evolve_spins=_evolve_spins, return_kwargs=True,
                    approximant=(
                        approximant[key] if approximant is not None else None
                    ), f_low=f_low[key] if f_low is not None else None
                )
                _cond = remnant_condition(sample, _suffix)
                if not _cond:
                    logger.info(
                        "{} remnant properties generated. Using these "
                        "samples for calculation".format(
                            "Evolved" if not len(_suffix) else "Non-evolved"
                        )
                    )
                    for fit in ["final_mass_NR_fits", "final_spin_NR_fits"]:
                        fits_data["{} {}".format(key, fit)] = returned_extra_kwargs[
                            "meta_data"
                        ][fit]
                    if len(_suffix):
                        evolved[idx] = False
                    break

            if num == 1:
                raise ValueError(
                    "Unable to compute the remnant properties"
                )
    if not all(_evolved == evolved[0] for _evolved in evolved):
        keys = list(samples.keys())
        _inspiral_index = keys.index(inspiral_string)
        _postinspiral_index = keys.index(postinspiral_string)
        raise ValueError(
            "Using {} remnant properties for the inspiral and {} remnant "
            "properties for the postinspiral. This must be the same for "
            "the calculation".format(
                "evolved" if evolved[_inspiral_index] else "non-evolved",
                "non-evolved" if not evolved[_postinspiral_index] else "evolved"
            )
        )
    samples_string = "final_{}"
    if not evolved[0]:
        samples_string += "_non_evolved"

    logger.info("Calculating IMRCT deviation parameters and GR Quantile")
    t0 = time.time()
    imrct_deviations = imrct_deviation_parameters_from_final_mass_final_spin(
        samples[inspiral_string][samples_string.format("mass")],
        samples[inspiral_string][samples_string.format("spin")],
        samples[postinspiral_string][samples_string.format("mass")],
        samples[postinspiral_string][samples_string.format("spin")],
        **kwargs,
    )
    gr_quantile = (
        imrct_deviations[
            "final_mass_final_spin_deviations"
        ].minimum_encompassing_contour_level(0.0, 0.0)
        * 100
    )
    t1 = time.time()
    data = kwargs.copy()
    data["evolve_spins"] = evolved
    data["Time (seconds)"] = round(t1 - t0, 2)
    data["GR Quantile (%)"] = gr_quantile[0]
    data.update(fits_data)
    logger.info(
        "Calculation Finished in {} seconds. GR Quantile is {} %.".format(
            data["Time (seconds)"], round(data["GR Quantile (%)"], 2)
        )
    )
    if return_samples_used:
        return imrct_deviations, data, evolved[0], samples
    return imrct_deviations, data, evolved[0]
