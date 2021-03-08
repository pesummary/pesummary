# Copyright (C) 2020  Aditya Vijaykumar <aditya.vijaykumar@ligo.org>
#                     Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from pesummary.core.plots.bounded_2d_kde import Bounded_2d_kde
from pesummary.utils.utils import logger
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp2d
import multiprocessing
from pesummary.utils.probability_dict import ProbabilityDict2D


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

    _prod = np.array(
        [np.sum(_P_i * _P_r * _abs) for _P_i, _P_r in zip(P_i, P_r)]
    ).reshape(_reshape)
    return _prod, P_i, P_r


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
    P = np.zeros(shape=(len(final_mass), len(final_mass)))
    if multi_process == 1:
        logger.debug("Performing calculation on a single cpu. This may take some " "time")
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
        _v2, _v1 = np.meshgrid(v2, v1)
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
        P = np.array([i[0][0] for i in _P]).reshape(len(final_mass), len(final_spin))
    return P, None, None


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
    final_mass_deviation_lim=1,
    final_spin_deviation_lim=1,
    N_bins=101,
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
    N_bins: int
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
    final_spin_lim = np.max(np.abs(np.append(final_spin_inspiral, final_spin_postinspiral)))

    # bin the data
    final_mass_bins = np.linspace(-final_mass_lim, final_mass_lim, N_bins)
    final_mass_df = final_mass_bins[1] - final_mass_bins[0]
    final_spin_bins = np.linspace(-final_spin_lim, final_spin_lim, N_bins)
    final_spin_df = final_spin_bins[1] - final_spin_bins[0]
    final_mass_intp = (final_mass_bins[:-1] + final_mass_bins[1:]) * 0.5
    final_spin_intp = (final_spin_bins[:-1] + final_spin_bins[1:]) * 0.5
    if use_kde:
        logger.debug("Using KDE to interpolate data")
        final_mass_intp = np.append(final_mass_intp, final_mass_bins[-1] + final_mass_df)
        final_spin_intp = np.append(final_spin_intp, final_spin_bins[-1] + final_spin_df)
        inspiral_interp = kde(
            np.array([final_mass_inspiral, final_spin_inspiral]), **kde_kwargs
        )
        postinspiral_interp = kde(
            np.array([final_mass_postinspiral, final_spin_postinspiral]), **kde_kwargs
        )
        final_mass_deviation_vec = np.linspace(
            -final_mass_deviation_lim, final_mass_deviation_lim, N_bins
        )
        final_spin_deviation_vec = np.linspace(
            -final_spin_deviation_lim, final_spin_deviation_lim, N_bins
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
        inspiral_interp = interp_method(
            final_mass_intp, final_spin_intp, _inspiral_2d_histogram, **interp_kwargs
        )
        postinspiral_interp = interp_method(
            final_mass_intp, final_spin_intp, _postinspiral_2d_histogram, **interp_kwargs
        )

        final_mass_deviation_vec = np.linspace(
            -final_mass_deviation_lim, final_mass_deviation_lim, N_bins - 1
        )
        final_spin_deviation_vec = np.linspace(
            -final_spin_deviation_lim, final_spin_deviation_lim, N_bins - 1
        )
        _wrapper_function = _wrapper_for_multiprocessing_interp

    diff_final_mass_deviation = np.mean(np.diff(final_mass_deviation_vec))
    diff_final_spin_deviation = np.mean(np.diff(final_spin_deviation_vec))

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
    )[0]

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
