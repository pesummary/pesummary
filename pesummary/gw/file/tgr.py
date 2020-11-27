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
import numpy as np
from scipy.stats import gaussian_kde
import multiprocessing


def _wrapper_for_multiprocessing_kde(kde, *args):
    """Wrapper to evaluate a KDE on multiple cpus

    Parameters
    ----------
    kde: func
        KDE you wish to evaluate
    *args: tuple
        all args are passed to the KDE
    """
    return kde([*args])


def imrct_deviation_parameters_integrand(
    final_mass,
    final_spin,
    v1,
    v2,
    P_final_mass_final_spin_i_interp_object,
    P_final_mass_final_spin_r_interp_object,
    multi_process=4,
):
    """Compute the integrand of P(delta_final_mass/final_mass_bar,
    delta_final_spin/final_spin_bar).

    Parameters
    ----------
    final_mass: np.ndarray
        samples drawn from the final mass posterior distribution
    final_spin: np.ndarray
        samples drawn from the final spin posterior distribution
    v1: np.ndarray
        array of delta_final_mass/final_mass_bar values
    v2: np.ndarray
        array of delta_final_spin/final_spin_bar values
    P_final_mass_final_spin_i_interp_object:
        interpolated P_i(final_mass, final_spin)
    P_final_massfinal_spin_r_interp_object:
        interpolated P_r(final_mass, final_spin)

    Returns
    -------
    np.array
        integrand of P(delta_final_mass/final_mass_bar,
        delta_final_spin/final_spin_bar)
    """
    # total runtime ~55s
    final_mass_mat, final_spin_mat = np.meshgrid(final_mass, final_spin)
    _abs = np.abs(final_mass_mat * final_spin_mat)
    _v1, _v2 = np.meshgrid(v1, v2)
    v1, v2 = _v1.ravel(), _v2.ravel()
    v1, v2 = v1.reshape(len(v1), 1), v2.reshape(len(v2), 1)

    delta_final_mass_i = np.abs((1.0 + v1 / 2.0)) * final_mass
    delta_final_spin_i = np.abs((1.0 + v2 / 2.0)) * final_spin
    delta_final_mass_r = np.abs((1.0 - v1 / 2.0)) * final_mass
    delta_final_spin_r = np.abs((1.0 - v2 / 2.0)) * final_spin

    # Evaluating the KDE for P_i and P_r takes ~50s
    split_mass_i = np.array_split(delta_final_mass_i.flatten(), multi_process)
    split_mass_r = np.array_split(delta_final_mass_r.flatten(), multi_process)
    split_spin_i = np.array_split(delta_final_spin_i.flatten(), multi_process)
    split_spin_r = np.array_split(delta_final_spin_r.flatten(), multi_process)
    _args_i, _args_r = [], []
    for num in range(multi_process):
        _args_i.append(
            [
                P_final_mass_final_spin_i_interp_object,
                split_mass_i[num],
                split_spin_i[num],
            ]
        )
        _args_r.append(
            [
                P_final_mass_final_spin_r_interp_object,
                split_mass_r[num],
                split_spin_r[num],
            ]
        )
    with multiprocessing.Pool(multi_process) as pool:
        _P_i = pool.starmap(_wrapper_for_multiprocessing_kde, _args_i)
    P_i = np.array(np.abs([item for sublist in _P_i for item in sublist])).reshape(
        delta_final_mass_i.shape
    )
    with multiprocessing.Pool(multi_process) as pool:
        _P_r = pool.starmap(_wrapper_for_multiprocessing_kde, _args_r)
    P_r = np.array(np.abs([item for sublist in _P_r for item in sublist])).reshape(
        delta_final_mass_r.shape
    )
    _prod = np.sum(np.dot(P_i * P_r, _abs.T), axis=1).reshape(
        len(final_mass), len(final_mass)
    )
    return _prod, P_i, P_r


def imrct_deviation_parameters_from_final_mass_final_spin(
    final_mass_inspiral,
    final_spin_inspiral,
    final_mass_postinspiral,
    final_spin_postinspiral,
    final_mass_deviation_lim=2,
    final_spin_deviation_lim=1,
    N_bins=401,
    multi_process=4,
    kde=gaussian_kde,
    kde_kwargs={},
):
    """Compute the IMR Consistency Test deviation parameters

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
    final_mass_deviation_lim: float, optional
        Maximum value of the final mass deviation parameter. Default 2.
    final_spin_deviation_lim: float, optional
        Maximum value of the final spin deviation parameter. Default 1.
    N_bins: int, optional
        Number of equally spaced bins between [-final_mass_deviation_lim,
        final_mass_deviation_lim] and [-final_spin_deviation_lim,
        final_spin_deviation_lim]

    Returns
    -------
    fill this in later
    """
    # Find the maximum values
    final_mass_lim = np.max(np.append([final_mass_inspiral, final_mass_postinspiral]))
    final_spin_lim = np.max(np.append([final_spin_inspiral, final_spin_postinspiral]))

    # bin the data
    final_mass_bins = np.linspace(-final_mass_lim, final_mass_lim, N_bins)
    final_mass_df = final_mass_bins[1] - final_mass_bins[0]
    final_spin_bins = np.linspace(-final_spin_lim, final_spin_lim, N_bins)
    final_spin_df = final_spin_bins[1] - final_spin_bins[0]
    final_mass_intp = np.append(
        (final_mass_bins[:-1] + final_mass_bins[1:]) / 2.0,
        final_mass_bins[-1] + final_mass_df,
    )
    final_spin_intp = np.append(
        (final_spin_bins[:-1] + final_spin_bins[1:]) / 2.0,
        final_spin_bins[-1] + final_spin_df,
    )

    # kde the samples for final mass and final spin
    inspiral_kde = kde(
        np.array([final_mass_inspiral, final_spin_inspiral]), **kde_kwargs
    )
    postinspiral_kde = kde(
        np.array([final_mass_postinspiral, final_spin_postinspiral]), **kde_kwargs
    )

    final_mass_deviation_vec = np.linspace(
        -final_mass_deviation_lim, final_mass_deviation_lim, N_bins
    )
    final_spin_deviation_vec = np.linspace(
        -final_spin_deviation_lim, final_spin_deviation_lim, N_bins
    )

    diff_final_mass_deviation = np.mean(np.diff(final_mass_deviation_vec))
    diff_final_spin_deviation = np.mean(np.diff(final_spin_deviation_vec))

    P_final_mass_deviation_final_spin_deviation = imrct_deviation_parameters_integrand(
        final_mass_intp,
        final_spin_intp,
        final_mass_deviation_vec,
        final_spin_deviation_vec,
        inspiral_kde,
        postinspiral_kde,
        multi_process=multi_process,
    )[0]
    P_final_mass_deviation_final_spin_deviation /= (
        np.sum(P_final_mass_deviation_final_spin_deviation)
        * diff_final_mass_deviation
        * diff_final_spin_deviation
    )

    # Marginalization to one-dimensional joint_posteriors
    P_final_mass_deviation = (
        np.sum(P_final_mass_deviation_final_spin_deviation, axis=0)
        * diff_final_spin_deviation
    )
    P_final_spin_deviation = (
        np.sum(P_final_mass_deviation_final_spin_deviation, axis=1)
        * diff_final_mass_deviation
    )
    return (
        P_final_mass_deviation,
        P_final_spin_deviation,
        P_final_mass_deviation_final_spin_deviation,
        final_mass_deviation_vec,
        final_spin_deviation_vec,
    )
