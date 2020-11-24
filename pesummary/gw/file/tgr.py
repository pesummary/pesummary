# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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


def P_integrand(chif, Mf, v1, v2, P_Mfchif_i_interp_object, P_Mfchif_r_interp_object):

    """Compute the integrand of P(dMf/Mfbar, dchif/chifbar).

    inputs:
    chif: vector of values of final spin
    Mf: vector of values of final mass
    v1: dMf/Mfbar value
    v2: dchif/chifbar value
    P_Mfchif_i_interp_object: interpolation function of P_i(Mf, chif)
    P_Mfchif_r_interp_object: interpolation function of P_r(Mf, chif)

    output: integrand of P(dMf/Mfbar, dchif/chifbar)
    """

    Mf_mat, chif_mat = np.meshgrid(Mf, chif)

    # Create dMf and dchif vectors corresponding to the given v1 and v2. These vectors have to be
    # monotonically increasing in order to evaluate the interpolated prob densities. Hence, for
    # v1, v2 < 0, flip them, evaluate the prob density (in column or row) and flip it back
    dMf_i = (1.0 + v1 / 2.0) * Mf
    dchif_i = (1.0 + v2 / 2.0) * chif

    dMf_r = (1.0 - v1 / 2.0) * Mf
    dchif_r = (1.0 - v2 / 2.0) * chif

    if (1.0 + v1 / 2.0) < 0.0:
        dMf_i = np.flipud(dMf_i)
    if (1.0 + v2 / 2.0) < 0.0:
        dchif_i = np.flipud(dchif_i)
    P_i = P_Mfchif_i_interp_object([dMf_i, dchif_i])

    if (1.0 + v1 / 2.0) < 0.0:
        P_i = np.fliplr(P_i)
    if (1.0 + v2 / 2.0) < 0.0:
        P_i = np.flipud(P_i)

    if (1.0 - v1 / 2.0) < 0.0:
        dMf_r = np.flipud(dMf_r)
    if (1.0 - v2 / 2.0) < 0.0:
        dchif_r = np.flipud(dchif_r)
    P_r = P_Mfchif_r_interp_object([dMf_r, dchif_r])

    if (1.0 - v1 / 2.0) < 0.0:
        P_r = np.fliplr(P_r)
    if (1.0 - v2 / 2.0) < 0.0:
        P_r = np.flipud(P_r)

    return P_i * P_r * abs(Mf_mat * chif_mat), P_i, P_r


def calc_sum(Mf, chif, v1, v2, P_Mfchif_i_interp_object, P_Mfchif_r_interp_object):
    Pintg, P_i, P_r = P_integrand(chif, Mf, v1, v2, P_Mfchif_i_interp_object, P_Mfchif_r_interp_object)
    return np.sum(Pintg)


def imrct_delta_parameters_from_Mf_af(
    Mf_inspiral, chif_inspiral, Mf_postinspiral, chif_postinspiral, dMfbyMf_lim=2, dchifbychif_lim=1, N_bins=401
):
    Mf_lim = np.amax(np.append(Mf_inspiral, Mf_postinspiral))
    Mf_bins = np.linspace(-Mf_lim, Mf_lim, N_bins)
    chif_lim = np.amax(np.append(chif_inspiral, chif_postinspiral))
    chif_bins = np.linspace(-chif_lim, chif_lim, N_bins)
    Mf_intp = (Mf_bins[:-1] + Mf_bins[1:]) / 2
    chif_intp = (chif_bins[:-1] + chif_bins[1:]) / 2.0

    inspiral_kde = Bounded_2d_kde(np.array([Mf_inspiral, chif_inspiral]))
    postinspiral_kde = Bounded_2d_kde(np.array([Mf_postinspiral, chif_postinspiral]))

    dMfbyMf_vec = np.linspace(-dMfbyMf_lim, dMfbyMf_lim, N_bins)
    dchifbychif_vec = np.linspace(-dchifbychif_lim, dchifbychif_lim, N_bins)

    diff_dMfbyMf = np.mean(np.diff(dMfbyMf_vec))
    diff_dchifbychif = np.mean(np.diff(dchifbychif_vec))

    P_dMfbyMf_dchifbychif = np.zeros(shape=(N_bins, N_bins))
    for i, v2 in enumerate(dchifbychif_vec):
        for j, v1 in enumerate(dMfbyMf_vec):
            P_dMfbyMf_dchifbychif[i, j] = calc_sum(Mf_intp, chif_intp, v1, v2, inspiral_kde, postinspiral_kde)

    P_dMfbyMf_dchifbychif /= np.sum(P_dMfbyMf_dchifbychif) * diff_dMfbyMf * diff_dchifbychif

    # Marginalization to one-dimensional joint_posteriors

    P_dMfbyMf = np.sum(P_dMfbyMf_dchifbychif, axis=0) * diff_dchifbychif
    P_dchifbychif = np.sum(P_dMfbyMf_dchifbychif, axis=1) * diff_dMfbyMf
    return P_dMfbyMf, P_dchifbychif, P_dMfbyMf_dchifbychif, dMfbyMf_vec, dchifbychif_vec
