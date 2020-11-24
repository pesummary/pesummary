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


def imrct_delta_parameters_from_Mf_af(
    Mf_inspiral, af_inspiral, Mf_postinspiral, af_postinspiral, Mf_lim=[-2, 2], af_lim=[-1, 1]
):
    kde_kwargs = dict(xlow=Mf_lim[0], xhigh=Mf_lim[0], ylow=Mf_lim[0], yhigh=af_lim[0])
    inspiral_kde = Bounded_2d_kde([Mf_inspiral, af_inspiral], **kde_kwargs)
    postinspiral_kde = Bounded_2d_kde([Mf_postinspiral, af_postinspiral], **kde_kwargs)
