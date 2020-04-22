# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org> This program is free
# software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from pesummary import conf
from astropy import cosmology as cosmo

_available_cosmologies = cosmo.parameters.available + ["Planck15_lal"]
available_cosmologies = [i.lower() for i in _available_cosmologies]


def get_cosmology(cosmology=conf.cosmology):
    """Return the cosmology that is being used

    Parameters
    ----------
    cosmology: str
        name of a known cosmology
    """
    if cosmology.lower() not in [i.lower() for i in available_cosmologies]:
        raise ValueError(
            "Unrecognised cosmology {}. Available cosmologies are {}".format(
                cosmology, ", ".join(available_cosmologies)
            )
        )
    if cosmology.lower() != "planck15_lal":
        return cosmo.__dict__[cosmology]
    return cosmo.LambdaCDM(H0=67.90, Om0=0.3065, Ode0=0.6935)
