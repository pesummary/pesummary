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

import numpy as np

from pesummary.utils.utils import logger

try:
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from lalsimulation import SimInspiralTransformPrecessingWvf2PE
    from lal import MSUN_SI
    LALINFERENCE_INSTALL = True
except ImportError:
    LALINFERENCE_INSTALL = False

try:
    from astropy.cosmology import z_at_value, Planck15
    import astropy.units as u
    ASTROPY = True
except ImportError:
    ASTROPY = False
    logger.warning("You do not have astropy installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


@np.vectorize
def z_from_dL(luminosity_distance):
    """Return the redshift given samples for the luminosity distance
    """
    return z_at_value(Planck15.luminosity_distance, luminosity_distance * u.Mpc)


def dL_from_z(redshift):
    """Return the luminosity distance given samples for the redshift
    """
    return Planck15.luminosity_distance(redshift).value


def comoving_distance_from_z(redshift):
    """Return the comoving distance given samples for the redshift
    """
    return Planck15.comoving_distance(redshift).value


def m1_source_from_m1_z(mass1, z):
    """Return the source mass of the bigger black hole given samples for the
    detector mass of the bigger black hole and the redshift
    """
    return mass1 / (1. + z)


def m2_source_from_m2_z(mass2, z):
    """Return the source mass of the smaller black hole given samples for the
    detector mass of the smaller black hole and the redshift
    """
    return mass2 / (1. + z)


def m_total_source_from_mtotal_z(total_mass, z):
    """Return the source total mass of the binary given samples for detector
    total mass and redshift
    """
    return total_mass / (1. + z)


def mtotal_from_mtotal_source_z(total_mass_source, z):
    """Return the total mass of the binary given samples for the source total
    mass and redshift
    """
    return total_mass_source * (1. + z)


def mchirp_source_from_mchirp_z(mchirp, z):
    """Return the source chirp mass of the binary given samples for detector
    chirp mass and redshift
    """
    return mchirp / (1. + z)


def mchirp_from_mchirp_source_z(mchirp_source, z):
    """Return the chirp mass of the binary given samples for the source chirp
    mass and redshift
    """
    return mchirp_source * (1. + z)


def mchirp_from_m1_m2(mass1, mass2):
    """Return the chirp mass given the samples for mass1 and mass2

    Parameters
    ----------
    """
    return (mass1 * mass2)**0.6 / (mass1 + mass2)**0.2


def m_total_from_m1_m2(mass1, mass2):
    """Return the total mass given the samples for mass1 and mass2
    """
    return mass1 + mass2


def m1_from_mchirp_q(mchirp, q):
    """Return the mass of the larger black hole given the chirp mass and
    mass ratio
    """
    return (q**(2. / 5.)) * ((1.0 + q)**(1. / 5.)) * mchirp


def m2_from_mchirp_q(mchirp, q):
    """Return the mass of the smaller black hole given the chirp mass and
    mass ratio
    """
    return (q**(-3. / 5.)) * ((1.0 + q)**(1. / 5.)) * mchirp


def eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    return (mass1 * mass2) / (mass1 + mass2)**2


def q_from_m1_m2(mass1, mass2):
    """Return the mass ratio given the samples for mass1 and mass2
    """
    return mass1 / mass2


def q_from_eta(symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5


def mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    mass1 = mass_ratio * total_mass / (1. + mass_ratio)
    mass2 = total_mass / (1. + mass_ratio)
    return eta_from_m1_m2(mass1, mass2)**(3. / 5) * (mass1 + mass2)


def chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return chi_p given samples for mass1, mass2, spin1x, spin1y, spin2x,
    spin2y
    """
    mass_ratio = mass1 / mass2
    B1 = 2.0 + 1.5 * mass_ratio
    B2 = 2.0 + 3.0 / (2 * mass_ratio)
    S1_perp = ((spin1x)**2 + (spin1y)**2)**0.5
    S2_perp = ((spin2x)**2 + (spin2y)**2)**0.5
    chi_p = 1.0 / B1 * np.maximum(B1 * S1_perp, B2 * S2_perp)
    return chi_p


def chi_eff(mass1, mass2, spin1z, spin2z):
    """Return chi_eff given samples for mass1, mass2, spin1z, spin2z
    """
    return (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)


def phi_12_from_phi1_phi2(phi1, phi2):
    """Return the difference in azimuthal angle between S1 and S2 given samples
    for phi1 and phi2
    """
    phi12 = phi2 - phi1
    if isinstance(phi12, float) and phi12 < 0.:
        phi12 += 2 * np.pi
    elif isinstance(phi12, np.ndarray):
        ind = np.where(phi12 < 0.)
        phi12[ind] += 2 * np.pi
    return phi12


def spin_angles(mass_1, mass_2, inc, spin1x, spin1y, spin1z, spin2x, spin2y,
                spin2z, f_ref, phase):
    """Return the spin angles given samples for mass_1, mass_2, inc, spin1x,
    spin1y, spin1z, spin2x, spin2y, spin2z, f_ref, phase
    """
    if LALINFERENCE_INSTALL:
        data = []
        for i in range(len(mass_1)):
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = \
                SimInspiralTransformPrecessingWvf2PE(
                    incl=inc[i], m1=mass_1[i], m2=mass_2[i], S1x=spin1x[i],
                    S1y=spin1y[i], S1z=spin1z[i], S2x=spin2z[i], S2y=spin2y[i],
                    S2z=spin2z[i], fRef=float(f_ref[i]), phiRef=phase[i])
            data.append([theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2])
        return data


def component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                    mass_2, f_ref, phase):
    """Return the component spins given samples for theta_jn, phi_jl, tilt_1,
    tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase
    """
    if LALINFERENCE_INSTALL:
        data = []
        for i in range(len(theta_jn)):
            iota, S1x, S1y, S1z, S2x, S2y, S2z = \
                SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn[i], phi_jl[i], tilt_1[i], tilt_2[i], phi_12[i],
                    a_1[i], a_2[i], mass_1[i] * MSUN_SI, mass_2[i] * MSUN_SI,
                    float(f_ref[i]), phase[i])
            data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
        return data
    else:
        raise Exception("Please install LALSuite for full conversions")
