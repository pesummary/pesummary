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
    from lalsimulation import DetectorPrefixToLALDetector
    from lal import MSUN_SI, C_SI
    LALINFERENCE_INSTALL = True
except ImportError:
    LALINFERENCE_INSTALL = False

try:
    from astropy.cosmology import z_at_value, Planck15
    import astropy.units as u
    from astropy.time import Time
    ASTROPY = True
except ImportError:
    ASTROPY = False
    logger.warning("You do not have astropy installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


@np.vectorize
def z_from_dL_exact(luminosity_distance):
    """Return the redshift given samples for the luminosity distance
    """
    logger.warning("Estimating the exact redshift for every luminosity "
                   "distance. This may take a few minutes.")
    return z_at_value(Planck15.luminosity_distance, luminosity_distance * u.Mpc)


def z_from_dL_approx(luminosity_distance):
    """Return the approximate redshift given samples for the luminosity
    distance. This technique uses interpolation to estimate the redshift
    """
    logger.warning("The redshift is being approximated using interpolation. "
                   "Bare in mind that this does introduce a small error.")
    d_min = np.min(luminosity_distance)
    d_max = np.max(luminosity_distance)
    zmin = z_at_value(Planck15.luminosity_distance, d_min * u.Mpc)
    zmax = z_at_value(Planck15.luminosity_distance, d_max * u.Mpc)
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
    Dgrid = [Planck15.luminosity_distance(i).value for i in zgrid]
    zvals = np.interp(luminosity_distance, Dgrid, zgrid)
    return zvals


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
    return ((1. / q)**(2. / 5.)) * ((1.0 + (1. / q))**(1. / 5.)) * mchirp


def m2_from_mchirp_q(mchirp, q):
    """Return the mass of the smaller black hole given the chirp mass and
    mass ratio
    """
    return ((1. / q)**(-3. / 5.)) * ((1.0 + (1. / q))**(1. / 5.)) * mchirp


def eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    return (mass1 * mass2) / (mass1 + mass2)**2


def q_from_m1_m2(mass1, mass2):
    """Return the mass ratio given the samples for mass1 and mass2
    """
    return mass2 / mass1


def q_from_eta(symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return (temp - (temp ** 2 - 1) ** 0.5)


def mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    mass1 = (1. / mass_ratio) * total_mass / (1. + (1. / mass_ratio))
    mass2 = total_mass / (1. + (1. / mass_ratio))
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
                    S1y=spin1y[i], S1z=spin1z[i], S2x=spin2x[i], S2y=spin2y[i],
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


def spin_angles_from_azimuthal_and_polar_angles(
        a_1, a_2, a_1_azimuthal, a_1_polar, a_2_azimuthal, a_2_polar):
    """Return the spin angles given samples for a_1, a_2, a_1_azimuthal,
    a_1_polar, a_2_azimuthal, a_2_polar
    """
    spin1x = a_1 * np.sin(a_1_polar) * np.cos(a_1_azimuthal)
    spin1y = a_1 * np.sin(a_1_polar) * np.sin(a_1_azimuthal)
    spin1z = a_1 * np.cos(a_1_polar)

    spin2x = a_2 * np.sin(a_2_polar) * np.cos(a_2_azimuthal)
    spin2y = a_2 * np.sin(a_2_polar) * np.sin(a_2_azimuthal)
    spin2z = a_2 * np.cos(a_2_polar)

    data = [[s1x, s1y, s1z, s2x, s2y, s2z] for s1x, s1y, s1z, s2x, s2y, s2z in
            zip(spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)]
    return data


def time_in_each_ifo(detector, ra, dec, time_gps):
    """Return the event time in a given detector, given samples for ra, dec,
    time
    """
    if LALINFERENCE_INSTALL and ASTROPY:
        gmst = Time(time_gps, format='gps', location=(0, 0))
        corrected_ra = gmst.sidereal_time('mean').rad - ra

        i = np.cos(dec) * np.cos(corrected_ra)
        j = np.cos(dec) * -1 * np.sin(corrected_ra)
        k = np.sin(dec)
        n = np.array([i, j, k])

        dx = [0, 0, 0] - DetectorPrefixToLALDetector(detector).location
        dt = dx.dot(n) / C_SI
        return time_gps + dt
    else:
        raise Exception("Please install LALSuite and astropy for full "
                        "conversions")


def lambda_tilde_from_lambda1_lambda2(lambda1, lambda2, mass1, mass2):
    """Return the dominant tidal term given samples for lambda1 and lambda2
    """
    eta = eta_from_m1_m2(mass1, mass2)
    plus = lambda1 + lambda2
    minus = lambda1 - lambda2
    lambda_tilde = 8 / 13 * (
        (1 + 7 * eta - 31 * eta**2) * plus
        + (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * minus)
    return lambda_tilde


def delta_lambda_from_lambda1_lambda2(lambda1, lambda2, mass1, mass2):
    """Return the second dominant tidal term given samples for lambda1 and
    lambda 2
    """
    eta = eta_from_m1_m2(mass1, mass2)
    plus = lambda1 + lambda2
    minus = lambda1 - lambda2
    delta_lambda = 1 / 2 * (
        (1 - 4 * eta) ** 0.5 * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
        * plus + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2
                  + 3380 / 1319 * eta**3) * minus)
    return delta_lambda


def lambda1_from_lambda_tilde(lambda_tilde, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda_tilde
    """
    eta = eta_from_m1_m2(mass1, mass2)
    q = q_from_m1_m2(mass1, mass2)
    lambda1 = 13 / 8 * lambda_tilde / (
        (1 + 7 * eta - 31 * eta**2) * (1 + q**-5)
        + (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * (1 - q**-5))
    return lambda1


def lambda2_from_lambda1(lambda1, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda1
    """
    q = q_from_m1_m2(mass1, mass2)
    lambda2 = lambda1 / q**5
    return lambda2
