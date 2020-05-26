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

from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.utils import logger
from pesummary.utils.decorators import array_input
from pesummary import conf

try:
    import lalsimulation
    from lalsimulation import (
        SimInspiralTransformPrecessingNewInitialConditions,
        SimInspiralTransformPrecessingWvf2PE, DetectorPrefixToLALDetector,
        FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL,
        FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION,
        FLAG_SEOBNRv4P_ZFRAME_L
    )
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

DEFAULT_SEOBFLAGS = {
    "SEOBNRv4P_SpinAlignedEOBversion": 4,
    "SEOBNRv4P_SymmetrizehPlminusm": 1,
    "SEOBNRv4P_HamiltonianDerivative": FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL,
    "SEOBNRv4P_euler_extension": FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION,
    "SEOBNRv4P_Zframe": FLAG_SEOBNRv4P_ZFRAME_L,
    "SEOBNRv4P_debug": 0
}


@array_input
def _z_from_dL_exact(luminosity_distance, cosmology):
    """Return the redshift given samples for the luminosity distance
    """
    return z_at_value(
        cosmology.luminosity_distance, luminosity_distance * u.Mpc
    )


def z_from_dL_exact(luminosity_distance, cosmology="Planck15", multi_process=1):
    """Return the redshift given samples for the luminosity distance
    """
    from pesummary.gw.cosmology import get_cosmology
    import multiprocessing

    logger.warning("Estimating the exact redshift for every luminosity "
                   "distance. This may take a few minutes.")
    cosmo = get_cosmology(cosmology)
    args = np.array(
        [luminosity_distance, [cosmo] * len(luminosity_distance)],
        dtype=object
    ).T
    with multiprocessing.Pool(multi_process) as pool:
        z = pool.starmap(_z_from_dL_exact, args)
    return z


@array_input
def z_from_dL_approx(
    luminosity_distance, N=100, cosmology="Planck15", **kwargs
):
    """Return the approximate redshift given samples for the luminosity
    distance. This technique uses interpolation to estimate the redshift
    """
    from pesummary.gw.cosmology import get_cosmology

    logger.warning("The redshift is being approximated using interpolation. "
                   "Bear in mind that this does introduce a small error.")
    cosmo = get_cosmology(cosmology)
    d_min = np.min(luminosity_distance)
    d_max = np.max(luminosity_distance)
    zmin = z_at_value(cosmo.luminosity_distance, d_min * u.Mpc)
    zmax = z_at_value(cosmo.luminosity_distance, d_max * u.Mpc)
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), N)
    Dgrid = [cosmo.luminosity_distance(i).value for i in zgrid]
    zvals = np.interp(luminosity_distance, Dgrid, zgrid)
    return zvals


@array_input
def dL_from_z(redshift, cosmology="Planck15"):
    """Return the luminosity distance given samples for the redshift
    """
    from pesummary.gw.cosmology import get_cosmology

    cosmo = get_cosmology(cosmology)
    return cosmo.luminosity_distance(redshift).value


@array_input
def comoving_distance_from_z(redshift, cosmology="Planck15"):
    """Return the comoving distance given samples for the redshift
    """
    from pesummary.gw.cosmology import get_cosmology

    cosmo = get_cosmology(cosmology)
    return cosmo.comoving_distance(redshift).value


def _source_from_detector(parameter, z):
    """Return the source parameter given samples for the detector parameter and
    the redshift
    """
    return parameter / (1. + z)


def _detector_from_source(parameter, z):
    """Return the detector parameter given samples for the source parameter and
    the redshift
    """
    return parameter * (1. + z)


@array_input
def m1_source_from_m1_z(mass_1, z):
    """Return the source mass of the bigger black hole given samples for the
    detector mass of the bigger black hole and the redshift
    """
    return _source_from_detector(mass_1, z)


@array_input
def m2_source_from_m2_z(mass_2, z):
    """Return the source mass of the smaller black hole given samples for the
    detector mass of the smaller black hole and the redshift
    """
    return _source_from_detector(mass_2, z)


@array_input
def m_total_source_from_mtotal_z(total_mass, z):
    """Return the source total mass of the binary given samples for detector
    total mass and redshift
    """
    return _source_from_detector(total_mass, z)


@array_input
def mtotal_from_mtotal_source_z(total_mass_source, z):
    """Return the total mass of the binary given samples for the source total
    mass and redshift
    """
    return _detector_from_source(total_mass_source, z)


@array_input
def mchirp_source_from_mchirp_z(mchirp, z):
    """Return the source chirp mass of the binary given samples for detector
    chirp mass and redshift
    """
    return _source_from_detector(mchirp, z)


@array_input
def mchirp_from_mchirp_source_z(mchirp_source, z):
    """Return the chirp mass of the binary given samples for the source chirp
    mass and redshift
    """
    return _detector_from_source(mchirp_source, z)


@array_input
def mchirp_from_m1_m2(mass1, mass2):
    """Return the chirp mass given the samples for mass1 and mass2

    Parameters
    ----------
    """
    return (mass1 * mass2)**0.6 / (mass1 + mass2)**0.2


@array_input
def m_total_from_m1_m2(mass1, mass2):
    """Return the total mass given the samples for mass1 and mass2
    """
    return mass1 + mass2


@array_input
def m1_from_mchirp_q(mchirp, q):
    """Return the mass of the larger black hole given the chirp mass and
    mass ratio
    """
    return ((1. / q)**(2. / 5.)) * ((1.0 + (1. / q))**(1. / 5.)) * mchirp


@array_input
def m2_from_mchirp_q(mchirp, q):
    """Return the mass of the smaller black hole given the chirp mass and
    mass ratio
    """
    return ((1. / q)**(-3. / 5.)) * ((1.0 + (1. / q))**(1. / 5.)) * mchirp


@array_input
def eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    return (mass1 * mass2) / (mass1 + mass2)**2


@array_input
def q_from_m1_m2(mass1, mass2):
    """Return the mass ratio given the samples for mass1 and mass2
    """
    return mass2 / mass1


@array_input
def invq_from_m1_m2(mass1, mass2):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass1 and mass2
    """
    return 1. / q_from_m1_m2(mass1, mass2)


@array_input
def invq_from_q(mass_ratio):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass ratio (mass2/mass1)
    """
    return 1. / mass_ratio


@array_input
def q_from_eta(symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return (temp - (temp ** 2 - 1) ** 0.5)


@array_input
def mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    mass1 = (1. / mass_ratio) * total_mass / (1. + (1. / mass_ratio))
    mass2 = total_mass / (1. + (1. / mass_ratio))
    return eta_from_m1_m2(mass1, mass2)**(3. / 5) * (mass1 + mass2)


@array_input
def chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return chi_p given samples for mass1, mass2, spin1x, spin1y, spin2x,
    spin2y
    """
    mass_ratio = mass2 / mass1
    S1_perp = ((spin1x)**2 + (spin1y)**2)**0.5
    S2_perp = ((spin2x)**2 + (spin2y)**2)**0.5
    chi_p = np.maximum(
        S1_perp, (4 * mass_ratio + 3) / (3 * mass_ratio + 4) * mass_ratio
        * S2_perp
    )
    return chi_p


@array_input
def chi_eff(mass1, mass2, spin1z, spin2z):
    """Return chi_eff given samples for mass1, mass2, spin1z, spin2z
    """
    return (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)


@array_input
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


@array_input
def phi1_from_spins(spin_1x, spin_1y):
    """Return phi_1 given samples for spin_1x and spin_1y
    """
    phi_1 = np.fmod(2 * np.pi + np.arctan2(spin_1y, spin_1x), 2 * np.pi)
    return phi_1


@array_input
def phi2_from_spins(spin_2x, spin_2y):
    """Return phi_2 given samples for spin_2x and spin_2y
    """
    phi_2 = np.fmod(2 * np.pi + np.arctan2(spin_2y, spin_2x), 2 * np.pi)
    return phi_2


@array_input
def spin_angles(mass_1, mass_2, inc, spin1x, spin1y, spin1z, spin2x, spin2y,
                spin2z, f_ref, phase):
    """Return the spin angles given samples for mass_1, mass_2, inc, spin1x,
    spin1y, spin1z, spin2x, spin2y, spin2z, f_ref, phase
    """
    return_float = False
    if isinstance(mass_1, (int, float)):
        return_float = True
        mass_1 = [mass_1]
        mass_2 = [mass_2]
        inc = [inc]
        spin1x = [spin1x]
        spin1y = [spin1y]
        spin1z = [spin1z]
        spin2x = [spin2x]
        spin2y = [spin2y]
        spin2z = [spin2z]
        f_ref = [f_ref]
        phase = [phase]

    if LALINFERENCE_INSTALL:
        data = []
        for i in range(len(mass_1)):
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = \
                SimInspiralTransformPrecessingWvf2PE(
                    incl=inc[i], m1=mass_1[i], m2=mass_2[i], S1x=spin1x[i],
                    S1y=spin1y[i], S1z=spin1z[i], S2x=spin2x[i], S2y=spin2y[i],
                    S2z=spin2z[i], fRef=float(f_ref[i]), phiRef=float(phase[i]))
            data.append([theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2])
        if return_float:
            return data[0]
        return data


@array_input
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
                    float(f_ref[i]), float(phase[i]))
            data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
        return data
    else:
        raise Exception("Please install LALSuite for full conversions")


@array_input
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


@array_input
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


@array_input
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


@array_input
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


@array_input
def lambda1_from_lambda_tilde(lambda_tilde, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda_tilde
    """
    eta = eta_from_m1_m2(mass1, mass2)
    q = q_from_m1_m2(mass1, mass2)
    lambda1 = 13 / 8 * lambda_tilde / (
        (1 + 7 * eta - 31 * eta**2) * (1 + q**-5)
        + (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * (1 - q**-5))
    return lambda1


@array_input
def lambda2_from_lambda1(lambda1, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda1
    """
    q = q_from_m1_m2(mass1, mass2)
    lambda2 = lambda1 / q**5
    return lambda2


@array_input
def _ifo_snr(IFO_abs_snr, IFO_snr_angle):
    """Return the matched filter SNR for a given IFO given samples for the
    absolute SNR and the angle
    """
    return IFO_abs_snr * np.cos(IFO_snr_angle)


@array_input
def network_snr(snrs):
    """Return the network SNR for N IFOs

    Parameters
    ----------
    snrs: list
        list of numpy.array objects containing the snrs samples for a particular
        IFO
    """
    squares = [i**2 for i in snrs]
    network_snr = np.sqrt(np.sum(squares, axis=0))
    return network_snr


@array_input
def tilt_angles_and_phi_12_from_spin_vectors_and_L(a_1, a_2, Ln):
    """Return the tilt angles and phi_12 given samples for the spin vectors
    and the orbital angular momentum

    Parameters
    ----------
    a_1: np.ndarray
        Spin vector for the larger object
    a_2: np.ndarray
        Spin vector for the smaller object
    Ln: np.ndarray
        Orbital angular momentum of the binary
    """
    a_1_norm = np.linalg.norm(a_1)
    a_2_norm = np.linalg.norm(a_2)
    Ln /= np.linalg.norm(Ln)
    a_1_dot = np.dot(a_1, Ln)
    a_2_dot = np.dot(a_2, Ln)
    a_1_perp = a_1 - a_1_dot * Ln
    a_2_perp = a_2 - a_2_dot * Ln
    cos_tilt_1 = a_1_dot / a_1_norm
    cos_tilt_2 = a_2_dot / a_2_norm
    cos_phi_12 = np.dot(a_1_perp, a_2_perp) / (
        np.linalg.norm(a_1_perp) * np.linalg.norm(a_2_perp)
    )
    # set quadrant of phi12
    phi_12 = np.arccos(cos_phi_12)
    if np.sign(np.dot(Ln, np.cross(a_1, a_2))) < 0.:
        phi_12 = 2. * np.pi - phi_12

    return np.arccos(cos_tilt_1), np.arccos(cos_tilt_2), phi_12


def _wrapper_return_final_mass_and_final_spin_from_waveform(args):
    """Wrapper function to calculate the remnant properties for a given waveform
    for a pool of workers

    Parameters
    ----------
    args: np.ndarray
        2 dimensional array giving arguments to pass to
        _return_final_mass_and_final_spin_from_waveform. The first argument
        in each sublist is the keyword and the second argument in each sublist
        is the item you wish to pass
    """
    kwargs = {arg[0]: arg[1] for arg in args}
    return _return_final_mass_and_final_spin_from_waveform(**kwargs)


def _return_final_mass_and_final_spin_from_waveform(
    mass_function=None, spin_function=None, mass_function_args=[],
    spin_function_args=[], mass_function_return_function=None,
    mass_function_return_index=None, spin_function_return_function=None,
    spin_function_return_index=None, mass_1_index=0, mass_2_index=1,
    nsamples=0, approximant=None, default_SEOBNRv4P_kwargs=False
):
    """Return the final mass and final spin given functions to use

    Parameters
    ----------
    mass_function: func
        function you wish to use to calculate the final mass
    spin_function: func
        function you wish to use to calculate the final spin
    mass_function_args: list
        list of arguments you wish to pass to mass_function
    spin_function_args: list
        list of arguments you wish to pass to spin_function
    mass_function_return_function: str, optional
        function used to extract the final mass from the quantity returned from
        mass_function. For example, if mass_function returns a list and the
        final_mass is a property of the 3 arg of this list,
        mass_function_return_function='[3].final_mass'
    mass_function_return_index: str, optional
        if mass_function returns a list of parameters,
        mass_function_return_index indicates the index of `final_mass` in the
        list
    spin_function_return_function: str, optional
        function used to extract the final spin from the quantity returned from
        spin_function. For example, if spin_function returns a list and the
        final_spin is a property of the 3 arg of this list,
        spin_function_return_function='[3].final_spin'
    spin_function_return_index: str, optional
        if spin_function returns a list of parameters,
        spin_function_return_index indicates the index of `final_spin` in the
        list
    mass_1_index: int, optional
        the index of mass_1 in mass_function_args. Default is 0
    mass_2_index: int, optional
        the index of mass_2 in mass_function_args. Default is 1
    nsamples: int, optional
        the total number of samples
    approximant: str, optional
        the approximant used
    default_SEOBNRv4P_kwargs: Bool, optional
        if True, use the default SEOBNRv4P flags
    """
    if default_SEOBNRv4P_kwargs:
        mode_array, seob_flags = _setup_SEOBNRv4P_args()
        mass_function_args += [mode_array, seob_flags]
        spin_function_args += [mode_array, seob_flags]
    fm = mass_function(*mass_function_args)
    if mass_function_return_function is not None:
        fm = eval("fm{}".format(mass_function_return_function))
    elif mass_function_return_index is not None:
        fm = fm[mass_function_return_index]
    fs = spin_function(*spin_function_args)
    if spin_function_return_function is not None:
        fs = eval("fs{}".format(spin_function_return_function))
    elif spin_function_return_index is not None:
        fs = fs[spin_function_return_index]
    final_mass = fm * (
        mass_function_args[mass_1_index] + mass_function_args[mass_2_index]
    ) / MSUN_SI
    final_spin = fs
    return final_mass, final_spin


def _setup_SEOBNRv4P_args(mode=[2, 2], seob_flags=DEFAULT_SEOBFLAGS):
    """Setup the SEOBNRv4P[HM] kwargs
    """
    from lalsimulation import (
        SimInspiralCreateModeArray, SimInspiralModeArrayActivateMode
    )
    from lal import DictInsertINT4Value, CreateDict

    mode_array = SimInspiralCreateModeArray()
    SimInspiralModeArrayActivateMode(mode_array, mode[0], mode[1])
    _seob_flags = CreateDict()
    for key, item in seob_flags.items():
        DictInsertINT4Value(_seob_flags, key, item)
    return mode_array, _seob_flags


@array_input
def _final_from_initial(
    mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
    approximant="SEOBNRv4", iota=None, luminosity_distance=None, f_ref=None,
    phi_ref=None, mode=[2, 2], delta_t=1. / 4096, seob_flags=DEFAULT_SEOBFLAGS,
    return_fits_used=False, multi_process=None
):
    """Calculate the final mass and final spin given the initial parameters
    of the binary using the approximant directly

    Parameters
    ----------
    mass_1: float/np.ndarray
        primary mass of the binary
    mass_2: float/np.ndarray
        secondary mass of the binary
    spin_1x: float/np.ndarray
        x component of the primary spin
    spin_1y: float/np.ndarray
        y component of the primary spin
    spin_1z: float/np.ndarray
        z component of the primary spin
    spin_2x: float/np.ndarray
        x component of the secondary spin
    spin_2y: float/np.ndarray
        y component of the secondary spin
    spin_2z: float/np.ndarray
        z component of the seconday spin
    approximant: str
        name of the approximant you wish to use for the remnant fits
    iota: float/np.ndarray, optional
        the angle between the total orbital angular momentum and the line of
        sight of the source. Used when calculating the remnant fits for
        SEOBNRv4PHM. Since we only need the EOB dynamics here it does not matter
        what we pass
    luminosity_distance: float/np.ndarray, optional
        the luminosity distance of the source. Used when calculating the
        remnant fits for SEOBNRv4PHM. Since we only need the EOB dynamics here
        it does not matter what we pass.
    f_ref: float/np.ndarray, optional
        the reference frequency at which the spins are defined
    phi_ref: float/np.ndarray, optional
        the coalescence phase of the binary
    mode: list, optional
        specific mode to use when calculating the remnant fits for SEOBNRv4PHM.
        Since we only need the EOB dynamics here it does not matter what we
        pass.
    delta_t: float, optional
        the sampling rate used in the analysis, Used when calculating the
        remnant fits for SEOBNRv4PHM
    seob_flags: dict, optional
        dictionary containing the SEOB flags. Used when calculating the remnant
        fits for SEOBNRv4PHM
    return_fits_used: Bool, optional
        if True, return the approximant that was used.
    multi_process: int, optional
        the number of cores to use when calculating the remnant fits
    """
    from lalsimulation import (
        SimIMREOBFinalMassSpin, SimIMREOBFinalMassSpinPrec,
        SimInspiralGetSpinSupportFromApproximant,
        SimIMRSpinPrecEOBWaveformAll, SimPhenomUtilsIMRPhenomDFinalMass,
        SimPhenomUtilsPhenomPv2FinalSpin
    )
    from pesummary.utils.utils import iterator
    import multiprocessing

    def convert_args_for_multi_processing(kwargs):
        args = []
        for n in range(kwargs["nsamples"]):
            _args = []
            for key, item in kwargs.items():
                if key == "mass_function_args" or key == "spin_function_args":
                    _args.append([key, [arg[n] for arg in item]])
                else:
                    _args.append([key, item])
            args.append(_args)
        return args

    try:
        approx = getattr(lalsimulation, approximant)
    except AttributeError:
        raise ValueError(
            "The waveform '{}' is not supported by lalsimulation"
        )

    m1 = mass_1 * MSUN_SI
    m2 = mass_2 * MSUN_SI
    kwargs = {"nsamples": len(mass_1), "approximant": approximant}
    if approximant.lower() in ["seobnrv4p", "seobnrv4phm"]:
        if any(i is None for i in [iota, luminosity_distance, f_ref, phi_ref]):
            raise ValueError(
                "The approximant '{}' requires samples for iota, f_ref, "
                "phi_ref and luminosity_distance. Please pass these "
                "samples.".format(approximant)
            )
        if len(delta_t) == 1:
            delta_t = [delta_t[0]] * len(mass_1)
        elif len(delta_t) != len(mass_1):
            raise ValueError(
                "Please provide either a single 'delta_t' that is is used for "
                "all samples, or a single 'delta_t' for each sample"
            )
        mode_array, _seob_flags = _setup_SEOBNRv4P_args(
            mode=mode, seob_flags=seob_flags
        )
        args = np.array([
            phi_ref, delta_t, m1, m2, f_ref, luminosity_distance, iota,
            spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
            [mode_array] * len(mass_1), [_seob_flags] * len(mass_1)
        ])
        kwargs.update(
            {
                "mass_function": SimIMRSpinPrecEOBWaveformAll,
                "spin_function": SimIMRSpinPrecEOBWaveformAll,
                "mass_function_args": args,
                "spin_function_args": args,
                "mass_function_return_function": "[21].data[6]",
                "spin_function_return_function": "[21].data[7]",
                "mass_1_index": 2,
                "mass_2_index": 3,
            }
        )
    elif approximant.lower() in ["seobnrv4"]:
        spin1 = np.array([spin_1x, spin_1y, spin_1z]).T
        spin2 = np.array([spin_2x, spin_2y, spin_2z]).T
        app = np.array([approx] * len(mass_1))
        kwargs.update(
            {
                "mass_function": SimIMREOBFinalMassSpin,
                "spin_function": SimIMREOBFinalMassSpin,
                "mass_function_args": [m1, m2, spin1, spin2, app],
                "spin_function_args": [m1, m2, spin1, spin2, app],
                "mass_function_return_index": 1,
                "spin_function_return_index": 2
            }
        )
    elif "phenompv3" in approximant.lower():
        kwargs.update(
            {
                "mass_function": SimPhenomUtilsIMRPhenomDFinalMass,
                "spin_function": SimPhenomUtilsPhenomPv2FinalSpin,
                "mass_function_args": [m1, m2, spin_1z, spin_2z],
                "spin_function_args": [m1, m2, spin_1z, spin_2z]
            }
        )
        if SimInspiralGetSpinSupportFromApproximant(approx) > 2:
            # matches the waveform's internal usage as corrected in
            # https://git.ligo.org/lscsoft/lalsuite/-/merge_requests/1270
            _chi_p = chi_p(mass_1, mass_2, spin_1x, spin_1y, spin_2x, spin_2y)
            kwargs["spin_function_args"].append(_chi_p)
        else:
            kwargs["spin_function_args"].append(np.zeros_like(mass_1))
    else:
        raise ValueError(
            "The waveform '{}' is not support by this function.".format(
                approximant
            )
        )

    args = convert_args_for_multi_processing(kwargs)
    if multi_process is not None and multi_process[0] != 1:
        _multi_process = multi_process[0]
        if approximant.lower() in ["seobnrv4p", "seobnrv4phm"]:
            logger.warn(
                "Ignoring passed 'mode' and 'seob_flags' options. Defaults "
                "must be used with multiprocessing. If you wish to use custom "
                "options, please set `multi_process=None`"
            )
            _kwargs = kwargs.copy()
            _kwargs["mass_function_args"] = kwargs["mass_function_args"][:-2]
            _kwargs["spin_function_args"] = kwargs["spin_function_args"][:-2]
            _kwargs["default_SEOBNRv4P_kwargs"] = True
            args = convert_args_for_multi_processing(_kwargs)
        with multiprocessing.Pool(_multi_process) as pool:
            data = np.array(list(
                iterator(
                    pool.imap(
                        _wrapper_return_final_mass_and_final_spin_from_waveform,
                        args
                    ), tqdm=True, desc="Evaluating {} fit".format(approximant),
                    logger=logger, total=len(mass_1)
                )
            )).T
    else:
        final_mass, final_spin = [], []
        _iterator = iterator(
            range(kwargs["nsamples"]), tqdm=True, total=len(mass_1),
            desc="Evaluating {} fit".format(approximant), logger=logger
        )
        for i in _iterator:
            data = _wrapper_return_final_mass_and_final_spin_from_waveform(
                args[i]
            )
            final_mass.append(data[0])
            final_spin.append(data[1])
        data = [final_mass, final_spin]
    if return_fits_used:
        return data, [approximant]
    return data


def final_remnant_properties_from_NRSurrogate(
    *args, f_low=20., f_ref=20., model="NRSur7dq4Remnant", return_fits_used=False,
    properties=["final_mass", "final_spin", "final_kick"], approximant="SEOBNRv4PHM"
):
    """Return the properties of the final remnant resulting from a BBH merger using
    NRSurrogate fits

    Parameters
    ---------
    f_low: float/np.ndarray
        The low frequency cut-off used in the analysis. Default is 20Hz
    f_ref: float/np.ndarray
        The reference frequency used in the analysis. Default is 20Hz
    model: str, optional
        The name of the NRSurrogate model you wish to use
    return_fits_used: Bool, optional
        if True, return the approximant that was used.
    properties: list, optional
        The list of properties you wish to calculate
    approximant: str, optional
        The approximant that was used to generate the posterior samples
    """
    from pesummary.gw.file.nrutils import NRSur_fit

    fit = NRSur_fit(
        *args, f_low=f_low, f_ref=f_ref, model=model, fits=properties,
        approximant=approximant
    )
    if return_fits_used:
        return fit, [model]
    return fit


def final_mass_of_merger_from_NR(
    *args, NRfit="average", final_spin=None, return_fits_used=False
):
    """Return the final mass resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
    final_spin: float/np.ndarray, optional
        precomputed final spin of the remnant.
    return_fits_used: Bool, optional
        if True, return the fits that were used. Only used when NRfit='average'
    """
    from pesummary.gw.file import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_final_mass_average")
    elif "panetal" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_mass_non_spinning_Panetal"
        )
    else:
        func = getattr(
            nrutils, "bbh_final_mass_non_precessing_{}".format(NRfit)
        )
    if "healy" in NRfit.lower():
        return func(*args, final_spin=final_spin)
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)


def final_mass_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final mass resulting from a BBH merger using NRSurrogate
    fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_mass"],
        return_fits_used=return_fits_used,
        approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_mass"], data[1]
    return data["final_mass"]


def final_mass_of_merger_from_waveform(*args, **kwargs):
    """Return the final mass resulting from a BBH merger using a given
    approximant
    """
    return _final_from_initial(*args, **kwargs)[0]


def final_spin_of_merger_from_NR(
    *args, NRfit="average", return_fits_used=False
):
    """Return the final spin resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
    return_fits_used: Bool, optional
        if True, return the fits that were used. Only used when NRfit='average'
    """
    from pesummary.gw.file import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_final_spin_average_precessing")
    elif "pan" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_spin_non_spinning_Panetal"
        )
    elif "precessing" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_spin_precessing_{}".format(
                NRfit.split("precessing_")[1]
            )
        )
    else:
        func = getattr(
            nrutils, "bbh_final_spin_non_precessing_{}".format(NRfit)
        )
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)


def final_spin_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final spin resulting from a BBH merger using NRSurrogate
    fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_spin"],
        return_fits_used=return_fits_used, approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_spin"], data[1]
    return data["final_spin"]


def final_spin_of_merger_from_waveform(*args, **kwargs):
    """Return the final spin resulting from a BBH merger using a given
    approximant
    """
    return _final_from_initial(*args, **kwargs)[1]


def final_kick_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final kick of the remnant resulting from a BBH merger
    using NRSurrogate fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_kick"],
        return_fits_used=return_fits_used, approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_kick"], data[1]
    return data["final_kick"]


def final_mass_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    final_spin=None, return_fits_used=False, model="NRSur7dq4Remnant"
):
    """Return the final mass resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    spin_1z: float/np.ndarray
        float/array of primary spin aligned with the orbital angular momentum
    spin_2z: float/np.ndarray
        float/array of secondary spin aligned with the orbital angular momentum
    method: str
        The method you wish to use to calculate the final mass of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if method.lower() == "nr":
        mass_func = final_mass_of_merger_from_NR
        kwargs = {
            "NRfit": NRfit, "final_spin": final_spin,
            "return_fits_used": return_fits_used
        }
    elif "nrsur" in method.lower():
        mass_func = final_mass_of_merger_from_NRSurrogate
        kwargs = {
            "approximant": approximant, "return_fits_used": return_fits_used,
            "model": model
        }
    else:
        mass_func = final_mass_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return mass_func(*args, **kwargs)


def final_spin_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    return_fits_used: False, model="NRSur7dq4Remnant"
):
    """Return the final mass resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular
        momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    method: str
        The method you wish to use to calculate the final mass of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if method.lower() == "nr":
        spin_func = final_spin_of_merger_from_NR
        kwargs = {"NRfit": NRfit, "return_fits_used": return_fits_used}
    elif "nrsur" in method.lower():
        spin_func = final_spin_of_merger_from_NRSurrogate
        kwargs = {
            "approximant": approximant, "return_fits_used": return_fits_used,
            "model": model
        }
    else:
        spin_func = final_spin_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return spin_func(*args, **kwargs)


def final_kick_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    return_fits_used: False, model="NRSur7dq4Remnant"
):
    """Return the final kick velocity of the remnant resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular
        momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    method: str
        The method you wish to use to calculate the final kick of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if "nrsur" not in method.lower():
        raise NotImplementedError(
            "Currently you can only work out the final kick velocity using "
            "NRSurrogate fits."
        )
    velocity_func = final_kick_of_merger_from_NRSurrogate
    kwargs = {
        "approximant": approximant, "return_fits_used": return_fits_used,
        "model": model
    }
    return velocity_func(*args, **kwargs)


def peak_luminosity_of_merger(*args, NRfit="average", return_fits_used=False):
    """Return the peak luminosity of an aligned-spin BBH using NR fits

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    spin_1z: float/np.ndarray
        float/array of primary spin aligned with the orbital angular momentum
    spin_2z: float/np.ndarray
        float/array of secondary spin aligned with the orbital angular momentum
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average'
    """
    from pesummary.gw.file import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_peak_luminosity_average")
    else:
        func = getattr(
            nrutils, "bbh_peak_luminosity_non_precessing_{}".format(NRfit)
        )
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)


def magnitude_from_vector(vector):
    """Return the magnitude of a vector

    Parameters
    ----------
    vector: list, np.ndarray
        The vector you wish to return the magnitude for.
    """
    vector = np.atleast_2d(vector)
    return np.linalg.norm(vector, axis=1)


class _Redshift(object):
    exact = z_from_dL_exact
    approx = z_from_dL_approx


class _Conversion(object):
    """Class to calculate all possible derived quantities

    Parameters
    ----------
    data: dict, list
        either a dictionary or samples or a list of parameters and a list of
        samples. See the examples below for details
    extra_kwargs: dict, optional
        dictionary of kwargs associated with this set of posterior samples.
    f_low: float, optional
        the low frequency cut-off to use when evolving the spins
    f_ref: float, optional
        the reference frequency when spins are defined
    approximant: str, optional
        the approximant to use when evolving the spins
    evolve_spins: float/str, optional
        the final velocity to evolve the spins up to.
    return_kwargs: Bool, optional
        if True, return a modified dictionary of kwargs containing information
        about the conversion
    NRSur_fits: float/str, optional
        the NRSurrogate model to use to calculate the remnant fits. If nothing
        passed, the average NR fits are used instead
    waveform_fits: Bool, optional
        if True, the approximant is used to calculate the remnant fits. Default
        is False which means that the average NR fits are used
    multi_process: int, optional
        number of cores to use to parallelize the computationally expensive
        conversions
    redshift_method: str, optional
        method you wish to use when calculating the redshift given luminosity
        distance samples. If redshift samples already exist, this method is not
        used. Default is 'approx' meaning that interpolation is used to calculate
        the redshift given N luminosity distance points.
    cosmology: str, optional
        cosmology you wish to use when calculating the redshift given luminosity
        distance samples.
    force_non_evolved: Bool, optional
        force non evolved remnant quantities to be calculated when evolved quantities
        already exist in the input. Default False
    force_remnant_computation: Bool, optional
        force remnant quantities to be calculated for systems that include
        tidal deformability parameters where BBH fits may not be applicable.
        Default False.
    add_zero_spin: Bool, optional
        if no spins are present in the posterior table, add spins with 0 value.
        Default False.
    regenerate: list, optional
        list of posterior distributions that you wish to regenerate
    return_dict: Bool, optional
        if True, return a pesummary.utils.utils.SamplesDict object

    Examples
    --------
    There are two ways of passing arguments to this conversion class, either
    a dictionary of samples or a list of parameters and a list of samples. See
    the examples below:

    >>> samples = {"mass_1": 10, "mass_2": 5}
    >>> converted_samples = _Conversion(samples)

    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [10, 5]
    >>> converted_samples = _Conversion(parameters, samples)

    >>> samples = {"mass_1": [10, 20], "mass_2": [5, 8]}
    >>> converted_samples = _Conversion(samples)

    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [[10, 5], [20, 8]]
    """
    def __new__(cls, *args, **kwargs):
        obj = super(_Conversion, cls).__new__(cls)
        base_replace = (
            "'{}': {} already found in the result file. Overwriting with "
            "the passed {}"
        )
        if len(args) > 2:
            raise ValueError(
                "The _Conversion module only takes as arguments a dictionary "
                "of samples or a list of parameters and a list of samples"
            )
        elif isinstance(args[0], dict):
            parameters = list(args[0].keys())
            samples = np.atleast_2d(
                np.array([args[0][i] for i in parameters]).T
            ).tolist()
        else:
            parameters, samples = args
            samples = np.atleast_2d(samples).tolist()
        extra_kwargs = kwargs.get("extra_kwargs", {"sampler": {}, "meta_data": {}})
        f_low = kwargs.get("f_low", None)
        f_ref = kwargs.get("f_ref", None)
        approximant = kwargs.get("approximant", None)
        NRSurrogate = kwargs.get("NRSur_fits", False)
        redshift_method = kwargs.get("redshift_method", "approx")
        cosmology = kwargs.get("cosmology", "Planck15")
        force_non_evolved = kwargs.get("force_non_evolved", False)
        force_remnant = kwargs.get("force_remnant_computation", False)
        if redshift_method not in ["approx", "exact"]:
            raise ValueError(
                "'redshift_method' can either be 'approx' corresponding to "
                "an approximant method, or 'exact' corresponding to an exact "
                "method of calculating the redshift"
            )
        if isinstance(NRSurrogate, bool) and NRSurrogate:
            raise ValueError(
                "'NRSur_fits' must be a string corresponding to the "
                "NRSurrogate model you wish to use to calculate the remnant "
                "quantities"
            )
        waveform_fits = kwargs.get("waveform_fits", False)
        if NRSurrogate and waveform_fits:
            raise ValueError(
                "Unable to use both the NRSurrogate and {} to calculate "
                "remnant quantities. Please select only one option".format(
                    approximant
                )
            )
        evolve_spins = kwargs.get("evolve_spins", False)
        if isinstance(evolve_spins, bool) and evolve_spins:
            raise ValueError(
                "'evolve_spins' must be a float, the final velocity to "
                "evolve the spins up to, or a string, 'ISCO', meaning "
                "evolve the spins up to the ISCO frequency"
            )
        if not evolve_spins and (NRSurrogate or waveform_fits):
            if "eob" in approximant or NRSurrogate:
                logger.warn(
                    "Only evolved spin remnant quantities are returned by the "
                    "{} fits.".format(
                        "NRSurrogate" if NRSurrogate else approximant
                    )
                )
        elif evolve_spins and (NRSurrogate or waveform_fits):
            if "eob" in approximant or NRSurrogate:
                logger.warn(
                    "The {} fits already evolve the spins. Therefore "
                    "additional spin evolution will not be performed.".format(
                        "NRSurrogate" if NRSurrogate else approximant
                    )
                )
            else:
                logger.warn(
                    "The {} fits are not applied with spin evolution.".format(
                        approximant
                    )
                )
            evolve_spins = False

        if f_low is not None and "f_low" in extra_kwargs["meta_data"].keys():
            logger.warn(
                base_replace.format(
                    "f_low", extra_kwargs["meta_data"]["f_low"], f_low
                )
            )
            extra_kwargs["meta_data"]["f_low"] = f_low
        elif f_low is not None:
            extra_kwargs["meta_data"]["f_low"] = f_low
        if approximant is not None and "approximant" in extra_kwargs["meta_data"].keys():
            logger.warn(
                base_replace.format(
                    "approximant", extra_kwargs["meta_data"]["approximant"],
                    approximant
                )
            )
            extra_kwargs["meta_data"]["approximant"] = approximant
        elif approximant is not None:
            extra_kwargs["meta_data"]["approximant"] = approximant
        if f_ref is not None and "f_ref" in extra_kwargs["meta_data"].keys():
            logger.warn(
                base_replace.format(
                    "f_ref", extra_kwargs["meta_data"]["f_ref"], f_ref
                )
            )
            extra_kwargs["meta_data"]["f_ref"] = f_ref
        elif f_ref is not None:
            extra_kwargs["meta_data"]["f_ref"] = f_ref
        regenerate = kwargs.get("regenerate", None)
        multi_process = kwargs.get("multi_process", None)
        if multi_process is not None:
            multi_process = int(multi_process)
        obj.__init__(
            parameters, samples, extra_kwargs, evolve_spins, NRSurrogate,
            waveform_fits, multi_process, regenerate, redshift_method,
            cosmology, force_non_evolved, force_remnant,
            kwargs.get("add_zero_spin", False)
        )
        return_kwargs = kwargs.get("return_kwargs", False)
        if kwargs.get("return_dict", True) and return_kwargs:
            return [
                SamplesDict(obj.parameters, np.array(obj.samples).T),
                obj.extra_kwargs
            ]
        elif kwargs.get("return_dict", True):
            return SamplesDict(obj.parameters, np.array(obj.samples).T)
        elif return_kwargs:
            return obj.parameters, obj.samples, obj.extra_kwargs
        else:
            return obj.parameters, obj.samples

    def __init__(
        self, parameters, samples, extra_kwargs, evolve_spins, NRSurrogate,
        waveform_fits, multi_process, regenerate, redshift_method,
        cosmology, force_non_evolved, force_remnant, add_zero_spin
    ):
        self.parameters = parameters
        self.samples = samples
        self.extra_kwargs = extra_kwargs
        self.NRSurrogate = NRSurrogate
        self.waveform_fit = waveform_fits
        self.multi_process = multi_process
        self.regenerate = regenerate
        self.redshift_method = redshift_method
        self.cosmology = cosmology
        self.force_non_evolved = force_non_evolved
        self.non_precessing = False
        if not any(param in self.parameters for param in conf.precessing_angles):
            self.non_precessing = True
        if self.non_precessing and evolve_spins:
            logger.info(
                "Spin evolution is trivial for a non-precessing system. No additional "
                "transformation required."
            )
            evolve_spins = False
        self.has_tidal = self._check_for_tidal_parameters()
        self.compute_remnant = True
        if force_remnant and self.has_tidal:
            logger.warn(
                "Posterior samples for tidal deformability found in the "
                "posterior table. Applying BBH remnant fits to this system. "
                "This may not give sensible results."
            )
        elif self.has_tidal:
            if evolve_spins:
                msg = (
                    "Not applying spin evolution as tidal parameters found "
                    "in the posterior table."
                )
                logger.info(msg)
            logger.debug(
                "Skipping remnant calculations as tidal deformability "
                "parameters found in the posterior table."
            )
            self.compute_remnant = False
        if self.regenerate is not None:
            for param in self.regenerate:
                self.remove_posterior(param)
        self.add_zero_spin = add_zero_spin
        self.generate_all_posterior_samples(evolve_spins=evolve_spins)

    def _check_for_tidal_parameters(self):
        """Check to see if any tidal parameters are stored in the table
        """
        from pesummary.gw.file.standard_names import tidal_params

        if any(param in self.parameters for param in tidal_params):
            return True
        return False

    def remove_posterior(self, parameter):
        if parameter in self.parameters:
            logger.info(
                "Removing the posterior samples for '{}'".format(parameter)
            )
            ind = self.parameters.index(parameter)
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        else:
            logger.info(
                "'{}' is not in the table of posterior samples. Unable to "
                "remove".format(parameter)
            )

    def _specific_parameter_samples(self, param):
        """Return the samples for a specific parameter

        Parameters
        ----------
        param: str
            the parameter that you would like to return the samples for
        """
        if param == "empty":
            return np.array(np.zeros(len(self.samples)))
        ind = self.parameters.index(param)
        samples = np.array([i[ind] for i in self.samples])
        return samples

    def specific_parameter_samples(self, param):
        """Return the samples for either a list or a single parameter

        Parameters
        ----------
        param: list/str
            the parameter/parameters that you would like to return the samples
            for
        """
        if type(param) == list:
            samples = [self._specific_parameter_samples(i) for i in param]
        else:
            samples = self._specific_parameter_samples(param)
        return samples

    def append_data(self, parameter, samples):
        """Add a list of samples to the existing samples data object

        Parameters
        ----------
        parameter: str
            the name of the parameter you would like to append
        samples: list
            the list of samples that you would like to append
        """
        if parameter not in self.parameters:
            self.parameters.append(parameter)
            for num, i in enumerate(self.samples):
                self.samples[num].append(samples[num])

    def _mchirp_from_mchirp_source_z(self):
        samples = self.specific_parameter_samples(["chirp_mass_source", "redshift"])
        chirp_mass = mchirp_from_mchirp_source_z(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _q_from_eta(self):
        samples = self.specific_parameter_samples("symmetric_mass_ratio")
        mass_ratio = q_from_eta(samples)
        self.append_data("mass_ratio", mass_ratio)

    def _q_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        mass_ratio = q_from_m1_m2(samples[0], samples[1])
        self.append_data("mass_ratio", mass_ratio)

    def _invert_q(self):
        ind = self.parameters.index("mass_ratio")
        for num, i in enumerate(self.samples):
            self.samples[num][ind] = 1. / self.samples[num][ind]

    def _invq_from_q(self):
        samples = self.specific_parameter_samples("mass_ratio")
        inverted_mass_ratio = invq_from_q(samples)
        self.append_data("inverted_mass_ratio", inverted_mass_ratio)

    def _mchirp_from_mtotal_q(self):
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        chirp_mass = mchirp_from_mtotal_q(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _m1_from_mchirp_q(self):
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_1 = m1_from_mchirp_q(samples[0], samples[1])
        self.append_data("mass_1", mass_1)

    def _m2_from_mchirp_q(self):
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_2 = m2_from_mchirp_q(samples[0], samples[1])
        self.append_data("mass_2", mass_2)

    def _reference_frequency(self):
        nsamples = len(self.samples)
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "f_ref" in list(extra_kwargs.keys()):
            self.append_data(
                "reference_frequency", [float(extra_kwargs["f_ref"])] * nsamples
            )
        else:
            logger.warn(
                "Could not find reference_frequency in input file. Using 20Hz "
                "as default")
            self.append_data("reference_frequency", [20.] * nsamples)

    def _mtotal_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        m_total = m_total_from_m1_m2(samples[0], samples[1])
        self.append_data("total_mass", m_total)

    def _mchirp_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        chirp_mass = mchirp_from_m1_m2(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _eta_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        eta = eta_from_m1_m2(samples[0], samples[1])
        self.append_data("symmetric_mass_ratio", eta)

    def _phi_12_from_phi1_phi2(self):
        samples = self.specific_parameter_samples(["phi_1", "phi_2"])
        phi_12 = phi_12_from_phi1_phi2(samples[0], samples[1])
        self.append_data("phi_12", phi_12)

    def _phi1_from_spins(self):
        samples = self.specific_parameter_samples(["spin_1x", "spin_1y"])
        phi_1 = phi1_from_spins(samples[0], samples[1])
        self.append_data("phi_1", phi_1)

    def _phi2_from_spins(self):
        samples = self.specific_parameter_samples(["spin_2x", "spin_2y"])
        phi_2 = phi2_from_spins(samples[0], samples[1])
        self.append_data("phi_2", phi_2)

    def _spin_angles(self):
        angles = ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12",
                  "a_1", "a_2"]
        spin_angles_to_calculate = [
            i for i in angles if i not in self.parameters]
        spin_components = [
            "mass_1", "mass_2", "iota", "spin_1x", "spin_1y", "spin_1z",
            "spin_2x", "spin_2y", "spin_2z", "reference_frequency"]
        samples = self.specific_parameter_samples(spin_components)
        if "phase" in self.parameters:
            spin_components.append("phase")
            samples.append(self.specific_parameter_samples("phase"))
        else:
            logger.warn("Phase it not given, we will be assuming that a "
                        "reference phase of 0 to calculate all the spin angles")
            samples.append([0] * len(samples[0]))
        angles = spin_angles(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spin_angles_to_calculate:
            ind = spin_angles_to_calculate.index(i)
            data = np.array([i[ind] for i in angles])
            self.append_data(i, data)

    def _non_precessing_component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        angles = ["a_1", "a_2", "theta_jn", "tilt_1", "tilt_2"]
        if all(i in self.parameters for i in angles):
            samples = self.specific_parameter_samples(angles)
            cond1 = all(i in [0, np.pi] for i in samples[3])
            cond2 = all(i in [0, np.pi] for i in samples[4])
            spins_to_calculate = [
                i for i in spins if i not in self.parameters]
            if cond1 and cond1:
                spin_1x = np.array([0.] * len(samples[0]))
                spin_1y = np.array([0.] * len(samples[0]))
                spin_1z = samples[0] * np.cos(samples[3])
                spin_2x = np.array([0.] * len(samples[0]))
                spin_2y = np.array([0.] * len(samples[0]))
                spin_2z = samples[1] * np.cos(samples[4])
                iota = np.array(samples[2])
                spin_components = [
                    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z]

                for i in spins_to_calculate:
                    ind = spins.index(i)
                    data = spin_components[ind]
                    self.append_data(i, data)

    def _component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        angles = [
            "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
            "mass_1", "mass_2", "reference_frequency"]
        samples = self.specific_parameter_samples(angles)
        if "phase" in self.parameters:
            angles.append("phase")
            samples.append(self.specific_parameter_samples("phase"))
        else:
            logger.warn("Phase it not given, we will be assuming that a "
                        "reference phase of 0 to calculate all the spin angles")
            samples.append([0] * len(samples[0]))
        spin_components = component_spins(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spins_to_calculate:
            ind = spins.index(i)
            data = np.array([i[ind] for i in spin_components])
            self.append_data(i, data)

    def _component_spins_from_azimuthal_and_polar_angles(self):
        spins = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        angles = [
            "a_1", "a_2", "a_1_azimuthal", "a_1_polar", "a_2_azimuthal",
            "a_2_polar"]
        samples = self.specific_parameter_samples(angles)
        spin_components = spin_angles_from_azimuthal_and_polar_angles(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        for i in spins_to_calculate:
            ind = spins.index(i)
            data = np.array([i[ind] for i in spin_components])
            self.append_data(i, data)

    def _chi_p(self):
        parameters = [
            "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]
        samples = self.specific_parameter_samples(parameters)
        chi_p_samples = chi_p(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        self.append_data("chi_p", chi_p_samples)

    def _chi_eff(self):
        parameters = ["mass_1", "mass_2", "spin_1z", "spin_2z"]
        samples = self.specific_parameter_samples(parameters)
        chi_eff_samples = chi_eff(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("chi_eff", chi_eff_samples)

    def _cos_tilt_1_from_tilt_1(self):
        samples = self.specific_parameter_samples("tilt_1")
        cos_tilt_1 = np.cos(samples)
        self.append_data("cos_tilt_1", cos_tilt_1)

    def _cos_tilt_2_from_tilt_2(self):
        samples = self.specific_parameter_samples("tilt_2")
        cos_tilt_2 = np.cos(samples)
        self.append_data("cos_tilt_2", cos_tilt_2)

    def _dL_from_z(self):
        samples = self.specific_parameter_samples("redshift")
        distance = dL_from_z(samples, cosmology=self.cosmology)
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("luminosity_distance", distance)

    def _z_from_dL(self):
        samples = self.specific_parameter_samples("luminosity_distance")
        func = getattr(_Redshift, self.redshift_method)
        redshift = func(
            samples, cosmology=self.cosmology, multi_process=self.multi_process
        )
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("redshift", redshift)

    def _comoving_distance_from_z(self):
        samples = self.specific_parameter_samples("redshift")
        distance = comoving_distance_from_z(samples, cosmology=self.cosmology)
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("comoving_distance", distance)

    def _m1_source_from_m1_z(self):
        samples = self.specific_parameter_samples(["mass_1", "redshift"])
        mass_1_source = m1_source_from_m1_z(samples[0], samples[1])
        self.append_data("mass_1_source", mass_1_source)

    def _m2_source_from_m2_z(self):
        samples = self.specific_parameter_samples(["mass_2", "redshift"])
        mass_2_source = m2_source_from_m2_z(samples[0], samples[1])
        self.append_data("mass_2_source", mass_2_source)

    def _mtotal_source_from_mtotal_z(self):
        samples = self.specific_parameter_samples(["total_mass", "redshift"])
        total_mass_source = m_total_source_from_mtotal_z(samples[0], samples[1])
        self.append_data("total_mass_source", total_mass_source)

    def _mchirp_source_from_mchirp_z(self):
        samples = self.specific_parameter_samples(["chirp_mass", "redshift"])
        chirp_mass_source = mchirp_source_from_mchirp_z(samples[0], samples[1])
        self.append_data("chirp_mass_source", chirp_mass_source)

    def _time_in_each_ifo(self):
        detectors = []
        if "IFOs" in list(self.extra_kwargs["meta_data"].keys()):
            detectors = self.extra_kwargs["meta_data"]["IFOs"].split(" ")
        else:
            for i in self.parameters:
                if "optimal_snr" in i and i != "network_optimal_snr":
                    det = i.split("_optimal_snr")[0]
                    detectors.append(det)

        samples = self.specific_parameter_samples(["ra", "dec", "geocent_time"])
        for i in detectors:
            time = time_in_each_ifo(i, samples[0], samples[1], samples[2])
            self.append_data("%s_time" % (i), time)

    def _lambda1_from_lambda_tilde(self):
        samples = self.specific_parameter_samples([
            "lambda_tilde", "mass_1", "mass_2"])
        lambda_1 = lambda1_from_lambda_tilde(samples[0], samples[1], samples[2])
        self.append_data("lambda_1", lambda_1)

    def _lambda2_from_lambda1(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "mass_1", "mass_2"])
        lambda_2 = lambda2_from_lambda1(samples[0], samples[1], samples[2])
        self.append_data("lambda_2", lambda_2)

    def _lambda_tilde_from_lambda1_lambda2(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        lambda_tilde = lambda_tilde_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("lambda_tilde", lambda_tilde)

    def _delta_lambda_from_lambda1_lambda2(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        delta_lambda = delta_lambda_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("delta_lambda", delta_lambda)

    def _ifo_snr(self):
        abs_snrs = [
            i for i in self.parameters if "_matched_filter_abs_snr" in i
        ]
        angle_snrs = [
            i for i in self.parameters if "_matched_filter_snr_angle" in i
        ]
        for ifo in [snr.split("_matched_filter_abs_snr")[0] for snr in abs_snrs]:
            if "{}_matched_filter_snr".format(ifo) not in self.parameters:
                samples = self.specific_parameter_samples(
                    [
                        "{}_matched_filter_abs_snr".format(ifo),
                        "{}_matched_filter_snr_angle".format(ifo)
                    ]
                )
                snr = _ifo_snr(samples[0], samples[1])
                self.append_data("{}_matched_filter_snr".format(ifo), snr)

    def _optimal_network_snr(self):
        snrs = [i for i in self.parameters if "_optimal_snr" in i]
        samples = self.specific_parameter_samples(snrs)
        snr = network_snr(samples)
        self.append_data("network_optimal_snr", snr)

    def _matched_filter_network_snr(self):
        snrs = [
            i for i in self.parameters if "_matched_filter_snr" in i
            and "_angle" not in i and "_abs" not in i
        ]
        if len(snrs) == 0:
            snrs = [
                i for i in self.parameters if "_matched_filter_snr_abs" in i
                and "_angle" not in i
            ]
        samples = self.specific_parameter_samples(snrs)
        snr = network_snr(samples)
        self.append_data("network_matched_filter_snr", snr)

    def _retrieve_f_low(self):
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "f_low" in list(extra_kwargs.keys()):
            f_low = extra_kwargs["f_low"]
        else:
            raise ValueError(
                "Could not find f_low in input file. Please either modify the "
                "input file or pass it from the command line"
            )
        return f_low

    def _retrieve_approximant(self):
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "approximant" in list(extra_kwargs.keys()):
            approximant = extra_kwargs["approximant"]
        else:
            raise ValueError(
                "Unable to find the approximant used to generate the posterior "
                "samples in the result file."
            )
        return approximant

    def _evolve_spins(self, final_velocity="ISCO"):
        from pesummary.gw.file.evolve import evolve_spins

        f_low = self._retrieve_f_low()
        approximant = self._retrieve_approximant()
        parameters = ["tilt_1", "tilt_2", "phi_12", "spin_1z", "spin_2z"]
        samples = self.specific_parameter_samples(
            ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2",
             "phi_12", "reference_frequency"]
        )
        tilt_1_evolved, tilt_2_evolved, phi_12_evolved = evolve_spins(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], f_low, samples[7][0],
            approximant, final_velocity=final_velocity,
            multi_process=self.multi_process
        )
        spin_1z_evolved = samples[2] * np.cos(tilt_1_evolved)
        spin_2z_evolved = samples[3] * np.cos(tilt_2_evolved)
        self.append_data("tilt_1_evolved", tilt_1_evolved)
        self.append_data("tilt_2_evolved", tilt_2_evolved)
        self.append_data("phi_12_evolved", phi_12_evolved)
        self.append_data("spin_1z_evolved", spin_1z_evolved)
        self.append_data("spin_2z_evolved", spin_2z_evolved)

    @staticmethod
    def _evolved_vs_non_evolved_parameter(
        parameter, evolved=False, core_param=False, non_precessing=False
    ):
        if non_precessing:
            base_string = ""
        elif evolved and core_param:
            base_string = "_evolved"
        elif evolved:
            base_string = ""
        elif core_param:
            base_string = ""
        else:
            base_string = "_non_evolved"
        return "{}{}".format(parameter, base_string)

    def _precessing_vs_non_precessing_parameters(
        self, non_precessing=False, evolved=False
    ):
        if not non_precessing:
            tilt_1 = self._evolved_vs_non_evolved_parameter(
                "tilt_1", evolved=evolved, core_param=True
            )
            tilt_2 = self._evolved_vs_non_evolved_parameter(
                "tilt_2", evolved=evolved, core_param=True
            )
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", "a_1", "a_2", tilt_1, tilt_2
            ])
            if "phi_12" in self.parameters and evolved:
                phi_12_samples = self.specific_parameter_samples([
                    self._evolved_vs_non_evolved_parameter(
                        "phi_12", evolved=True, core_param=True
                    )
                ])[0]
            elif "phi_12" in self.parameters:
                phi_12_samples = self.specific_parameter_samples(["phi_12"])[0]
            else:
                phi_12_samples = np.zeros_like(samples[0])
            samples.append(phi_12_samples)
            if self.NRSurrogate:
                NRSurrogate_samples = self.specific_parameter_samples([
                    "phi_jl", "theta_jn", "phase"
                ])
                for ss in NRSurrogate_samples:
                    samples.append(ss)
        else:
            spin_1z = self._evolved_vs_non_evolved_parameter(
                "spin_1z", evolved=evolved, core_param=True, non_precessing=True
            )
            spin_2z = self._evolved_vs_non_evolved_parameter(
                "spin_2z", evolved=evolved, core_param=True, non_precessing=True
            )
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", spin_1z, spin_2z
            ])
            samples = [
                samples[0], samples[1], np.abs(samples[2]), np.abs(samples[3]),
                0.5 * np.pi * (1 - np.sign(samples[2])),
                0.5 * np.pi * (1 - np.sign(samples[3])),
                np.zeros_like(samples[0])
            ]
        return samples

    def _peak_luminosity_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "peak_luminosity", evolved=evolved, non_precessing=self.non_precessing
        )
        spin_1z_param = self._evolved_vs_non_evolved_parameter(
            "spin_1z", evolved=evolved, core_param=True, non_precessing=self.non_precessing
        )
        spin_2z_param = self._evolved_vs_non_evolved_parameter(
            "spin_2z", evolved=evolved, core_param=True, non_precessing=self.non_precessing
        )

        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", spin_1z_param, spin_2z_param
        ])
        peak_luminosity, fits = peak_luminosity_of_merger(
            samples[0], samples[1], samples[2], samples[3],
            return_fits_used=True
        )
        self.append_data(param, peak_luminosity)
        self.extra_kwargs["meta_data"]["peak_luminosity_NR_fits"] = fits

    def _final_remnant_properties_from_NRSurrogate(
        self, non_precessing=False, parameters=["final_mass", "final_spin", "final_kick"]
    ):
        f_low = self._retrieve_f_low()
        approximant = self._retrieve_approximant()
        samples = self._precessing_vs_non_precessing_parameters(
            non_precessing=non_precessing, evolved=False
        )
        frequency_samples = self.specific_parameter_samples([
            "reference_frequency"
        ])
        data, fits = final_remnant_properties_from_NRSurrogate(
            *samples, f_low=f_low, f_ref=frequency_samples[0],
            properties=parameters, return_fits_used=True,
            approximant=approximant
        )
        for param in parameters:
            self.append_data(param, data[param])
            self.extra_kwargs["meta_data"]["{}_NR_fits".format(param)] = fits

    def _final_remnant_properties_from_waveform(
        self, non_precessing=False, parameters=["final_mass", "final_spin"]
    ):
        f_low = self._retrieve_f_low()
        approximant = self._retrieve_approximant()
        if "delta_t" in self.extra_kwargs["meta_data"].keys():
            delta_t = self.extra_kwargs["meta_data"]["delta_t"]
        else:
            delta_t = 1. / 4096
            if "seob" in approximant.lower():
                logger.warn(
                    "Could not find 'delta_t' in the meta data. Using {} as "
                    "default.".format(delta_t)
                )
        if non_precessing:
            sample_params = [
                "mass_1", "mass_2", "empty", "empty", "spin_1z", "empty",
                "empty", "spin_2z", "iota", "luminosity_distance",
                "phase"
            ]
        else:
            sample_params = [
                "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z",
                "spin_2x", "spin_2y", "spin_2z", "iota", "luminosity_distance",
                "phase"
            ]
        samples = self.specific_parameter_samples(sample_params)
        ind = self.parameters.index("spin_1x")
        _data, fits = _final_from_initial(
            *samples[:8], iota=samples[8], luminosity_distance=samples[9],
            f_ref=[f_low] * len(samples[0]), phi_ref=samples[10],
            delta_t=1. / 4096, approximant=approximant, return_fits_used=True,
            multi_process=self.multi_process
        )
        data = {"final_mass": _data[0], "final_spin": _data[1]}
        for param in parameters:
            self.append_data(param, data[param])
            self.extra_kwargs["meta_data"]["{}_NR_fits".format(param)] = fits

    def _final_mass_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved, non_precessing=self.non_precessing
        )
        spin_1z_param = self._evolved_vs_non_evolved_parameter(
            "spin_1z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )
        spin_2z_param = self._evolved_vs_non_evolved_parameter(
            "spin_2z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", spin_1z_param, spin_2z_param
        ])
        final_mass, fits = final_mass_of_merger(
            *samples, return_fits_used=True
        )
        self.append_data(param, final_mass)
        self.extra_kwargs["meta_data"]["final_mass_NR_fits"] = fits

    def _final_mass_source(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([param, "redshift"])
        final_mass_source = _source_from_detector(
            samples[0], samples[1]
        )
        self.append_data(param.replace("mass", "mass_source"), final_mass_source)

    def _final_spin_of_merger(self, non_precessing=False, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_spin", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self._precessing_vs_non_precessing_parameters(
            non_precessing=non_precessing, evolved=evolved
        )
        final_spin, fits = final_spin_of_merger(
            *samples, return_fits_used=True
        )
        self.append_data(param, final_spin)
        self.extra_kwargs["meta_data"]["final_spin_NR_fits"] = fits

    def _radiated_energy(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "radiated_energy", evolved=evolved, non_precessing=self.non_precessing
        )
        final_mass_param = self._evolved_vs_non_evolved_parameter(
            "final_mass_source", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([
            "total_mass_source", final_mass_param
        ])
        radiated_energy = samples[0] - samples[1]
        self.append_data(param, radiated_energy)

    def _cos_angle(self, parameter_to_add, reverse=False):
        if reverse:
            samples = self.specific_parameter_samples(
                ["cos_" + parameter_to_add])
            cos_samples = np.arccos(samples[0])
        else:
            samples = self.specific_parameter_samples(
                [parameter_to_add.split("cos_")[1]]
            )
            cos_samples = np.cos(samples[0])
        self.append_data(parameter_to_add, cos_samples)

    def _check_parameters(self):
        params = ["mass_1", "mass_2", "a_1", "a_2", "mass_1_source", "mass_2_source",
                  "mass_ratio", "total_mass", "chirp_mass"]
        for i in params:
            if i in self.parameters:
                samples = self.specific_parameter_samples([i])
                if "mass" in i:
                    cond = any(np.array(samples[0]) <= 0.)
                else:
                    cond = any(np.array(samples[0]) < 0.)
                if cond:
                    if "mass" in i:
                        ind = np.argwhere(np.array(samples[0]) <= 0.)
                    else:
                        ind = np.argwhere(np.array(samples[0]) < 0.)
                    logger.warn("Removing %s samples because they have unphysical "
                                "values (%s < 0)" % (len(ind), i))
                    for i in np.arange(len(ind) - 1, -1, -1):
                        self.samples.remove(list(np.array(self.samples)[ind[i][0]]))

    def generate_all_posterior_samples(self, evolve_spins=False):
        logger.debug("Starting to generate all derived posteriors")
        evolve_condition = (
            True if evolve_spins and self.compute_remnant else False
        )
        if "cos_theta_jn" in self.parameters and "theta_jn" not in self.parameters:
            self._cos_angle("theta_jn", reverse=True)
        if "cos_iota" in self.parameters and "iota" not in self.parameters:
            self._cos_angle("iota", reverse=True)
        if "cos_tilt_1" in self.parameters and "tilt_1" not in self.parameters:
            self._cos_angle("tilt_1", reverse=True)
        if "cos_tilt_2" in self.parameters and "tilt_2" not in self.parameters:
            self._cos_angle("tilt_2", reverse=True)
        spin_magnitudes = ["a_1", "a_2"]
        angles = ["phi_jl", "tilt_1", "tilt_2", "phi_12"]
        cartesian = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
        cond1 = all(i in self.parameters for i in spin_magnitudes)
        cond2 = all(i in self.parameters for i in angles)
        cond3 = all(i in self.parameters for i in cartesian)
        if cond1 and not cond2:
            self.parameters.append("tilt_1")
            self.parameters.append("tilt_2")
            for num, i in enumerate(self.samples):
                self.samples[num].append(
                    np.arccos(np.sign(i[self.parameters.index("a_1")])))
                self.samples[num].append(
                    np.arccos(np.sign(i[self.parameters.index("a_2")])))
            ind_a1 = self.parameters.index("a_1")
            ind_a2 = self.parameters.index("a_2")
            for num, i in enumerate(self.samples):
                self.samples[num][ind_a1] = abs(self.samples[num][ind_a1])
                self.samples[num][ind_a2] = abs(self.samples[num][ind_a2])
        elif not cond1 and not cond2 and not cond3 and self.add_zero_spin:
            parameters = ["a_1", "a_2", "spin_1z", "spin_2z"]
            for param in parameters:
                self.parameters.append(param)
                for num, i in enumerate(self.samples):
                    self.samples[num].append(0)
        self._check_parameters()
        if "cos_theta_jn" in self.parameters and "theta_jn" not in self.parameters:
            self._cos_angle("theta_jn", reverse=True)
        if "cos_iota" in self.parameters and "iota" not in self.parameters:
            self._cos_angle("iota", reverse=True)
        if "cos_tilt_1" in self.parameters and "tilt_1" not in self.parameters:
            self._cos_angle("tilt_1", reverse=True)
        if "cos_tilt_2" in self.parameters and "tilt_2" not in self.parameters:
            self._cos_angle("tilt_2", reverse=True)
        if "chirp_mass" not in self.parameters and "chirp_mass_source" in \
                self.parameters and "redshift" in self.parameters:
            self._mchirp_from_mchirp_source_z()
        if "mass_ratio" not in self.parameters and "symmetric_mass_ratio" in \
                self.parameters:
            self._q_from_eta()
        if "mass_ratio" not in self.parameters and "mass_1" in self.parameters \
                and "mass_2" in self.parameters:
            self._q_from_m1_m2()
        if "mass_ratio" in self.parameters:
            ind = self.parameters.index("mass_ratio")
            median = np.median([i[ind] for i in self.samples])
            if median > 1.:
                self._invert_q()
        if "inverted_mass_ratio" not in self.parameters and "mass_ratio" in \
                self.parameters:
            self._invq_from_q()
        if "chirp_mass" not in self.parameters and "total_mass" in self.parameters:
            self._mchirp_from_mtotal_q()
        if "mass_1" not in self.parameters and "chirp_mass" in self.parameters:
            self._m1_from_mchirp_q()
        if "mass_2" not in self.parameters and "chirp_mass" in self.parameters:
            self._m2_from_mchirp_q()
        if "reference_frequency" not in self.parameters:
            self._reference_frequency()
        condition1 = "phi_12" not in self.parameters
        condition2 = "phi_1" in self.parameters and "phi_2" in self.parameters
        if condition1 and condition2:
            self._phi_12_from_phi1_phi2()
        angles = [
            "a_1", "a_2", "a_1_azimuthal", "a_1_polar", "a_2_azimuthal",
            "a_2_polar"]
        if all(i in self.parameters for i in angles):
            self._component_spins_from_azimuthal_and_polar_angles()
        if "mass_1" in self.parameters and "mass_2" in self.parameters:
            if "total_mass" not in self.parameters:
                self._mtotal_from_m1_m2()
            if "chirp_mass" not in self.parameters:
                self._mchirp_from_m1_m2()
            if "symmetric_mass_ratio" not in self.parameters:
                self._eta_from_m1_m2()
            spin_components = [
                "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z",
                "iota"
            ]
            angles = ["a_1", "a_2", "tilt_1", "tilt_2", "theta_jn"]
            if all(i in self.parameters for i in spin_components):
                self._spin_angles()
            if all(i in self.parameters for i in angles):
                samples = self.specific_parameter_samples(["tilt_1", "tilt_2"])
                cond1 = all(i in [0, np.pi] for i in samples[0])
                cond2 = all(i in [0, np.pi] for i in samples[1])
                if cond1 and cond1:
                    self._non_precessing_component_spins()
                else:
                    angles = [
                        "phi_jl", "phi_12", "reference_frequency"]
                    if all(i in self.parameters for i in angles):
                        self._component_spins()
            cond1 = "spin_1x" in self.parameters and "spin_1y" in self.parameters
            if "phi_1" not in self.parameters and cond1:
                self._phi1_from_spins()
            cond1 = "spin_2x" in self.parameters and "spin_2y" in self.parameters
            if "phi_2" not in self.parameters and cond1:
                self._phi2_from_spins()
            if "chi_eff" not in self.parameters:
                if all(i in self.parameters for i in spin_components):
                    self._chi_eff()
            if "chi_p" not in self.parameters:
                if all(i in self.parameters for i in spin_components):
                    self._chi_p()
            if "lambda_tilde" in self.parameters and "lambda_1" not in self.parameters:
                self._lambda1_from_lambda_tilde()
            if "lambda_2" not in self.parameters and "lambda_1" in self.parameters:
                self._lambda2_from_lambda1()
            if "lambda_1" in self.parameters and "lambda_2" in self.parameters:
                if "lambda_tilde" not in self.parameters:
                    self._lambda_tilde_from_lambda1_lambda2()
                if "delta_lambda" not in self.parameters:
                    self._delta_lambda_from_lambda1_lambda2()

            evolve_suffix = "_non_evolved"
            final_spin_params = ["a_1", "a_2"]
            non_precessing_NR_params = ["spin_1z", "spin_2z"]
            if evolve_condition:
                final_spin_params += [
                    "tilt_1_evolved", "tilt_2_evolved", "phi_12_evolved"
                ]
                non_precessing_NR_params = [
                    "{}_evolved".format(i) for i in non_precessing_NR_params
                ]
                evolve_suffix = "_evolved"
                evolve_spins_params = ["tilt_1", "tilt_2", "phi_12"]
                if all(i in self.parameters for i in evolve_spins_params):
                    self._evolve_spins(final_velocity=evolve_spins)
                else:
                    evolve_condition = False
            else:
                final_spin_params += ["tilt_1", "tilt_2", "phi_12"]

            check_for_evolved_parameter = lambda suffix, param, params: (
                param not in params and param + suffix not in params if
                len(suffix) else param not in params
            )
            condition_peak_luminosity = check_for_evolved_parameter(
                evolve_suffix, "peak_luminosity", self.parameters
            )
            condition_final_spin = check_for_evolved_parameter(
                evolve_suffix, "final_spin", self.parameters
            )
            condition_final_mass = check_for_evolved_parameter(
                evolve_suffix, "final_mass", self.parameters
            )
            if (self.NRSurrogate or self.waveform_fit) and self.compute_remnant:
                parameters = []
                _default = ["final_mass", "final_spin"]
                if self.NRSurrogate:
                    _default.append("final_kick")
                    function = self._final_remnant_properties_from_NRSurrogate
                else:
                    final_spin_params = [
                        "spin_1x", "spin_1y", "spin_1z", "spin_2x",
                        "spin_2y", "spin_2z"
                    ]
                    function = self._final_remnant_properties_from_waveform

                for param in _default:
                    if param not in self.parameters:
                        parameters.append(param)
                if all(i in self.parameters for i in final_spin_params):
                    function(non_precessing=False, parameters=parameters)
                elif all(i in self.parameters for i in non_precessing_NR_params):
                    function(non_precessing=True, parameters=parameters)
                if all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_peak_luminosity or self.force_non_evolved:
                        self._peak_luminosity_of_merger(evolved=evolve_condition)
            elif self.compute_remnant:
                if all(i in self.parameters for i in final_spin_params):
                    if condition_final_spin or self.force_non_evolved:
                        self._final_spin_of_merger(evolved=evolve_condition)
                elif all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_final_spin or self.force_non_evolved:
                        self._final_spin_of_merger(
                            non_precessing=True, evolved=False
                        )
                if all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_peak_luminosity or self.force_non_evolved:
                        self._peak_luminosity_of_merger(evolved=evolve_condition)
                    if condition_final_mass or self.force_non_evolved:
                        self._final_mass_of_merger(evolved=evolve_condition)
        if "cos_tilt_1" not in self.parameters and "tilt_1" in self.parameters:
            self._cos_tilt_1_from_tilt_1()
        if "cos_tilt_2" not in self.parameters and "tilt_2" in self.parameters:
            self._cos_tilt_2_from_tilt_2()
        if "luminosity_distance" not in self.parameters and "redshift" in self.parameters:
            self._dL_from_z()
        if "redshift" not in self.parameters and "luminosity_distance" in self.parameters:
            self._z_from_dL()
        if "comoving_distance" not in self.parameters and "redshift" in self.parameters:
            self._comoving_distance_from_z()

        evolve_suffix = "_non_evolved"
        if evolve_condition or self.NRSurrogate or self.waveform_fit or self.non_precessing:
            evolve_suffix = ""
            evolve_condition = True
        if "redshift" in self.parameters:
            if "mass_1_source" not in self.parameters and "mass_1" in self.parameters:
                self._m1_source_from_m1_z()
            if "mass_2_source" not in self.parameters and "mass_2" in self.parameters:
                self._m2_source_from_m2_z()
            if "total_mass_source" not in self.parameters and "total_mass" in self.parameters:
                self._mtotal_source_from_mtotal_z()
            if "chirp_mass_source" not in self.parameters and "chirp_mass" in self.parameters:
                self._mchirp_source_from_mchirp_z()
            condition_final_mass_source = check_for_evolved_parameter(
                evolve_suffix, "final_mass_source", self.parameters
            )
            if condition_final_mass_source or self.force_non_evolved:
                if "final_mass{}".format(evolve_suffix) in self.parameters:
                    self._final_mass_source(evolved=evolve_condition)
        if "total_mass_source" in self.parameters:
            if "final_mass_source{}".format(evolve_suffix) in self.parameters:
                condition_radiated_energy = check_for_evolved_parameter(
                    evolve_suffix, "radiated_energy", self.parameters
                )
                if condition_radiated_energy or self.force_non_evolved:
                    self._radiated_energy(evolved=evolve_condition)
        location = ["geocent_time", "ra", "dec"]
        if all(i in self.parameters for i in location):
            try:
                self._time_in_each_ifo()
            except Exception as e:
                logger.warn("Failed to generate posterior samples for the time in each "
                            "detector because %s" % (e))
        if any("_matched_filter_snr_angle" in i for i in self.parameters):
            if any("_matched_filter_abs_snr" in i for i in self.parameters):
                self._ifo_snr()
        if any("_optimal_snr" in i for i in self.parameters):
            if "network_optimal_snr" not in self.parameters:
                self._optimal_network_snr()
        if any("_matched_filter_snr" in i for i in self.parameters):
            if "network_matched_filter_snr" not in self.parameters:
                self._matched_filter_network_snr()
        if "theta_jn" in self.parameters and "cos_theta_jn" not in self.parameters:
            self._cos_angle("cos_theta_jn")
        if "iota" in self.parameters and "cos_iota" not in self.parameters:
            self._cos_angle("cos_iota")
        remove_parameters = [
            "tilt_1_evolved", "tilt_2_evolved", "phi_12_evolved",
            "spin_1z_evolved", "spin_2z_evolved", "reference_frequency",
            "minimum_frequency"
        ]
        for param in remove_parameters:
            if param in self.parameters:
                ind = self.parameters.index(param)
                self.parameters.remove(self.parameters[ind])
                for i in self.samples:
                    del i[ind]
