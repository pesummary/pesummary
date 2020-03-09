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

from pesummary.utils.utils import logger, SamplesDict

try:
    import lalsimulation
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
                   "Bear in mind that this does introduce a small error.")
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


def m1_source_from_m1_z(mass_1, z):
    """Return the source mass of the bigger black hole given samples for the
    detector mass of the bigger black hole and the redshift
    """
    return _source_from_detector(mass_1, z)


def m2_source_from_m2_z(mass_2, z):
    """Return the source mass of the smaller black hole given samples for the
    detector mass of the smaller black hole and the redshift
    """
    return _source_from_detector(mass_2, z)


def m_total_source_from_mtotal_z(total_mass, z):
    """Return the source total mass of the binary given samples for detector
    total mass and redshift
    """
    return _source_from_detector(total_mass, z)


def mtotal_from_mtotal_source_z(total_mass_source, z):
    """Return the total mass of the binary given samples for the source total
    mass and redshift
    """
    return _detector_from_source(total_mass_source, z)


def mchirp_source_from_mchirp_z(mchirp, z):
    """Return the source chirp mass of the binary given samples for detector
    chirp mass and redshift
    """
    return _source_from_detector(mchirp, z)


def mchirp_from_mchirp_source_z(mchirp_source, z):
    """Return the chirp mass of the binary given samples for the source chirp
    mass and redshift
    """
    return _detector_from_source(mchirp_source, z)


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


def invq_from_m1_m2(mass1, mass2):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass1 and mass2
    """
    return 1. / q_from_m1_m2(mass1, mass2)


def invq_from_q(mass_ratio):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass ratio (mass2/mass1)
    """
    return 1. / mass_ratio


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


def phi1_from_spins(spin_1x, spin_1y):
    """Return phi_1 given samples for spin_1x and spin_1y
    """
    phi_1 = np.fmod(2 * np.pi + np.arctan2(spin_1y, spin_1x), 2 * np.pi)
    return phi_1


def phi2_from_spins(spin_2x, spin_2y):
    """Return phi_2 given samples for spin_2x and spin_2y
    """
    phi_2 = np.fmod(2 * np.pi + np.arctan2(spin_2y, spin_2x), 2 * np.pi)
    return phi_2


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


def _ifo_snr(IFO_abs_snr, IFO_snr_angle):
    """Return the matched filter SNR for a given IFO given samples for the
    absolute SNR and the angle
    """
    return IFO_abs_snr * np.cos(IFO_snr_angle)


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


def _final_from_initial(
    mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
    approximant="SEOBNRv4"
):
    """Calculate the final mass and final spin given the initial parameters
    of the binary without using the EOB spin evolution in precessing cases
    """
    from lalsimulation import (
        SimIMREOBFinalMassSpin, SimIMREOBFinalMassSpinPrec,
        SimInspiralGetSpinSupportFromApproximant
    )

    ALLOWED = ["SEOBNRv4", "SEOBNRv4P", "SEOBNRv4PHM"]

    try:
        approx = getattr(lalsimulation, approximant)
    except AttributeError:
        raise ValueError(
            "The waveform '{}' is not supported by lalsimulation"
        )

    if approximant not in ALLOWED:
        raise ValueError(
            "The waveform '{}' is not support by this function. The list "
            "of supported approximants are: {}".format(
                approximant, ", ".join(ALLOWED)
            )
        )

    return_float = False
    if isinstance(mass_1, (int, float)):
        return_float = True
        mass_1 = [mass_1]
        mass_2 = [mass_2]
        spin_1x = [spin_1x]
        spin_1y = [spin_1y]
        spin_1z = [spin_1z]
        spin_2x = [spin_2x]
        spin_2y = [spin_2y]
        spin_2z = [spin_2z]

    final_mass = []
    final_spin = []
    for i in range(len(mass_1)):
        m1 = mass_1[i] * MSUN_SI
        m2 = mass_2[i] * MSUN_SI
        spin1 = [spin_1x[i], spin_1y[i], spin_1z[i]]
        spin2 = [spin_2x[i], spin_2y[i], spin_2z[i]]
        if SimInspiralGetSpinSupportFromApproximant(approx) == 3:
            _, fm, fs = SimIMREOBFinalMassSpinPrec(
                m1, m2, spin1, spin2, getattr(lalsimulation, approximant)
            )
        elif SimInspiralGetSpinSupportFromApproximant(approx) == 2:
            _, fm, fs = SimIMREOBFinalMassSpin(
                m1, m2, spin1, spin2, getattr(lalsimulation, approximant)
            )
        final_mass.append((fm * (m1 + m2)) / MSUN_SI)
        final_spin.append(fs)
    if return_float:
        return final_mass[0], final_spin[0]
    return final_mass, final_spin


def final_mass_of_merger_from_NR(*args, NRfit="average", final_spin=None):
    """Return the final mass resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
    final_spin: float/np.ndarray, optional
        precomputed final spin of the remnant.
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
    return func(*args)


def final_mass_of_merger_from_waveform(*args, approximant="SEOBNRv4"):
    """Return the final mass resulting from a BBH merger using a given
    approximant
    """
    return _final_from_initial(*args, approximant=approximant)[0]


def final_spin_of_merger_from_NR(*args, NRfit="average"):
    """Return the final spin resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
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
    return func(*args)


def final_spin_of_merger_from_waveform(*args, approximant="SEOBNRv4"):
    """Return the final spin resulting from a BBH merger using a given
    approximant
    """
    return _final_from_initial(*args, approximant=approximant)[1]


def final_mass_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average", final_spin=None
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
        NR or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    """
    if method.lower() == "nr":
        mass_func = final_mass_of_merger_from_NR
        kwargs = {"NRfit": NRfit, "final_spin": final_spin}
    else:
        mass_func = final_mass_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return mass_func(*args, **kwargs)


def final_spin_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average"
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
        NR or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    """
    if method.lower() == "nr":
        spin_func = final_spin_of_merger_from_NR
        kwargs = {"NRfit": NRfit}
    else:
        spin_func = final_spin_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return spin_func(*args, **kwargs)


def peak_luminosity_of_merger(*args, NRfit="average"):
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
    """
    from pesummary.gw.file import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_peak_luminosity_average")
    else:
        func = getattr(
            nrutils, "bbh_peak_luminosity_non_precessing_{}".format(NRfit)
        )
    return func(*args)


class _Conversion(object):
    """Class to calculate all possible derived quantities

    Parameters
    ----------
    data: dict, list
        either a dictionary or samples or a list of parameters and a list of
        samples. See the examples below for details

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
        f_ref = kwargs.get("f_ref", None)
        if f_ref is not None:
            extra_kwargs["sampler"]["f_ref"] = f_ref
        obj.__init__(parameters, samples, extra_kwargs)
        if kwargs.get("return_dict", True):
            return SamplesDict(obj.parameters, np.array(obj.samples).T)
        else:
            return obj.parameters, obj.samples

    def __init__(self, parameters, samples, extra_kwargs):
        self.parameters = parameters
        self.samples = samples
        self.extra_kwargs = extra_kwargs
        self.generate_all_posterior_samples()

    def _specific_parameter_samples(self, param):
        """Return the samples for a specific parameter

        Parameters
        ----------
        param: str
            the parameter that you would like to return the samples for
        """
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

    def append_data(self, samples):
        """Add a list of samples to the existing samples data object

        Parameters
        ----------
        samples: list
            the list of samples that you would like to append
        """
        for num, i in enumerate(self.samples):
            self.samples[num].append(samples[num])

    def _mchirp_from_mchirp_source_z(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["chirp_mass_source", "redshift"])
        chirp_mass = mchirp_from_mchirp_source_z(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _q_from_eta(self):
        self.parameters.append("mass_ratio")
        samples = self.specific_parameter_samples("symmetric_mass_ratio")
        mass_ratio = q_from_eta(samples)
        self.append_data(mass_ratio)

    def _q_from_m1_m2(self):
        self.parameters.append("mass_ratio")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        mass_ratio = q_from_m1_m2(samples[0], samples[1])
        self.append_data(mass_ratio)

    def _invert_q(self):
        ind = self.parameters.index("mass_ratio")
        for num, i in enumerate(self.samples):
            self.samples[num][ind] = 1. / self.samples[num][ind]

    def _invq_from_q(self):
        self.parameters.append("inverted_mass_ratio")
        samples = self.specific_parameter_samples("mass_ratio")
        inverted_mass_ratio = invq_from_q(samples)
        self.append_data(inverted_mass_ratio)

    def _mchirp_from_mtotal_q(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        chirp_mass = mchirp_from_mtotal_q(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _m1_from_mchirp_q(self):
        self.parameters.append("mass_1")
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_1 = m1_from_mchirp_q(samples[0], samples[1])
        self.append_data(mass_1)

    def _m2_from_mchirp_q(self):
        self.parameters.append("mass_2")
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_2 = m2_from_mchirp_q(samples[0], samples[1])
        self.append_data(mass_2)

    def _reference_frequency(self):
        self.parameters.append("reference_frequency")
        nsamples = len(self.samples)
        extra_kwargs = self.extra_kwargs["sampler"]
        if extra_kwargs != {} and "f_ref" in list(extra_kwargs.keys()):
            self.append_data([float(extra_kwargs["f_ref"])] * nsamples)
        else:
            logger.warn(
                "Could not find reference_frequency in input file. Using 20Hz "
                "as default")
            self.append_data([20.] * nsamples)

    def _mtotal_from_m1_m2(self):
        self.parameters.append("total_mass")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        m_total = m_total_from_m1_m2(samples[0], samples[1])
        self.append_data(m_total)

    def _mchirp_from_m1_m2(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        chirp_mass = mchirp_from_m1_m2(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _eta_from_m1_m2(self):
        self.parameters.append("symmetric_mass_ratio")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        eta = eta_from_m1_m2(samples[0], samples[1])
        self.append_data(eta)

    def _phi_12_from_phi1_phi2(self):
        self.parameters.append("phi_12")
        samples = self.specific_parameter_samples(["phi_1", "phi_2"])
        phi_12 = phi_12_from_phi1_phi2(samples[0], samples[1])
        self.append_data(phi_12)

    def _phi1_from_spins(self):
        self.parameters.append("phi_1")
        samples = self.specific_parameter_samples(["spin_1x", "spin_1y"])
        phi_1 = phi1_from_spins(samples[0], samples[1])
        self.append_data(phi_1)

    def _phi2_from_spins(self):
        self.parameters.append("phi_2")
        samples = self.specific_parameter_samples(["spin_2x", "spin_2y"])
        phi_2 = phi2_from_spins(samples[0], samples[1])
        self.append_data(phi_2)

    def _spin_angles(self):
        angles = ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12",
                  "a_1", "a_2"]
        spin_angles_to_calculate = [
            i for i in angles if i not in self.parameters]
        for i in spin_angles_to_calculate:
            self.parameters.append(i)
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
            self.append_data(data)

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
                    self.parameters.append(i)
                    ind = spins.index(i)
                    data = spin_components[ind]
                    self.append_data(data)

    def _component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        for i in spins_to_calculate:
            self.parameters.append(i)
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
            self.append_data(data)

    def _component_spins_from_azimuthal_and_polar_angles(self):
        spins = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        for i in spins_to_calculate:
            self.parameters.append(i)
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
            self.append_data(data)

    def _chi_p(self):
        self.parameters.append("chi_p")
        parameters = [
            "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]
        samples = self.specific_parameter_samples(parameters)
        chi_p_samples = chi_p(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        self.append_data(chi_p_samples)

    def _chi_eff(self):
        self.parameters.append("chi_eff")
        parameters = ["mass_1", "mass_2", "spin_1z", "spin_2z"]
        samples = self.specific_parameter_samples(parameters)
        chi_eff_samples = chi_eff(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data(chi_eff_samples)

    def _cos_tilt_1_from_tilt_1(self):
        self.parameters.append("cos_tilt_1")
        samples = self.specific_parameter_samples("tilt_1")
        cos_tilt_1 = np.cos(samples)
        self.append_data(cos_tilt_1)

    def _cos_tilt_2_from_tilt_2(self):
        self.parameters.append("cos_tilt_2")
        samples = self.specific_parameter_samples("tilt_2")
        cos_tilt_2 = np.cos(samples)
        self.append_data(cos_tilt_2)

    def _dL_from_z(self):
        self.parameters.append("luminosity_distance")
        samples = self.specific_parameter_samples("redshift")
        distance = dL_from_z(samples)
        self.append_data(distance)

    def _z_from_dL(self):
        self.parameters.append("redshift")
        samples = self.specific_parameter_samples("luminosity_distance")
        redshift = z_from_dL_approx(samples)
        self.append_data(redshift)

    def _comoving_distance_from_z(self):
        self.parameters.append("comoving_distance")
        samples = self.specific_parameter_samples("redshift")
        distance = comoving_distance_from_z(samples)
        self.append_data(distance)

    def _m1_source_from_m1_z(self):
        self.parameters.append("mass_1_source")
        samples = self.specific_parameter_samples(["mass_1", "redshift"])
        mass_1_source = m1_source_from_m1_z(samples[0], samples[1])
        self.append_data(mass_1_source)

    def _m2_source_from_m2_z(self):
        self.parameters.append("mass_2_source")
        samples = self.specific_parameter_samples(["mass_2", "redshift"])
        mass_2_source = m2_source_from_m2_z(samples[0], samples[1])
        self.append_data(mass_2_source)

    def _mtotal_source_from_mtotal_z(self):
        self.parameters.append("total_mass_source")
        samples = self.specific_parameter_samples(["total_mass", "redshift"])
        total_mass_source = m_total_source_from_mtotal_z(samples[0], samples[1])
        self.append_data(total_mass_source)

    def _mchirp_source_from_mchirp_z(self):
        self.parameters.append("chirp_mass_source")
        samples = self.specific_parameter_samples(["chirp_mass", "redshift"])
        chirp_mass_source = mchirp_source_from_mchirp_z(samples[0], samples[1])
        self.append_data(chirp_mass_source)

    def _time_in_each_ifo(self):
        detectors = []
        for i in self.parameters:
            if "optimal_snr" in i and i != "network_optimal_snr":
                det = i.split("_optimal_snr")[0]
                detectors.append(det)

        samples = self.specific_parameter_samples(["ra", "dec", "geocent_time"])
        for i in detectors:
            time = time_in_each_ifo(i, samples[0], samples[1], samples[2])
            self.append_data(time)
            self.parameters.append("%s_time" % (i))

    def _lambda1_from_lambda_tilde(self):
        self.parameters.append("lambda_1")
        samples = self.specific_parameter_samples([
            "lambda_tilde", "mass_1", "mass_2"])
        lambda_1 = lambda1_from_lambda_tilde(samples[0], samples[1], samples[2])
        self.append_data(lambda_1)

    def _lambda2_from_lambda1(self):
        self.parameters.append("lambda_2")
        samples = self.specific_parameter_samples([
            "lambda_1", "mass_1", "mass_2"])
        lambda_2 = lambda2_from_lambda1(samples[0], samples[1], samples[2])
        self.append_data(lambda_2)

    def _lambda_tilde_from_lambda1_lambda2(self):
        self.parameters.append("lambda_tilde")
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        lambda_tilde = lambda_tilde_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data(lambda_tilde)

    def _delta_lambda_from_lambda1_lambda2(self):
        self.parameters.append("delta_lambda")
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        delta_lambda = delta_lambda_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data(delta_lambda)

    def _ifo_snr(self):
        abs_snrs = [
            i for i in self.parameters if "_matched_filter_abs_snr" in i
        ]
        angle_snrs = [
            i for i in self.parameters if "_matched_filter_snr_angle" in i
        ]
        for ifo in [snr.split("_matched_filter_abs_snr")[0] for snr in abs_snrs]:
            self.parameters.append("{}_matched_filter_snr".format(ifo))
            samples = self.specific_parameter_samples(
                [
                    "{}_matched_filter_abs_snr".format(ifo),
                    "{}_matched_filter_snr_angle".format(ifo)
                ]
            )
            snr = _ifo_snr(samples[0], samples[1])
            self.append_data(snr)

    def _optimal_network_snr(self):
        snrs = [i for i in self.parameters if "_optimal_snr" in i]
        samples = self.specific_parameter_samples(snrs)
        self.parameters.append("network_optimal_snr")
        snr = network_snr(samples)
        self.append_data(snr)

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
        self.parameters.append("network_matched_filter_snr")
        snr = network_snr(samples)
        self.append_data(snr)

    @staticmethod
    def _evolved_vs_non_evolved_parameter(parameter, evolved=False):
        if evolved:
            base_string = ""
        else:
            base_string = "_non_evolved"
        return "{}{}".format(parameter, base_string)

    def _peak_luminosity_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "peak_luminosity", evolved=evolved
        )
        self.parameters.append(param)
        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", "spin_1z", "spin_2z"
        ])
        peak_luminosity = peak_luminosity_of_merger(
            samples[0], samples[1], samples[2], samples[3]
        )
        self.append_data(peak_luminosity)

    def _final_mass_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved
        )
        self.parameters.append(param)
        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", "spin_1z", "spin_2z"
        ])
        final_mass = final_mass_of_merger(
            samples[0], samples[1], samples[2], samples[3]
        )
        self.append_data(final_mass)

    def _final_mass_source(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved
        )
        self.parameters.append(param.replace("mass", "mass_source"))
        samples = self.specific_parameter_samples([param, "redshift"])
        final_mass_source = _source_from_detector(
            samples[0], samples[1]
        )
        self.append_data(final_mass_source)

    def _final_spin_of_merger(self, non_precessing=False, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_spin", evolved=evolved
        )
        self.parameters.append(param)
        if not non_precessing:
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2"
            ])
            if "phi_12" in self.parameters:
                phi_12_samples = self.specific_parameter_samples(["phi_12"])[0]
            else:
                phi_12_samples = np.zeros_like(samples[0])
            final_spin = final_spin_of_merger(
                samples[0], samples[1], samples[2], samples[3], samples[4],
                samples[5], phi_12_samples
            )
        else:
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", "spin_1z", "spin_2z"
            ])
            final_spin = final_spin_of_merger(
                samples[0], samples[1], np.abs(samples[2]), np.abs(samples[3]),
                0.5 * np.pi * (1 - np.sign(samples[2])),
                0.5 * np.pi * (1 - np.sign(samples[3])),
                np.zeros_like(samples[0])
            )
        self.append_data(final_spin)

    def _radiated_energy(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "radiated_energy", evolved=evolved
        )
        self.parameters.append(param)
        final_mass_param = self._evolved_vs_non_evolved_parameter(
            "final_mass_source", evolved=evolved
        )
        samples = self.specific_parameter_samples([
            "total_mass_source", final_mass_param
        ])
        radiated_energy = samples[0] - samples[1]
        self.append_data(radiated_energy)

    def _cos_angle(self, parameter_to_add, reverse=False):
        self.parameters.append(parameter_to_add)
        if reverse:
            samples = self.specific_parameter_samples(
                ["cos_" + parameter_to_add])
            cos_samples = np.arccos(samples[0])
        else:
            samples = self.specific_parameter_samples(
                [parameter_to_add.split("cos_")[1]]
            )
            cos_samples = np.cos(samples[0])
        self.append_data(cos_samples)

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

    def generate_all_posterior_samples(self):
        logger.debug("Starting to generate all derived posteriors")
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
        if all(i in self.parameters for i in spin_magnitudes):
            if all(i not in self.parameters for i in angles):
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
        if not all(i in self.parameters for i in spin_magnitudes):
            cartesian = [
                "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                "spin_2z"
            ]
            if not all(i in self.parameters for i in cartesian):
                self.parameters.append("a_1")
                self.parameters.append("a_2")
                for num, i in enumerate(self.samples):
                    self.samples[num].append(0)
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
            final_spin_params = ["a_1", "a_2", "tilt_1", "tilt_2"]
            if all(i in self.parameters for i in final_spin_params):
                if "final_spin_non_evolved" not in self.parameters:
                    self._final_spin_of_merger(evolved=False)
            elif all(i in self.parameters for i in ["spin_1z", "spin_2z"]):
                if "final_spin_non_evolved" not in self.parameters:
                    self._final_spin_of_merger(
                        non_precessing=True, evolved=False
                    )
            if all(i in self.parameters for i in ["spin_1z", "spin_2z"]):
                if "peak_luminosity_non_evolved" not in self.parameters:
                    self._peak_luminosity_of_merger(evolved=False)
                if "final_mass_non_evolved" not in self.parameters:
                    self._final_mass_of_merger(evolved=False)
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
        if "redshift" in self.parameters:
            if "mass_1_source" not in self.parameters and "mass_1" in self.parameters:
                self._m1_source_from_m1_z()
            if "mass_2_source" not in self.parameters and "mass_2" in self.parameters:
                self._m2_source_from_m2_z()
            if "total_mass_source" not in self.parameters and "total_mass" in self.parameters:
                self._mtotal_source_from_mtotal_z()
            if "chirp_mass_source" not in self.parameters and "chirp_mass" in self.parameters:
                self._mchirp_source_from_mchirp_z()
            if "final_mass_source_non_evolved" not in self.parameters:
                if "final_mass_non_evolved" in self.parameters:
                    self._final_mass_source(evolved=False)
        if "total_mass_source" in self.parameters:
            if "final_mass_source_non_evolved" in self.parameters:
                if "radiated_energy_non_evolved" not in self.parameters:
                    self._radiated_energy(evolved=False)
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
        if "reference_frequency" in self.parameters:
            ind = self.parameters.index("reference_frequency")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        if "minimum_frequency" in self.parameters:
            ind = self.parameters.index("minimum_frequency")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
