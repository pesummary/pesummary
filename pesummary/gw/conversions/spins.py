# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from lalsimulation import (
        SimInspiralTransformPrecessingWvf2PE,
        SimInspiralTransformPrecessingNewInitialConditions
    )
    from lal import MSUN_SI
except ImportError:
    pass


@array_input()
def viewing_angle_from_inclination(inclination):
    """Return the viewing angle of the binary given samples for the source
    inclination angle. For a precessing system, the source inclination angle
    is theta_jn: the angle between the total angular momentum J and the line of
    sight N. For a non-precessing system, the source inclination angle is
    iota: the angle between the total orbital angular momentum L and the line
    of sight N
    """
    return np.min([inclination, np.pi - inclination], axis=0)


@array_input()
def _chi_p(mass1, mass2, S1_perp, S2_perp):
    """Return chi_p given samples for mass1, mass2, S1_perp, S2_perp
    """
    mass_ratio = mass2 / mass1
    chi_p = np.maximum(
        S1_perp, (4 * mass_ratio + 3) / (3 * mass_ratio + 4) * mass_ratio
        * S2_perp
    )
    return chi_p


@array_input()
def chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return chi_p given samples for mass1, mass2, spin1x, spin1y, spin2x,
    spin2y
    """
    S1_perp = np.linalg.norm([spin1x, spin1y], axis=0)
    S2_perp = np.linalg.norm([spin2x, spin2y], axis=0)
    return _chi_p(mass1, mass2, S1_perp, S2_perp)


@array_input()
def chi_p_from_tilts(mass1, mass2, a_1, tilt_1, a_2, tilt_2):
    """Return chi_p given samples for mass1, mass2, a_1, tilt_2, a_2, tilt_2
    """
    S1_perp = a_1 * np.sin(tilt_1)
    S2_perp = a_2 * np.sin(tilt_2)
    return _chi_p(mass1, mass2, S1_perp, S2_perp)


@array_input()
def chi_p_2spin(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return the magnitude of a modified chi_p which takes into account
    precessing spin information from both compact objects given samples for
    mass1, mass2, spin1x, spin1y, spin2x, spin2y. See Eq.9 of arXiv:2012.02209
    """
    chi_p_2spin = np.zeros((2, len(mass1)))
    S1_perp = mass1**2 * np.array([spin1x, spin1y])
    S2_perp = mass2**2 * np.array([spin2x, spin2y])
    S_perp = np.sum([S1_perp, S2_perp], axis=0)
    S1_perp_mag = np.linalg.norm(S1_perp, axis=0)
    S2_perp_mag = np.linalg.norm(S2_perp, axis=0)
    mask = S1_perp_mag >= S2_perp_mag
    chi_p_2spin[:, mask] = (S_perp / (mass1**2 + S2_perp_mag))[:, mask]
    chi_p_2spin[:, ~mask] = (S_perp / (mass2**2 + S1_perp_mag))[:, ~mask]
    return np.linalg.norm(chi_p_2spin, axis=0)


@array_input()
def chi_eff(mass1, mass2, spin1z, spin2z):
    """Return chi_eff given samples for mass1, mass2, spin1z, spin2z
    """
    return (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)


@array_input()
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


@array_input()
def phi1_from_spins(spin_1x, spin_1y):
    """Return phi_1 given samples for spin_1x and spin_1y
    """
    phi_1 = np.fmod(2 * np.pi + np.arctan2(spin_1y, spin_1x), 2 * np.pi)
    return phi_1


@array_input()
def phi2_from_spins(spin_2x, spin_2y):
    """Return phi_2 given samples for spin_2x and spin_2y
    """
    return phi1_from_spins(spin_2x, spin_2y)


@array_input()
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


@array_input()
def component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                    mass_2, f_ref, phase):
    """Return the component spins given samples for theta_jn, phi_jl, tilt_1,
    tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase
    """
    data = []
    for i in range(len(theta_jn)):
        iota, S1x, S1y, S1z, S2x, S2y, S2z = \
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn[i], phi_jl[i], tilt_1[i], tilt_2[i], phi_12[i],
                a_1[i], a_2[i], mass_1[i] * MSUN_SI, mass_2[i] * MSUN_SI,
                float(f_ref[i]), float(phase[i]))
        data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
    return data


@array_input()
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


@array_input()
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


@array_input()
def opening_angle(
    mass_1, mass_2, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, f_ref, phase
):
    """Return the opening angle of the system given samples for mass_1, mass_2,
    cartesian spins and a reference frequency
    """
    data = []
    for i in range(len(mass_1)):
        beta, _, _, _, _, _, _ = \
            SimInspiralTransformPrecessingNewInitialConditions(
                0., phi_jl[i], tilt_1[i], tilt_2[i], phi_12[i],
                a_1[i], a_2[i], mass_1[i] * MSUN_SI, mass_2[i] * MSUN_SI,
                float(f_ref[i]), float(phase[i]))
        data.append(beta)
    return data
