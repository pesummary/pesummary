# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input

__author__ = [
    "Stephen Fairhurst <stephen.fairhurst@ligo.org>",
    "Rhys Green <rhys.green@ligo.org>",
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def _dpsi(theta_jn, phi_jl, beta):
    """Calculate the difference between the polarization with respect to the
    total angular momentum and the polarization with respect to the orbital
    angular momentum
    """
    if theta_jn == 0:
        return -1. * phi_jl
    n = np.array([np.sin(theta_jn), 0, np.cos(theta_jn)])
    j = np.array([0, 0, 1])
    l = np.array([
        np.sin(beta) * np.sin(phi_jl), np.sin(beta) * np.cos(phi_jl), np.cos(beta)
    ])
    p_j = np.cross(n, j)
    p_j /= np.linalg.norm(p_j)
    p_l = np.cross(n, l)
    p_l /= np.linalg.norm(p_l)
    cosine = np.inner(p_j, p_l)
    sine = np.inner(n, np.cross(p_j, p_l))
    dpsi = np.pi / 2 + np.sign(sine) * np.arccos(cosine)
    return dpsi


@array_input()
def _dphi(theta_jn, phi_jl, beta):
    """Calculate the difference in the phase angle between J-aligned
    and L-aligned frames

    Parameters
    ----------
    theta_jn: np.ndarray
        the angle between J and line of sight
    phi_jl: np.ndarray
        the precession phase
    beta: np.ndarray
        the opening angle (angle between J and L)
    """
    n = np.column_stack(
        [np.repeat([0], len(theta_jn)), np.sin(theta_jn), np.cos(theta_jn)]
    )
    l = np.column_stack(
        [
            np.sin(beta) * np.cos(phi_jl), np.sin(beta) * np.sin(phi_jl),
            np.cos(beta)
        ]
    )
    cosi = [np.inner(nn, ll) for nn, ll in zip(n, l)]
    inc = np.arccos(cosi)
    sign = np.sign(np.cos(theta_jn) - (np.cos(beta) * np.cos(inc)))
    cos_d = np.cos(phi_jl) * np.sin(theta_jn) / np.sin(inc)
    dphi = -1. * sign * np.arccos(cos_d)
    return dphi


@array_input()
def psi_J(psi_L, theta_jn, phi_jl, beta):
    """Calculate the polarization with respect to the total angular momentum
    """
    dpsi = []
    for i in range(len(theta_jn)):
        dpsi.append(_dpsi(theta_jn[i], phi_jl[i], beta[i]))
    psi = psi_L + np.array(dpsi)
    return psi
