# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


@array_input()
def mass2_from_m1_q(mass1, q):
    """Return the secondary mass given samples for mass1 and mass ratio
    """
    return mass1 * q


@array_input()
def m_total_from_m1_m2(mass1, mass2):
    """Return the total mass given the samples for mass1 and mass2
    """
    return mass1 + mass2


@array_input()
def component_mass_product(mass1, mass2):
    """Return the product of mass1 and mass2
    """
    return mass1 * mass2


@array_input()
def mchirp_from_m1_m2(mass1, mass2):
    """Return the chirp mass given the samples for mass1 and mass2

    Parameters
    ----------
    """
    total_mass = m_total_from_m1_m2(mass1, mass2)
    return component_mass_product(mass1, mass2)**0.6 / total_mass**0.2


@array_input()
def component_masses_from_mtotal_q(mtotal, q):
    """Return the primary and secondary masses given samples for the total mass
    and mass ratio
    """
    m1 = mtotal / (1. + q)
    return m1, mass2_from_m1_q(m1, q)


@array_input()
def component_masses_from_mchirp_q(mchirp, q):
    """Return the primary and secondary masses given samples for the chirp mass
    and mass ratio
    """
    m1 = ((1. / q)**(2. / 5.)) * ((1.0 + (1. / q))**(1. / 5.)) * mchirp
    return m1, mass2_from_m1_q(m1, q)


@array_input()
def m1_from_mchirp_q(mchirp, q):
    """Return the mass of the larger component given the chirp mass and
    mass ratio
    """
    return component_masses_from_mchirp_q(mchirp, q)[0]


@array_input()
def m2_from_mchirp_q(mchirp, q):
    """Return the mass of the smaller component given the chirp mass and
    mass ratio
    """
    return component_masses_from_mchirp_q(mchirp, q)[1]


@array_input()
def m1_from_mtotal_q(mtotal, q):
    """Return the mass of the larger component given the total mass and
    mass ratio
    """
    return component_masses_from_mtotal_q(mtotal, q)[0]


@array_input()
def m2_from_mtotal_q(mtotal, q):
    """Return the mass of the smaller component given the total mass and
    mass ratio
    """
    return component_masses_from_mtotal_q(mtotal, q)[1]


@array_input()
def eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    total_mass = m_total_from_m1_m2(mass1, mass2)
    return component_mass_product(mass1, mass2) / total_mass**2


@array_input()
def eta_from_mtotal_q(total_mass, mass_ratio):
    """Return the symmetric mass ratio given samples for the total mass and
    mass ratio
    """
    mass1, mass2 = component_masses_from_mtotal_q(total_mass, mass_ratio)
    return eta_from_m1_m2(mass1, mass2)


@array_input()
def q_from_m1_m2(mass1, mass2):
    """Return the mass ratio given the samples for mass1 and mass2
    """
    return mass2 / mass1


@array_input()
def invq_from_m1_m2(mass1, mass2):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass1 and mass2
    """
    return 1. / q_from_m1_m2(mass1, mass2)


@array_input()
def invq_from_q(mass_ratio):
    """Return the inverted mass ratio (mass1/mass2 for mass1 > mass2)
    given the samples for mass ratio (mass2/mass1)
    """
    return 1. / mass_ratio


@array_input()
def q_from_eta(symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return (temp - (temp ** 2 - 1) ** 0.5)


@array_input()
def mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    return eta_from_mtotal_q(total_mass, mass_ratio)**(3. / 5) * total_mass
