# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input
from pesummary.utils.utils import logger, iterator

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from lalsimulation import (
        CreateSimNeutronStarFamily, SimNeutronStarRadius,
        SimNeutronStarLoveNumberK2, SimNeutronStarEOS4ParameterPiecewisePolytrope,
        SimNSBH_compactness_from_lambda, SimIMRPhenomNSBHProperties,
        SimNeutronStarEOS4ParameterSpectralDecomposition,
        SimIMRPhenomNSBH_baryonic_mass_from_C
    )
    from lal import MRSUN_SI, MSUN_SI
except ImportError:
    pass


@array_input()
def lambda1_plus_lambda2(lambda1, lambda2):
    """Return the sum of the primary objects tidal deformability and the
    secondary objects tidal deformability
    """
    return lambda1 + lambda2


@array_input()
def lambda1_minus_lambda2(lambda1, lambda2):
    """Return the primary objects tidal deformability minus the secondary
    objests tidal deformability
    """
    return lambda1 - lambda2


@array_input()
def lambda_tilde_from_lambda1_lambda2(lambda1, lambda2, mass1, mass2):
    """Return the dominant tidal term given samples for lambda1 and lambda2
    """
    from pesummary.gw.conversions import eta_from_m1_m2
    eta = eta_from_m1_m2(mass1, mass2)
    plus = lambda1_plus_lambda2(lambda1, lambda2)
    minus = lambda1_minus_lambda2(lambda1, lambda2)
    lambda_tilde = 8 / 13 * (
        (1 + 7 * eta - 31 * eta**2) * plus
        + (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * minus)
    return lambda_tilde


@array_input()
def delta_lambda_from_lambda1_lambda2(lambda1, lambda2, mass1, mass2):
    """Return the second dominant tidal term given samples for lambda1 and
    lambda 2
    """
    from pesummary.gw.conversions import eta_from_m1_m2
    eta = eta_from_m1_m2(mass1, mass2)
    plus = lambda1_plus_lambda2(lambda1, lambda2)
    minus = lambda1_minus_lambda2(lambda1, lambda2)
    delta_lambda = 1 / 2 * (
        (1 - 4 * eta) ** 0.5 * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
        * plus + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2
                  + 3380 / 1319 * eta**3) * minus)
    return delta_lambda


@array_input()
def lambda1_from_lambda_tilde(lambda_tilde, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda_tilde
    """
    from pesummary.gw.conversions import eta_from_m1_m2, q_from_m1_m2
    eta = eta_from_m1_m2(mass1, mass2)
    q = q_from_m1_m2(mass1, mass2)
    lambda1 = 13 / 8 * lambda_tilde / (
        (1 + 7 * eta - 31 * eta**2) * (1 + q**-5)
        + (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * (1 - q**-5))
    return lambda1


@array_input()
def lambda2_from_lambda1(lambda1, mass1, mass2):
    """Return the individual tidal parameter given samples for lambda1
    """
    from pesummary.gw.conversions import q_from_m1_m2
    q = q_from_m1_m2(mass1, mass2)
    lambda2 = lambda1 / q**5
    return lambda2


def _lambda1_lambda2_from_eos(eos, mass_1, mass_2):
    """Return lambda_1 and lambda_2 assuming a given equation of state
    """
    fam = CreateSimNeutronStarFamily(eos)
    _lambda = []
    for mass in [mass_1, mass_2]:
        r = SimNeutronStarRadius(mass * MSUN_SI, fam)
        k = SimNeutronStarLoveNumberK2(mass * MSUN_SI, fam)
        c = mass * MRSUN_SI / r
        _lambda.append((2. / 3.) * k / c**5.0)
    return _lambda


def wrapper_for_lambda1_lambda2_polytrope_EOS(args):
    """Wrapper function to calculate the tidal deformability parameters from the
    4_parameter_piecewise_polytrope_equation_of_state parameters for a pool
    of workers
    """
    return _lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(*args)


def _lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
    log_pressure_si, gamma_1, gamma_2, gamma_3, mass_1, mass_2
):
    """Wrapper function to calculate the tidal deformability parameters from the
    4_parameter_piecewise_polytrope_equation_of_state parameters for a pool
    of workers
    """
    eos = SimNeutronStarEOS4ParameterPiecewisePolytrope(
        log_pressure_si, gamma_1, gamma_2, gamma_3
    )
    return _lambda1_lambda2_from_eos(eos, mass_1, mass_2)


def wrapper_for_lambda1_lambda2_from_spectral_decomposition(args):
    """Wrapper function to calculate the tidal deformability parameters from
    the spectral decomposition parameters for a pool of workers
    """
    return _lambda1_lambda2_from_spectral_decomposition(*args)


def _lambda1_lambda2_from_spectral_decomposition(
    spectral_decomposition_gamma_0, spectral_decomposition_gamma_1,
    spectral_decomposition_gamma_2, spectral_decomposition_gamma_3,
    mass_1, mass_2
):
    """Wrapper function to calculate the tidal deformability parameters from
    the spectral decomposition parameters for a pool of workers
    """
    gammas = [
        spectral_decomposition_gamma_0, spectral_decomposition_gamma_1,
        spectral_decomposition_gamma_2, spectral_decomposition_gamma_3
    ]
    eos = SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
    return _lambda1_lambda2_from_eos(eos, mass_1, mass_2)


def _lambda1_lambda2_from_eos_multiprocess(function, args, multi_process=1):
    """
    """
    import multiprocessing

    with multiprocessing.Pool(multi_process[0]) as pool:
        lambdas = np.array(
            list(
                iterator(
                    pool.imap(function, args), tqdm=True, logger=logger,
                    total=len(args), desc="Calculating tidal parameters"
                )
            )
        )
    lambdas = np.array(lambdas).T
    return lambdas[0], lambdas[1]


@array_input()
def lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
    log_pressure, gamma_1, gamma_2, gamma_3, mass_1, mass_2, multi_process=1
):
    """Convert 4 parameter piecewise polytrope EOS parameters to the tidal
    deformability parameters lambda_1, lambda_2
    """
    logger.warn(
        "Calculating the tidal deformability parameters based on the 4 "
        "parameter piecewise polytrope equation of state parameters. This may "
        "take some time"
    )
    log_pressure_si = log_pressure - 1.
    args = np.array(
        [log_pressure_si, gamma_1, gamma_2, gamma_3, mass_1, mass_2]
    ).T
    return _lambda1_lambda2_from_eos_multiprocess(
        wrapper_for_lambda1_lambda2_polytrope_EOS, args,
        multi_process=multi_process[0]
    )


@array_input()
def lambda1_lambda2_from_spectral_decomposition(
    spectral_decomposition_gamma_0, spectral_decomposition_gamma_1,
    spectral_decomposition_gamma_2, spectral_decomposition_gamma_3,
    mass_1, mass_2, multi_process=1
):
    """Convert spectral decomposition parameters to the tidal deformability
    parameters lambda_1, lambda_2
    """
    logger.warn(
        "Calculating the tidal deformability parameters from the spectral "
        "decomposition equation of state parameters. This may take some time"
    )
    args = np.array(
        [
            spectral_decomposition_gamma_0, spectral_decomposition_gamma_1,
            spectral_decomposition_gamma_2, spectral_decomposition_gamma_3,
            mass_1, mass_2
        ]
    ).T
    return _lambda1_lambda2_from_eos_multiprocess(
        wrapper_for_lambda1_lambda2_from_spectral_decomposition, args,
        multi_process=multi_process[0]
    )


@array_input()
def lambda1_from_4_parameter_piecewise_polytrope_equation_of_state(
    log_pressure, gamma_1, gamma_2, gamma_3, mass_1, mass_2
):
    """Convert 4 parameter piecewise polytrope EOS parameters to the tidal
    deformability parameters lambda_1
    """
    return lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
        log_pressure, gamma_1, gamma_2, gamma_3, mass_1, mass_2
    )[0]


@array_input()
def lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
    log_pressure, gamma_1, gamma_2, gamma_3, mass_1, mass_2
):
    """Convert 4 parameter piecewise polytrope EOS parameters to the tidal
    deformability parameters lambda_2
    """
    return lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
        log_pressure, gamma_1, gamma_2, gamma_3, mass_1, mass_2
    )[1]


@array_input()
def NS_compactness_from_lambda(lambda_x):
    """Calculate neutron star compactness from its tidal deformability
    """
    data = np.zeros(len(lambda_x))
    for num, _lambda in enumerate(lambda_x):
        data[num] = SimNSBH_compactness_from_lambda(float(_lambda))
    return data


@array_input()
def NS_baryonic_mass(compactness, NS_mass):
    """Calculate the neutron star baryonic mass from its compactness and
    gravitational mass in solar masses
    """
    data = np.zeros(len(NS_mass))
    for num in np.arange(len(NS_mass)):
        data[num] = SimIMRPhenomNSBH_baryonic_mass_from_C(
            compactness[num], NS_mass[num]
        )
    return data


@array_input()
def _IMRPhenomNSBH_properties(mass_1, mass_2, spin_1z, lambda_2):
    """Calculate NSBH specific properties using the IMRPhenomNSBH waveform
    model given samples for mass_1, mass_2, spin_1z and lambda_2. mass_1 and
    mass_2 should be in solar mass units
    """
    data = np.zeros((len(mass_1), 6))
    for num in range(len(mass_1)):
        data[num] = SimIMRPhenomNSBHProperties(
            float(mass_1[num]) * MSUN_SI, float(mass_2[num]) * MSUN_SI,
            float(spin_1z[num]), float(lambda_2[num])
        )
    transpose_data = data.T
    # convert final mass and torus mass to solar masses
    transpose_data[2] /= MSUN_SI
    transpose_data[4] /= MSUN_SI
    return transpose_data


def _check_NSBH_approximant(approximant, *args, _raise=True):
    """Check that the supplied NSBH waveform model is allowed
    """
    if approximant.lower() == "imrphenomnsbh":
        return _IMRPhenomNSBH_properties(*args)
    msg = (
        "You have supplied the waveform model: '{}'. Currently only the "
        "IMRPhenomNSBH waveform model can be used. Unable to calculate "
        "the NSBH conversion".format(approximant)
    )
    if not _raise:
        logger.warn(msg)
    else:
        raise ValueError(msg)


@array_input()
def NSBH_merger_type(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH",
    percentages=True, percentage_round=2, _ringdown=None, _disruption=None,
    _torus=None
):
    """Determine the merger type based on the disruption frequency, ringdown
    frequency and torus mass. If percentages = True, a dictionary is returned
    showing the number of samples which fall in each category. If
    percentages = False, an array of length mass_1 is returned with
    elements indicating the merger type for each sample
    """
    _type = np.zeros(len(mass_1), dtype='U15')
    _type[:] = "disruptive"
    if not all(param is not None for param in [_ringdown, _disruption, _torus]):
        ringdown, disruption, torus, _, _, _ = _check_NSBH_approximant(
            approximant, mass_1, mass_2, spin_1z, lambda_2
        )
    else:
        ringdown = _ringdown
        disruption = _disruption
        torus = _torus
    freq_ratio = disruption / ringdown
    non_disruptive_inds = np.where(freq_ratio > 1)
    _type[non_disruptive_inds] = "non_disruptive"
    mildly_disruptive_inds = np.where((freq_ratio < 1) & (torus == 0))
    _type[mildly_disruptive_inds] = "mildly_disruptive"
    if percentages:
        _percentages = {
            "non_disruptive": 100 * len(non_disruptive_inds[0]) / len(mass_1),
            "mildly_disruptive": 100 * len(mildly_disruptive_inds[0]) / len(mass_1)
        }
        _percentages["disruptive"] = (
            100 - _percentages["non_disruptive"] - _percentages["mildly_disruptive"]
        )
        for key, value in _percentages.items():
            _percentages[key] = np.round(value, percentage_round)
        return _percentages
    return _type


@array_input()
def NSBH_ringdown_frequency(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the ringdown frequency given samples for mass_1, mass_2,
    spin_1z, lambda_2. mass_1 and mass_2 should be in solar mass units.
    """
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[0]


@array_input()
def NSBH_tidal_disruption_frequency(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the tidal disruption frequency given samples for mass_1,
    mass_2, spin_1z, lambda_2. mass_1 and mass_2 should be in solar mass units.
    """
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[1]


@array_input()
def NSBH_baryonic_torus_mass(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the baryonic torus mass given samples for mass_1, mass_2,
    spin_1z, lambda_2. mass_1 and mass_2 should be in solar mass units.
    """
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[2]


@array_input()
def NS_compactness_from_NSBH(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the neutron star compactness given samples for mass_1, mass_2,
    spin_1z, lambda_2. mass_1 and mass_2 should be in solar mass units.
    """
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[3]
