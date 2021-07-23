# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import multiprocessing

from pesummary.gw.conversions import (
    tilt_angles_and_phi_12_from_spin_vectors_and_L
)
from pesummary.utils.utils import iterator, logger
from pesummary.utils.exceptions import EvolveSpinError
from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from lal import MTSUN_SI, MSUN_SI
    import lalsimulation
    from lalsimulation import (
        SimInspiralGetSpinFreqFromApproximant, SIM_INSPIRAL_SPINS_CASEBYCASE,
        SIM_INSPIRAL_SPINS_FLOW, SimInspiralSpinTaylorPNEvolveOrbit
    )
except ImportError:
    pass


def evolve_spins(*args, evolve_limit="ISCO", **kwargs):
    """Evolve spins to a given limit.

    Parameters
    ----------
    *args: tuple
        all arguments passed to either evolve_angles_forwards or
        evolve_angles_backwards
    evolve_limit: str/float, optional
        limit to evolve frequencies. If evolve_limit=='infinite_separation' or
        evolve_limit==0, evolve spins to infinite separation. if
        evolve_limit=='ISCO', evolve spins to ISCO frequency. If any other
        float, evolve spins to that frequency.
    **kwargs: dict, optional
        all kwargs passed to either evolve_angles_forwards or evolve_angles_backwards
    """
    _infinite_string = "infinite_separation"
    cond1 = isinstance(evolve_limit, str) and evolve_limit.lower() == _infinite_string
    if cond1 or evolve_limit == 0:
        return evolve_angles_backwards(*args, **kwargs)
    else:
        return evolve_angles_forwards(*args, **kwargs)


def evolve_angles_forwards(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, f_low, f_ref,
    approximant, final_velocity="ISCO", tolerance=1e-3,
    dt=0.1, multi_process=1, evolution_approximant="SpinTaylorT5"
):
    """Evolve the BBH spin angles forwards to a specified value using
    lalsimulation.SimInspiralSpinTaylorPNEvolveOrbit. By default this is
    the Schwarzchild ISCO velocity.

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of primary mass samples of the binary
    mass_2: float/np.ndarray
        float/array of secondary mass samples of the binary
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    f_low: float
        low frequency cutoff used in the analysis
    f_ref: float
        reference frequency where spins are defined
    approximant: str
        Approximant used to generate the posterior samples
    final_velocity: str, float
        final orbital velocity for the evolution. This can either be the
        Schwarzschild ISCO velocity 6**-0.5 ~= 0.408 ('ISCO') or a
        fraction of the speed of light
    tolerance: float
        Only evolve spins if at least one spin's magnitude is greater than
        tolerance
    dt: float
        steps in time for the integration, in terms of the mass of the binary
    multi_process: int, optional
        number of cores to run on when evolving the spins. Default: 1
    evolution_approximant: str
        name of the approximant you wish to use to evolve the spins. Default
        is SpinTaylorT5. Other choices are SpinTaylorT1 or SpinTaylorT4
    """
    if isinstance(final_velocity, str) and final_velocity.lower() == "isco":
        final_velocity = 6. ** -0.5
    else:
        final_velocity = float(final_velocity)

    spinfreq_enum = SimInspiralGetSpinFreqFromApproximant(
        getattr(lalsimulation, approximant)
    )
    if spinfreq_enum == SIM_INSPIRAL_SPINS_CASEBYCASE:
        _msg = (
            "Unable to evolve spins as '{}' does not have a set frequency "
            "at which the spins are defined".format(approximant)
        )
        logger.warning(_msg)
        raise EvolveSpinError(_msg)
    f_start = float(np.where(
        np.array(spinfreq_enum == SIM_INSPIRAL_SPINS_FLOW), f_low, f_ref
    ))
    with multiprocessing.Pool(multi_process) as pool:
        args = np.array([
            mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12,
            [f_start] * len(mass_1), [final_velocity] * len(mass_1),
            [tolerance] * len(mass_1), [dt] * len(mass_1),
            [evolution_approximant] * len(mass_1)
        ], dtype="object").T
        data = np.array(
            list(
                iterator(
                    pool.imap(_wrapper_for_evolve_angles_forwards, args),
                    tqdm=True, logger=logger, total=len(mass_1),
                    desc="Evolving spins forward for remnant fits evaluation"
                )
            )
        )
    tilt_1_evol, tilt_2_evol, phi_12_evol = data.T
    return tilt_1_evol, tilt_2_evol, phi_12_evol


def _wrapper_for_evolve_angles_forwards(args):
    """Wrapper function for _evolve_angles_forwards for a pool of workers

    Parameters
    ----------
    args: tuple
        All args passed to _evolve_angles_forwards
    """
    return _evolve_angles_forwards(*args)


def _evolve_angles_forwards(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, f_start, final_velocity,
    tolerance, dt, evolution_approximant
):
    """Wrapper function for the SimInspiralSpinTaylorPNEvolveOrbit function

    Parameters
    ----------
    mass_1: float
        primary mass of the binary
    mass_2: float
        secondary mass of the binary
    a_1: float
        primary spin magnitude
    a_2: float
        secondary spin magnitude
    tilt_1: float
        primary spin tilt angle from the orbital angular momentum
    tilt_2: float
        secondary spin tilt angle from the orbital angular momentum
    phi_12: float
        the angle between the in-plane spin components
    f_start: float
        frequency to start the evolution from
    final_velocity: float
        Final velocity to evolve the spins up to
    tolerance: float
        Only evolve spins if at least one spins magnitude is greater than
        tolerance
    dt: float
        steps in time for the integration, in terms of the mass of the binary
    evolution_approximant: str
        name of the approximant you wish to use to evolve the spins.
    """
    from packaging import version
    if np.logical_or(a_1 > tolerance, a_2 > tolerance):
        # Total mass in seconds
        total_mass = (mass_1 + mass_2) * MTSUN_SI
        f_final = final_velocity ** 3 / (total_mass * np.pi)
        _approx = getattr(lalsimulation, evolution_approximant)
        if version.parse(lalsimulation.__version__) >= version.parse("2.5.2"):
            spinO = 6
        else:
            spinO = 7
        data = SimInspiralSpinTaylorPNEvolveOrbit(
            deltaT=dt * total_mass, m1=mass_1 * MSUN_SI,
            m2=mass_2 * MSUN_SI, fStart=f_start, fEnd=f_final,
            s1x=a_1 * np.sin(tilt_1), s1y=0.,
            s1z=a_1 * np.cos(tilt_1),
            s2x=a_2 * np.sin(tilt_2) * np.cos(phi_12),
            s2y=a_2 * np.sin(tilt_2) * np.sin(phi_12),
            s2z=a_2 * np.cos(tilt_2), lnhatx=0., lnhaty=0., lnhatz=1.,
            e1x=1., e1y=0., e1z=0., lambda1=0., lambda2=0., quadparam1=1.,
            quadparam2=1., spinO=spinO, tideO=0, phaseO=7, lscorr=0,
            approx=_approx
        )
        # Set index to take from array output by SimInspiralSpinTaylorPNEvolveOrbit:
        # -1 for evolving forward in time and 0 for evolving backward in time
        if f_start <= f_final:
            idx_use = -1
        else:
            idx_use = 0
        a_1_evolve = np.array(
            [
                data[2].data.data[idx_use], data[3].data.data[idx_use],
                data[4].data.data[idx_use]
            ]
        )
        a_2_evolve = np.array(
            [
                data[5].data.data[idx_use], data[6].data.data[idx_use],
                data[7].data.data[idx_use]
            ]
        )
        Ln_evolve = np.array(
            [
                data[8].data.data[idx_use], data[9].data.data[idx_use],
                data[10].data.data[idx_use]
            ]
        )
        tilt_1_evol, tilt_2_evol, phi_12_evol = \
            tilt_angles_and_phi_12_from_spin_vectors_and_L(
                a_1_evolve, a_2_evolve, Ln_evolve
            )
    else:
        tilt_1_evol, tilt_2_evol, phi_12_evol = tilt_1, tilt_2, phi_12
    return tilt_1_evol, tilt_2_evol, phi_12_evol


def _wrapper_for_evolve_angles_backwards(args):
    """Wrapper function for evolving tilts backwards for a pool of workers

    Parameters
    ----------
    args: tuple
        Zeroth arg is the function you wish to use when evolving the tilts.
        1st to 8th args are arguments passed to function. All other arguments
        are treated as kwargs passed to function
    """
    _function = args[0]
    _args = args[1:9]
    _kwargs = args[9:]
    return _function(*_args, **_kwargs[0])


@array_input(
    ignore_kwargs=[
        "method", "multi_process", "return_fits_used", "version"
    ], force_return_array=True
)
def evolve_angles_backwards(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, f_ref,
    method="precession_averaged", multi_process=1, return_fits_used=False,
    version="v1", **kwargs
):
    """Evolve BBH tilt angles backwards to infinite separation

    Parameters
    ----------
    mass_1: float/np.ndarray
       float/array of primary mass samples of the binary
    mass_2: float/np.ndarray
        float/array of secondary mass samples of the binary
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    f_ref: float
        reference frequency where spins are defined
    method: str
        Method to use when evolving tilts to infinity. Possible options are
        'precession_averaged' and 'hybrid_orbit_averaged'. 'precession_averaged'
        computes tilt angles at infinite separation assuming that precession
        averaged spin evolution from Gerosa et al. is valid starting from f_ref.
        'hybrid_orbit_averaged' combines orbit-averaged evolution and
        'precession_averaged' evolution as in Johnson-McDaniel et al. This is more
        accurate but slower than the 'precession_averaged' method.
    multi_process: int, optional
        number of cores to run on when evolving the spins. Default: 1
    return_fits_used: Bool, optional
        return a dictionary of fits used. Default False
    version: str, optional
        version of the
        tilts_at_infinity.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve
        function to use within the lalsimulation library. Default 'v1'
    **kwargs: dict, optional
        all kwargs passed to the
        tilts_at_infinity.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve
        function in the lalsimulation library
    """
    from lalsimulation.tilts_at_infinity import hybrid_spin_evolution
    _mds = ["precession_averaged", "hybrid_orbit_averaged"]
    if method.lower() not in _mds:
        raise ValueError(
            "Invalid method. Please choose either {}".format(", ".join(_mds))
        )
    kwargs.update(
        {"prec_only": method.lower() == "precession_averaged", "version": version}
    )

    with multiprocessing.Pool(multi_process) as pool:
        args = np.array(
            [
                [hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve] * len(mass_1),
                mass_1 * MSUN_SI, mass_2 * MSUN_SI, a_1, a_2, tilt_1, tilt_2, phi_12,
                [f_ref] * len(mass_1), [kwargs] * len(mass_1)
            ], dtype=object
        ).T
        data = np.array(
            list(
                iterator(
                    pool.imap(_wrapper_for_evolve_angles_backwards, args),
                    tqdm=True, desc="Evolving spins backwards to infinite separation",
                    logger=logger, total=len(mass_1)
                )
            )
        )
    tilt_1_inf = np.array([l["tilt1_inf"] for l in data])
    tilt_2_inf = np.array([l["tilt2_inf"] for l in data])
    if return_fits_used:
        fits_used = [
            method.lower(), (
                "lalsimulation.tilts_at_infinity.hybrid_spin_evolution."
                "calc_tilts_at_infty_hybrid_evolve=={}".format(version)
            )
        ]
        return [tilt_1_inf, tilt_2_inf, phi_12], fits_used
    return tilt_1_inf, tilt_2_inf, phi_12
