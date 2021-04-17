# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.utils import logger
from pesummary.utils.decorators import bound_samples

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from lalinference.imrtgr import nrutils
    import lalsimulation
    from lalsimulation import (
        SimInspiralGetSpinFreqFromApproximant, SIM_INSPIRAL_SPINS_CASEBYCASE,
        SIM_INSPIRAL_SPINS_FLOW
    )
except ImportError:
    pass

try:
    from lalsimulation.nrfits.eval_fits import eval_nrfit as _eval_nrfit
    NRSUR_MODULE = True
except (ModuleNotFoundError, ImportError):
    NRSUR_MODULE = False

LPEAK_FITS = ["UIB2016", "Healyetal"]
FINALMASS_FITS = ["UIB2016", "Healyetal"]
FINALSPIN_FITS = ["UIB2016", "Healyetal", "HBR2016"]

NRSUR_FITS = ["final_mass", "final_spin", "final_kick"]
NRSUR_MODEL = "NRSur7dq4Remnant"


def bbh_final_mass_non_spinning_Panetal(*args):
    """Return the final mass of the BH resulting from the merger of a non
    spinning BBH using the fit from Pan et al: Phys Rev D 84, 124052 (2011).

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    """
    return nrutils.bbh_final_mass_non_spinning_Panetal(*args)


@bound_samples(minimum=-1., maximum=1., logger_level="debug")
def bbh_final_spin_non_spinning_Panetal(*args):
    """Return the final spin of the BH resulting from the merger of a non
    spinning BBH using the fit from Pan et al: Phys Rev D 84, 124052 (2011).

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    """
    return nrutils.bbh_final_spin_non_spinning_Panetal(*args)


@bound_samples(minimum=-1., maximum=1., logger_level="debug")
def bbh_final_spin_non_precessing_Healyetal(*args, **kwargs):
    """Return the final spin of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Healy and Lousto: arXiv:1610.09713

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
    version: str, optional
        version of the fitting coefficients you wish to use. Default are the
        fits from 2016
    """
    if "version" not in kwargs.keys():
        kwargs.update({"version": "2016"})
    return nrutils.bbh_final_spin_non_precessing_Healyetal(*args, **kwargs)


def bbh_final_mass_non_precessing_Healyetal(*args, **kwargs):
    """Return the final mass of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Healy et al. If no version specified,
    the default fit is Healy and Lousto: arXiv:1610.09713

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
    version: str, optional
        version of the fitting coefficients you wish to use. Default are the
        fits from 2016
    final_spin: float/np.ndarray, optional
        float/array of precomputed final spins
    """
    chif = kwargs.pop("final_spin", None)
    kwargs.update({"chif": chif})
    if "version" not in kwargs.keys():
        kwargs.update({"version": "2016"})
    return nrutils.bbh_final_mass_non_precessing_Healyetal(*args, **kwargs)


def bbh_final_mass_non_precessing_Husaetal(*args):
    """Return the final mass of the BH resulting from the merge of a BBH for an
    aligned-spin system using the fit from Husa et al: arXiv:1508.07250

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
    return nrutils.bbh_final_mass_non_precessing_Husaetal(*args)


@bound_samples(minimum=-1., maximum=1., logger_level="debug")
def bbh_final_spin_non_precessing_Husaetal(*args):
    """Return the final spin of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Husa et al: arXiv:1508.07250

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
    return nrutils.bbh_final_spin_non_precessing_Husaetal(*args)


def bbh_final_mass_non_precessing_UIB2016(*args, **kwargs):
    """Return the final mass of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from https://arxiv.org/abs/1611.00332

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
    version: str, optional
        version of the fitting coefficients you wish to use
    """
    return nrutils.bbh_final_mass_non_precessing_UIB2016(*args, **kwargs)


@bound_samples(minimum=-1., maximum=1., logger_level="debug")
def bbh_final_spin_non_precessing_UIB2016(*args, **kwargs):
    """Return the final spin of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from https://arxiv.org/abs/1611.00332

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
    version: str, optional
        version of the fitting coefficients you wish to use
    """
    return nrutils.bbh_final_spin_non_precessing_UIB2016(*args, **kwargs)


@bound_samples(minimum=-1., maximum=1., logger_level="debug")
def bbh_final_spin_non_precessing_HBR2016(*args, **kwargs):
    """Return the final spin of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Hofmann, Barausse, and Rezzolla
    ApJL 825, L19 (2016)

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
    version: str, optional
        version of the fitting coefficients you wish to use
    """
    return nrutils.bbh_final_spin_non_precessing_HBR2016(*args, **kwargs)


@bound_samples(maximum=1., logger_level="debug")
def _bbh_final_spin_precessing_using_non_precessing_fit(
    mass_1, mass_2, spin_1z, spin_2z, fit
):
    """Return the final spin of a BH results from the merger of a BH for a
    precessing system using non_precessing fits

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
    fit: str
        name of the NR fit you wish to use
    """
    return nrutils.bbh_final_spin_precessing(
        mass_1, mass_2, np.abs(spin_1z), np.abs(spin_2z),
        0.5 * np.pi * (1 - np.sign(spin_1z)),
        0.5 * np.pi * (1 - np.sign(spin_2z)),
        np.zeros_like(mass_1), fit
    )


def bbh_final_spin_Panetal(*args):
    """Return the final spin of the BH resulting from the merger of a BBH for a
    precessing system using the fit from Pan et al: Phys Rev D 84, 124052 (2011)

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
    return _bbh_final_spin_precessing_using_non_precessing_fit(*args, "Pan2011")


def bbh_final_spin_precessing_Healyetal(*args):
    """Return the final spin of the BH resulting from the merger of a BBH for a
    precessing using the fit from Healy et al. If no version specified,
    the default fit is Healy and Lousto: arXiv:1610.09713

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
    return _bbh_final_spin_precessing_using_non_precessing_fit(*args, "HL2016")


def bbh_final_spin_precessing_UIB2016(*args):
    """Return the final spin of the BH resulting from the merger of a BBH for a
    precessing using the fit from David Keitel, Xisco Jimenez Forteza,
    Sascha Husa, Lionel London et al. arxiv:1612.09566v1

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
    return _bbh_final_spin_precessing_using_non_precessing_fit(*args, "UIB2016")


@bound_samples(maximum=1., logger_level="debug")
def _bbh_final_spin_precessing_projected(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, function=None
):
    """Project the precessing spins along the orbital angular momentum and
    calculate the final spin of the BH with an aligned-spin fit from the
    literature augmenting it with the leading contribution from the in-plane
    spins

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
    """
    spin_1z = a_1 * np.cos(tilt_1)
    spin_2z = a_2 * np.cos(tilt_2)
    final_spin_aligned = function(mass_1, mass_2, spin_1z, spin_2z)
    final_spin_aligned_squared = final_spin_aligned * final_spin_aligned
    total_mass = mass_1 + mass_2

    a_1perp = mass_1 * mass_1 * a_1 * np.sin(tilt_1)
    a_2perp = mass_2 * mass_2 * a_2 * np.sin(tilt_2)
    a_perp_squared = (
        a_1perp * a_1perp + a_2perp * a_2perp
        + 2. * a_1perp * a_2perp * np.cos(phi_12)
    )
    return (final_spin_aligned_squared + a_perp_squared / (total_mass)**4.)**0.5


def bbh_final_spin_precessing_projected_Healyetal(*args):
    """Return the final spin of the BH calculated from projected spins using
    the fit from Healy and Lousto: arXiv:1610.09713 augmenting it with the
    leading contribution from the in-plane spins

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
    """
    return _bbh_final_spin_precessing_projected(
        *args, function=bbh_final_spin_precessing_Healyetal
    )


def bbh_final_spin_precessing_projected_UIB2016(*args):
    """Return the final spin of the BH calculated from projected spins using
    the fit by David Keitel, Xisco Jimenez Forteza, Sascha Husa, Lionel London
    et al. arxiv:1612.09566v1 augmenting it with the leading contribution from
    the in-plane spins

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
    """
    return _bbh_final_spin_precessing_projected(
        *args, function=bbh_final_spin_precessing_UIB2016
    )


@bound_samples(maximum=1., logger_level="debug")
def bbh_final_spin_precessing_HBR2016(*args, **kwargs):
    """Return the final spin of the BH resulting from the merger of a BBH for a
    precessing system using the fit from Hofmann, Barausse, and Rezzolla ApJL
    825, L19 (2016)

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
    version: str, optional
        version of the fitting coefficients you wish to use
    """
    return nrutils.bbh_final_spin_precessing_HBR2016(*args, **kwargs)


def bbh_peak_luminosity_non_precessing_T1600018(*args):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the fit by Sascha Husa, Xisco Jimenez Forteza, David Keitel
    [LIGO-T1500598] using 5th order in chieff

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
    return nrutils.bbh_peak_luminosity_non_precessing_T1600018(*args)


def bbh_peak_luminosity_non_precessing_UIB2016(*args):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the fit by David Keitel, Xisco Jimenez Forteza, Sascha Husa,
    Lionel London et al. arxiv:1612.09566v1

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
    return nrutils.bbh_peak_luminosity_non_precessing_UIB2016(*args)


def bbh_peak_luminosity_non_precessing_Healyetal(*args):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the fit from Healy and Lousto: arXiv:1610.09713

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
    return nrutils.bbh_peak_luminosity_non_precessing_Healyetal(*args)


class PeakLuminosityFits(object):
    UIB2016 = bbh_peak_luminosity_non_precessing_UIB2016
    Healyetal = bbh_peak_luminosity_non_precessing_Healyetal
    T1600018 = bbh_peak_luminosity_non_precessing_T1600018


class FinalMassFits(object):
    Panetal = bbh_final_mass_non_spinning_Panetal
    Healyetal = bbh_final_mass_non_precessing_Healyetal
    Husaetal = bbh_final_mass_non_precessing_Husaetal
    UIB2016 = bbh_final_mass_non_precessing_UIB2016


class FinalSpinFits(object):
    Panetal = bbh_final_spin_non_spinning_Panetal
    Healyetal = bbh_final_spin_non_precessing_Healyetal
    Husaetal = bbh_final_spin_non_precessing_Husaetal
    UIB2016 = bbh_final_spin_non_precessing_UIB2016
    HBR2016 = bbh_final_spin_non_precessing_HBR2016


class FinalSpinPrecessingFits(object):
    Healyetal = bbh_final_spin_precessing_projected_Healyetal
    UIB2016 = bbh_final_spin_precessing_projected_UIB2016
    HBR2016 = bbh_final_spin_precessing_HBR2016


def _bbh_average_quantity(
    *args, fits=None, cls=None, quantity=None, return_fits_used=False
):
    """Average the result from multiple fits

    Parameters
    ----------
    *args: tuple
        tuple of arguments for the fitting functions
    fits: list
        list of fits that you wish to use
    cls: class
        class which maps a string to a given fitting function
    quantity: str
        quantity that you are combining results for
    return_fits_used: Bool, optional
        if True, return the fits that were used to calculate the average
    """
    data, used_fits = [], []
    for fit in fits:
        if hasattr(cls, fit):
            function = getattr(cls, fit)
            used_fits.append(str(function).replace("<", "").replace(">", ""))
            data.append(function(*args))
    logger.info(
        "Averaging the {} from the following fits: {}".format(
            quantity, ", ".join(used_fits)
        )
    )
    if return_fits_used:
        return np.mean(data, axis=0), used_fits
    return np.mean(data, axis=0)


def bbh_final_mass_average(*args, fits=FINALMASS_FITS, return_fits_used=False):
    """Return the final mass averaged across multiple fits

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
    fits: list, optional
        list of fits that you wish to use
    return_fits_used: Bool, optional
        if True, return the fits that were used to calculate the average
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalMassFits, quantity="final mass",
        return_fits_used=return_fits_used
    )


def bbh_final_spin_average_non_precessing(
    *args, fits=FINALSPIN_FITS, return_fits_used=False
):
    """Return the final spin averaged across multiple non-precessing fits

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
    fits: list, optional
        list of fits that you wish to use
    return_fits_used: Bool, optional
        if True, return the fits that were used to calculate the average
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalSpinFits, quantity="final spin",
        return_fits_used=return_fits_used
    )


def bbh_final_spin_average_precessing(
    *args, fits=FINALSPIN_FITS, return_fits_used=False
):
    """Return the final spin averaged across multiple fits

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
    fits: list, optional
        list of fits that you wish to use
    return_fits_used: Bool, optional
        if True, return the fits that were used to calculate the average
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalSpinPrecessingFits, quantity="final spin",
        return_fits_used=return_fits_used
    )


def bbh_peak_luminosity_average(*args, fits=LPEAK_FITS, return_fits_used=False):
    """Return the peak luminosity (in units of 10^56 ergs/s) averaged across
    multiple fits.

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
    fits: list, optional
        list of fits that you wish to use
    return_fits_used: Bool, optional
        if True, return the fits that were used to calculate the average
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=PeakLuminosityFits, quantity="peak luminosity",
        return_fits_used=return_fits_used
    )


def eval_nrfit(*args, **kwargs):
    from contextlib import contextmanager
    import ctypes
    import io
    import os
    import sys
    import tempfile

    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

    @contextmanager
    def stdout_redirector(stream):
        original_stdout_fd = sys.stdout.fileno()

        def _redirect_stdout(to_fd):
            """Redirect stdout to the given file descriptor."""
            libc.fflush(c_stdout)
            sys.stdout.close()
            os.dup2(to_fd, original_stdout_fd)
            sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

        saved_stdout_fd = os.dup(original_stdout_fd)
        try:
            tfile = tempfile.TemporaryFile(mode='w+b')
            _redirect_stdout(tfile.fileno())
            yield
            _redirect_stdout(saved_stdout_fd)
            tfile.flush()
            tfile.seek(0, io.SEEK_SET)
            stream.write(tfile.read())
        finally:
            tfile.close()
            os.close(saved_stdout_fd)

    f = io.BytesIO()
    with stdout_redirector(f):
        NRSurrogate_kwargs = kwargs.copy()
        approximant = kwargs.get("approximant", None)
        f_low = kwargs.get("f_low", None)
        f_ref = kwargs.get("f_ref", None)
        spinfreq_enum = SimInspiralGetSpinFreqFromApproximant(
            getattr(lalsimulation, approximant)
        )
        if spinfreq_enum == SIM_INSPIRAL_SPINS_CASEBYCASE:
            raise ValueError(
                "Unable to evolve spins as '{}' does not have a set frequency "
                "at which the spins are defined".format(approximant)
            )
        f_start = float(np.where(
            np.array(spinfreq_enum == SIM_INSPIRAL_SPINS_FLOW), f_low, f_ref
        ))
        NRSurrogate_kwargs["f_ref"] = f_start
        NRSurrogate_kwargs.pop("f_low")
        NRSurrogate_kwargs.pop("approximant")
        data = _eval_nrfit(*args, **NRSurrogate_kwargs)
    return data


def NRSur_fit(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phi_ref,
    f_low=20., f_ref=20., model=NRSUR_MODEL, fits=NRSUR_FITS, return_fits_used=False,
    approximant=None, **kwargs
):
    """Return the NR fits based on a chosen NRSurrogate model

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object. In units of solar mass
    mass_2: float/np.ndarray
        float/array of masses for the secondary object. In units of solar mass
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
    phi_jl: float/np.ndarray
        float/array of samples for the azimuthal angle of the orbital angular momentum
        around the total orbital angular momentum
    theta_jn: float/np.ndarray
        float/array of samples for the angle between the total angular momentum and
        the line of sight
    phi_ref: float/np.ndarray
        float/array of samples for the reference phase used in the analysis
    f_low: float
        the low frequency cut-off used in the analysis
    f_ref: float/np.ndarray, optional
        the reference frequency used in the analysis
    model: str, optional
        NRSurrogate model that you wish to use for the calculation
    fits: list, optional
        list of fits that you wish to evaluate
    approximant: str, optional
        The approximant that was used to generate the posterior samples
    kwargs: dict, optional
        optional kwargs that are passed directly to the
        `lalsimulation.nrfits.eval_fits.eval_nrfit` function
    """
    from lal import MSUN_SI, C_SI
    from .spins import component_spins
    from .utils import magnitude_from_vector
    from pesummary.utils.utils import iterator
    import copy

    if not NRSUR_MODULE:
        raise ImportError(
            "Unable to import `lalsimulation.nrfits.eval_fits`. This is likely "
            "due to the installed version of lalsimulation. Please update."
        )

    fits_map = {
        "final_mass": "FinalMass", "final_spin": "FinalSpin",
        "final_kick": "RecoilKick"
    }
    inverse_fits_map = {item: key for key, item in fits_map.items()}
    description = "Evaluating NRSurrogate fit for {}".format(", ".join(fits))
    converted_fits = copy.deepcopy(fits)
    for fit, conversion in fits_map.items():
        if fit in converted_fits:
            ind = fits.index(fit)
            converted_fits[ind] = conversion

    spins = np.array(
        component_spins(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
            f_ref, phi_ref
        )
    )
    a_1_vec = np.array([spins.T[1], spins.T[2], spins.T[3]]).T
    a_2_vec = np.array([spins.T[4], spins.T[5], spins.T[6]]).T
    mass_1 *= MSUN_SI
    mass_2 *= MSUN_SI
    try:
        _fits = [
            eval_nrfit(
                mass_1[num], mass_2[num], a_1_vec[num], a_2_vec[num], model,
                converted_fits, f_low=f_low, f_ref=f_ref[num],
                approximant=approximant, extra_params_dict=kwargs
            ) for num in iterator(
                range(len(mass_1)), desc=description, tqdm=True, total=len(mass_1),
                logger=logger
            )
        ]
    except ValueError as e:
        base = (
            "Failed to generate remnant quantities with the NRSurrogate "
            "remnant model. {}"
        )
        if "symbol not found" in str(e):
            raise NameError(
                base.format(
                    "This could be because the 'LAL_DATA_PATH' has not been "
                    "set."
                )
            )
        raise ValueError(base.format(""))
    nr_fits = {key: np.array([dic[key] for dic in _fits]) for key in _fits[0]}
    if fits_map["final_mass"] in nr_fits.keys():
        nr_fits[fits_map["final_mass"]] = np.array(
            [final_mass[0] for final_mass in nr_fits[fits_map["final_mass"]]]
        ) / MSUN_SI
    if fits_map["final_kick"] in nr_fits.keys():
        nr_fits[fits_map["final_kick"]] *= C_SI / 1000
        final_kick_abs = magnitude_from_vector(
            nr_fits[fits_map["final_kick"]]
        )
        nr_fits[fits_map["final_kick"]] = final_kick_abs
    if fits_map["final_spin"] in nr_fits.keys():
        final_spin_abs = magnitude_from_vector(
            nr_fits[fits_map["final_spin"]]
        )
        nr_fits[fits_map["final_spin"]] = final_spin_abs
    return {
        key if key not in inverse_fits_map.keys() else inverse_fits_map[key]:
        item for key, item in nr_fits.items()
    }
