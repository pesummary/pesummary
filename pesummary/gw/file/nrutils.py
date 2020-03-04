# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
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
from lalinference.imrtgr import nrutils

LPEAK_FITS = ["UIB2016", "Healyetal"]
FINALMASS_FITS = ["UIB2016", "Healyetal"]
FINALSPIN_FITS = ["UIB2016", "Healyetal", "HBR2016"]


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
        version of the fitting coefficients you wish to use
    """
    return nrutils.bbh_final_spin_non_precessing_Healyetal(*args, **kwargs)


def bbh_final_mass_non_precessing_Healyetal(*args, **kwargs):
    """Return the final mass of the BH resulting from the merger of a BBH for an
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
        version of the fitting coefficients you wish to use
    final_spin: float/np.ndarray, optional
        float/array of precomputed final spins
    """
    chif = kwargs.pop("final_spin", None)
    kwargs.update({"chif": chif})
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


def _bbh_final_spin_precessing_using_non_precessing_fit(*args):
    """
    """
    return nrutils.bbh_final_spin_precessing(
        args[0], args[1], np.abs(args[2]), np.abs(args[3]),
        0.5 * np.pi * (1 - np.sign(args[2])),
        0.5 * np.pi * (1 - np.sign(args[3])),
        np.zeros_like(args[0]), args[4]
    )


def bbh_final_spin_precessing_Panetal(*args):
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
    precessing using the fit from Healy and Lousto: arXiv:1610.09713

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


def _bbh_final_spin_precessing_projected(*args, function=None):
    """Project the precessing spins and calculate the final spin of the BH

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
    spin_1z = args[2] * np.cos(args[4])
    spin_2z = args[3] * np.cos(args[5])
    return function(args[0], args[1], spin_1z, spin_2z)


def bbh_final_spin_precessing_projected_Healyetal(*args):
    """Return the final spin of the BH calculated from projected spins using
    the fit from Healy and Lousto: arXiv:1610.09713

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
    et al. arxiv:1612.09566v1

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
    Panetal = bbh_final_spin_precessing_Panetal
    Healyetal = bbh_final_spin_precessing_projected_Healyetal
    UIB2016 = bbh_final_spin_precessing_projected_UIB2016
    HBR2016 = bbh_final_spin_precessing_HBR2016


def _bbh_average_quantity(*args, fits=None, cls=None, quantity=None):
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
    """
    data, used_fits = [], []
    for fit in fits:
        if hasattr(cls, fit):
            function = getattr(cls, fit)
            used_fits.append(str(function))
            data.append(function(*args))
    logger.info(
        "Averaging the {} from the following fits: {}".format(
            quantity, ", ".join(used_fits)
        )
    )
    return np.mean(data, axis=0)


def bbh_final_mass_average(*args, fits=FINALMASS_FITS):
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
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalMassFits, quantity="final mass"
    )


def bbh_final_spin_average_non_precessing(*args, fits=FINALSPIN_FITS):
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
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalSpinFits, quantity="final spin"
    )


def bbh_final_spin_average_precessing(*args, fits=FINALSPIN_FITS):
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
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=FinalSpinPrecessingFits, quantity="final spin"
    )


def bbh_peak_luminosity_average(*args, fits=LPEAK_FITS):
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
    """
    return _bbh_average_quantity(
        *args, fits=fits, cls=PeakLuminosityFits, quantity="peak luminosity"
    )
