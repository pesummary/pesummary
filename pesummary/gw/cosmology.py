# Licensed under an MIT style license -- see LICENSE.md

from pesummary import conf
from astropy import cosmology as cosmo
from astropy.cosmology import parameters

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_astropy_cosmologies = [i.lower() for i in parameters.available]
available_cosmologies = _astropy_cosmologies + ["planck15_lal"]
available_cosmologies += [
    _cosmology + "_with_riess2019_h0" for _cosmology in available_cosmologies
]


def get_cosmology(cosmology=conf.cosmology):
    """Return the cosmology that is being used

    Parameters
    ----------
    cosmology: str
        name of a known cosmology
    """
    if cosmology.lower() not in available_cosmologies:
        raise ValueError(
            "Unrecognised cosmology {}. Available cosmologies are {}".format(
                cosmology, ", ".join(available_cosmologies)
            )
        )
    elif cosmology.lower() in _astropy_cosmologies:
        ind = [
            num for num, name in enumerate(_astropy_cosmologies) if
            name == cosmology.lower()
        ][0]
        return getattr(cosmo, list(parameters.available)[ind])
    elif cosmology.lower() == "planck15_lal":
        return Planck15_lal_cosmology()
    elif "_with_riess2019_h0" in cosmology.lower():
        base_cosmology = cosmology.lower().split("_with_riess2019_h0")[0]
        return Riess2019_H0_cosmology(base_cosmology)


def Planck15_lal_cosmology():
    """Return the Planck15 cosmology coded up in lalsuite
    """
    return cosmo.LambdaCDM(H0=67.90, Om0=0.3065, Ode0=0.6935)


def Riess2019_H0_cosmology(base_cosmology):
    """Return the base cosmology but with the Riess2019 H0 value. For details
    see https://arxiv.org/pdf/1903.07603.pdf.

    Parameters
    ----------
    base_cosmology: str
        name of cosmology to use as the base
    """
    _base_cosmology = get_cosmology(base_cosmology)
    return cosmo.LambdaCDM(
        H0=74.03, Om0=_base_cosmology.Om0, Ode0=_base_cosmology.Ode0
    )
