# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from ..utils.utils import logger
from ..core.reweight import rejection_sampling, options
from .cosmology import get_cosmology

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def uniform_in_comoving_volume_from_uniform_in_volume(
    samples, redshift_method="exact", cosmology="Planck15", convert_kwargs={},
    star_formation_rate_power=0, **kwargs
):
    """Resample a table of posterior distributions from a uniform in volume
    distance prior to a uniform in comoving volume distance prior. For details
    see Appendix C of https://arxiv.org/abs/2010.14527

    Parameters
    ----------
    samples: pesummary.utils.samples_dict.SamplesDict
        table of posterior distributions you wish to resample
    redshift_method: str, optional
        method to use when generating a 'redshift' posterior distribution from
        the 'luminosity_distance' posterior distribution. This is only used
        when 'redshift' samples are not found in 'samples'. Default "exact"
    cosmology: str, optional
        cosmology you wish to use for reweighting. Default "Planck15"
    covert_kwargs: dict, optional
        kwargs to pass to pesummary.gw.conversions.dL_from_z when calculating
        the 'luminosity_distance' posterior from the 'redshift' posterior
        or kwargs to pass to pesummary.gw.conversions.z_from_dL* when
        calculating the 'redshift' posterior from the 'luminosity_distance'
        posterior
    star_formation_rate_power: int, optional
        power to use to include a star formation rate evolution. Default 0,
        i.e. no evolution
    """
    import astropy.units as u
    parameters = samples.keys()
    if "redshift" not in parameters and "luminosity_distance" in parameters:
        from pesummary.gw.conversions import Redshift
        logger.info(
            "Unable to find samples for 'redshift'. Calculating them from "
            "the 'luminosity_distance' posterior distribution"
        )
        luminosity_distance = samples["luminosity_distance"]
        redshift = getattr(Redshift, redshift_method)(
            luminosity_distance, cosmology=cosmology, **convert_kwargs
        )
    elif "redshift" not in parameters:
        raise ValueError(
            "Unable to reweight to uniform in comoving volume prior because "
            "unable to find samples for 'redshift' and 'luminosity_distance'"
        )
    elif "redshift" in parameters and "luminosity_distance" not in parameters:
        from pesummary.gw.conversions import dL_from_z
        logger.info(
            "Unable to find samples for 'luminosity_distance'. Calculating "
            "them from the 'redshift' posterior distribution"
        )
        redshift = samples["redshift"]
        luminosity_distance = dL_from_z(
            redshift, cosmology=cosmology, **convert_kwargs
        )
    else:
        redshift = samples["redshift"]
        luminosity_distance = samples["luminosity_distance"]
    cosmology = get_cosmology(cosmology)
    hubble_distance = (u.cds.c / cosmology.H0).to_value(unit=u.Mpc)
    hubble_parameter = cosmology.efunc(redshift)
    weights = 1.0 / (
        (1 + redshift)**(2. - star_formation_rate_power) * (
            hubble_parameter * (luminosity_distance / hubble_distance)
            + (1. + redshift)**2.
        )
    )
    return rejection_sampling(samples, weights)


options.update(
    {
        "uniform_in_comoving_volume": (
            uniform_in_comoving_volume_from_uniform_in_volume
        )
    }
)
