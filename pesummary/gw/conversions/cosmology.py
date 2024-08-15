# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.gw.cosmology import get_cosmology
from pesummary.utils.utils import logger
from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from astropy.cosmology import z_at_value
    import astropy.units as u
except ImportError:
    pass


def _wrapper_for_z_from_quantity_exact(args):
    """Wrapper function for _z_from_quantity_exact for a pool of workers
Parameters
    ----------
    args: tuple
        All args passed to _z_from_quantity_exact
    """
    return _z_from_quantity_exact(*args)


def _z_from_quantity_exact(samples, quantity, unit, cosmology):
    """Return the redshift given samples for a cosmological quantity, assuming
    a given cosmology

    Parameters
    ----------
    samples: float/np.array
        samples for a cosmological quantity, e.g. luminosity_distance
    quantity: str
        name of the quantity you wish to compute. This must be an attribute of
        cosmology
    unit: astropy.units
        unit of the samples
    cosmology: astropy.cosmology.LambdaCDM
        the cosmology to use for conversions
    """
    _z = z_at_value(getattr(cosmology, quantity), samples * unit)
    return _z.value


@array_input(ignore_kwargs=["cosmology", "multi_process"])
def z_from_quantity_exact(
    samples, quantity, unit, cosmology="Planck15", multi_process=1
):
    """Return the redshift given samples for a cosmological quantity
    """
    import multiprocessing
    from pesummary.utils.utils import iterator

    logger.warning(
        "Estimating the exact redshift for every {}. This may take a few "
        "minutes".format(quantity)
    )
    cosmo = get_cosmology(cosmology)
    args = np.array(
        [
            samples, [quantity] * len(samples), [unit] * len(samples),
            [cosmo] * len(samples)
        ], dtype=object
    ).T
    with multiprocessing.Pool(multi_process) as pool:
        z = np.array(
            list(
                iterator(
                    pool.imap(_wrapper_for_z_from_quantity_exact, args),
                    tqdm=True, desc="Calculating redshift", logger=logger,
                    total=len(samples)
                )
            )
        )
    return z


def _z_from_quantity_approx(samples, quantity, unit, cosmology, N=100, **kwargs):
    """Return the redshift given samples for a cosmological quantity, assuming
    a given cosmology. This technique uses interpolation to estimate the
    redshift

    Parameters
    ----------
    samples: float/np.array
        samples for a cosmological quantity, e.g. luminosity_distance
    quantity: str
        name of the quantity you wish to compute. This must be an attribute of
        cosmology
    unit: astropy.units
        unit of the samples
    cosmology: astropy.cosmology.LambdaCDM
        the cosmology to use for conversions
    """
    logger.warning("The redshift is being approximated using interpolation. "
                   "Bear in mind that this does introduce a small error.")
    cosmo = get_cosmology(cosmology)
    zmin = _z_from_quantity_exact(np.min(samples), quantity, unit, cosmo)
    zmax = _z_from_quantity_exact(np.max(samples), quantity, unit, cosmo)
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), N)
    grid = _quantity_from_z(zgrid, quantity, cosmology)
    zvals = np.interp(samples, grid, zgrid)
    return zvals


@array_input(ignore_kwargs=["cosmology", "multi_process"])
def z_from_dL_exact(luminosity_distance, cosmology="Planck15", multi_process=1):
    """Return the redshift given samples for the luminosity distance
    """
    return z_from_quantity_exact(
        luminosity_distance, "luminosity_distance", u.Mpc, cosmology=cosmology,
        multi_process=multi_process
    )


@array_input(ignore_kwargs=["N", "cosmology"])
def z_from_dL_approx(
    luminosity_distance, N=100, cosmology="Planck15", **kwargs
):
    """Return the approximate redshift given samples for the luminosity
    distance. This technique uses interpolation to estimate the redshift
    """
    return _z_from_quantity_approx(
        luminosity_distance, "luminosity_distance", u.Mpc, cosmology, N=N,
        **kwargs
    )


@array_input(ignore_kwargs=["cosmology", "multi_process"])
def z_from_comoving_volume_exact(
    comoving_volume, cosmology="Planck15", multi_process=1):
    """Return the redshift given samples for the comoving volume
    """
    return z_from_quantity_exact(
        comoving_volume, "comoving_volume", u.Mpc**3, cosmology=cosmology,
        multi_process=multi_process
    )


@array_input(ignore_kwargs=["N", "cosmology"])
def z_from_comoving_volume_approx(
    comoving_volume, N=100, cosmology="Planck15", **kwargs
):
    """Return the approximate redshift given samples for the comoving volume.
    This technique uses interpolation to estimate the redshift
    """
    return _z_from_quantity_approx(
        comoving_volume, "comoving_volume", u.Mpc**3, cosmology, N=N, **kwargs
    )


def _quantity_from_z(redshift, quantity, cosmology="Planck15"):
    """Return a cosmological quantify given samples for the redshift
    """
    cosmo = get_cosmology(cosmology)
    return getattr(cosmo, quantity)(redshift).value


@array_input(ignore_kwargs=["cosmology"])
def dL_from_z(redshift, cosmology="Planck15"):
    """Return the luminosity distance given samples for the redshift
    """
    return _quantity_from_z(redshift, "luminosity_distance", cosmology=cosmology)


@array_input(ignore_kwargs=["cosmology"])
def comoving_distance_from_z(redshift, cosmology="Planck15"):
    """Return the comoving distance given samples for the redshift
    """
    return _quantity_from_z(redshift, "comoving_distance", cosmology=cosmology)


@array_input(ignore_kwargs=["cosmology"])
def comoving_volume_from_z(redshift, cosmology="Planck15"):
    """Return the comoving volume given samples for the redshift
    """
    return _quantity_from_z(redshift, "comoving_volume", cosmology=cosmology)


def _source_from_detector(parameter, z):
    """Return the source-frame parameter given samples for the detector-frame parameter
    and the redshift
    """
    return parameter / (1. + z)


def _detector_from_source(parameter, z):
    """Return the detector-frame parameter given samples for the source-frame parameter
    and the redshift
    """
    return parameter * (1. + z)


@array_input()
def m1_source_from_m1_z(mass_1, z):
    """Return the source-frame primary mass given samples for the
    detector-frame primary mass and the redshift
    """
    return _source_from_detector(mass_1, z)


@array_input()
def m1_from_m1_source_z(mass_1_source, z):
    """Return the detector-frame primary mass given samples for the
    source-frame primary mass and the redshift
    """
    return _detector_from_source(mass_1_source, z)


@array_input()
def m2_source_from_m2_z(mass_2, z):
    """Return the source-frame secondary mass given samples for the
    detector-frame secondary mass and the redshift
    """
    return _source_from_detector(mass_2, z)


@array_input()
def m2_from_m2_source_z(mass_2_source, z):
    """Return the detector-frame secondary mass given samples for the
    source-frame secondary mass and the redshift
    """
    return _detector_from_source(mass_2_source, z)


@array_input()
def m_total_source_from_mtotal_z(total_mass, z):
    """Return the source-frame total mass of the binary given samples for
    the detector-frame total mass and redshift
    """
    return _source_from_detector(total_mass, z)


@array_input()
def mtotal_from_mtotal_source_z(total_mass_source, z):
    """Return the detector-frame total mass of the binary given samples for
    the source-frame total mass and redshift
    """
    return _detector_from_source(total_mass_source, z)


@array_input()
def mchirp_source_from_mchirp_z(mchirp, z):
    """Return the source-frame chirp mass of the binary given samples for
    detector-frame chirp mass and redshift
    """
    return _source_from_detector(mchirp, z)


@array_input()
def mchirp_from_mchirp_source_z(mchirp_source, z):
    """Return the detector-frame chirp mass of the binary given samples for
    the source-frame chirp mass and redshift
    """
    return _detector_from_source(mchirp_source, z)


class Redshift(object):
    class Distance(object):
        exact = z_from_dL_exact
        approx = z_from_dL_approx

    class ComovingVolume(object):
        exact = z_from_comoving_volume_exact
        approx = z_from_comoving_volume_exact
