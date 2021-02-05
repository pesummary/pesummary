# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.cosmology import get_cosmology, available_cosmologies
import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestCosmology(object):
    """Test the get_cosmology function as part of the `pesummary.gw.cosmology`
    package
    """
    def test_invalid_string(self):
        """Test that a ValueError is passed when an invalid string is passed
        """
        with pytest.raises(ValueError):
            cosmology = get_cosmology(cosmology="random")

    def test_lal_cosmology(self):
        """Test that the correct values are stored for the lal cosmology
        """
        lal_values = dict(H0=67.90, Om0=0.3065, Ode0=0.6935)
        cosmology = get_cosmology(cosmology="planck15_lal")
        for param, value in lal_values.items():
            if param == "H0":
                assert lal_values[param] == getattr(cosmology, param).value
            else:
                assert lal_values[param] == getattr(cosmology, param)

    def test_astropy_cosmology(self):
        """Test that the astropy cosmology is correct
        """
        from astropy import cosmology

        for cosmo in cosmology.parameters.available:
            _cosmo = get_cosmology(cosmology=cosmo)
            astropy_cosmology = getattr(cosmology, cosmo)
            for key, value in vars(_cosmo).items():
                assert vars(astropy_cosmology)[key] == value

    def test_Riess2019_H0(self):
        """Test that the Riess2019 H0 cosmology is correct
        """
        riess_H0 = 74.03
        for cosmo in available_cosmologies:
            if "riess" not in cosmo:
                continue
            _cosmo = get_cosmology(cosmology=cosmo)
            base_cosmo = cosmo.split("_with_riess2019_h0")[0]
            _base_cosmo = get_cosmology(cosmology=base_cosmo)
            assert _cosmo.H0.value == riess_H0
            for key in ["Om0", "Ode0"]:
                assert getattr(_base_cosmo, key) == getattr(_cosmo, key)
