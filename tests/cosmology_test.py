from pesummary.gw.cosmology import get_cosmology
import pytest


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
