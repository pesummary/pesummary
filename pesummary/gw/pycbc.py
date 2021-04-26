# Licensed under an MIT style license -- see LICENSE.md

error_msg = (
    "Unable to install '{}'. You will not be able to use some of the inbuilt "
    "functions."
)
from pesummary.utils.utils import logger
import numpy as np
try:
    import pycbc
except ImportError:
    logger.warn(error_msg.format("pycbc"))

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def optimal_snr(
    template, psd, low_frequency_cutoff=20., high_frequency_cutoff=1024.
):
    """Calculate the loudness of the waveform. See Duncan Brown’s thesis for
    details

    Parameters
    ----------
    template: pycbc.type.frequencyseries.FrequencySeries
        waveform you wish to calculate the loudness for
    psd: pycbc.type.frequencyseries.FrequencySeries
        psd to use when calculating the integral
    low_frequency_cutoff: float, optional
        low frequency cut-off to start calculating the integral
    """
    from pycbc.filter import sigma
    return sigma(
        template, psd, low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff
    )


def compute_the_overlap(template, data, psd, **kwargs):
    """Wrapper for pycbc.filter.overlap_cplx

    Parameters
    ----------
    template: pycbc.types.frequencyseries.FrequencySeries
        frequency domain template
    data: pycbc.types.frequencyseries.FrequencySeries
        frequency domain data
    psd: pycbc.types.frequencyseries.FrequencySeries
        the PSD to use when computing the overlap
    **kwargs: dict, optional
        all additional kwargs passed to pycbc.filter.overlap_cplx
    """
    from pycbc.filter import overlap_cplx
    return overlap_cplx(template, data, psd=psd, **kwargs)


def pycbc_default_psd(
    mass_1=None, mass_2=None, chi_eff=None, spin_1z=None, spin_2z=None, psd=None,
    duration=None, df=None, detectors=["H1", "L1"], f_low=20., f_final=1024.
):
    """Return a dictionary of pycbc psds for a given detector network

    Parameters
    ----------
    mass_1: np.ndarray, optional
        array of primary masses to use when working out the duration and
        consequently delta_f for the PSD
    mass_2: np.ndarray, optional
        array of secondary masses to use when working out the duration and
        consequently delta_f for the PSD
    chi_eff: np.ndarray, optional
        array of effective spin samples to use when working out the duration and
        consequently delta_f for the psd. Used as an approximant when spin_1z
        and spin_2z samples are not provided
    spin_1z: np.ndarray, optional
        array of samples for the primary spin parallel to the orbital angular
        momentum to use when computing the duration and consequently delta_f for
        the psd
    spin_2z: np.ndarray, optional
        array of samples for the secondary spin parallel to the orbital angular
        momentum to use when computing the duration and consequently delta_f for
        the psd
    psd: func, optional
        pycbc function to use when calculating the default psd. Default is
        aLIGOZeroDetHighPower
    duration: np.npdarray
        array containing the durations for a given set of waveform. The maximum
        duration is then used to calculate the delta_f for the PSD
    detectors: list, optional
        the detector network you wish to calculate psds for
    f_low: float, optional
        the low frequency to start computing the psd from. Default 20Hz
    f_final: float, optional
        the highest frequency to finish computing the psd. Default 1024Hz
    """
    from pycbc import pnutils

    if psd is None:
        from pycbc.psd import aLIGOZeroDetHighPower

        psd = aLIGOZeroDetHighPower
    elif isinstance(psd, str):
        import pycbc.psd

        psd = getattr(pycbc.psd, psd)

    logger.warn("No PSD provided. Using '{}' for the psd".format(psd.__name__))
    _required = [mass_1, mass_2]
    if df is None:
        cond1 = all(i is not None for i in _required + [spin_1z, spin_2z])
        cond2 = all(i is not None for i in _required + [chi_eff])
        if cond1 and duration is None:
            duration = pnutils.get_imr_duration(
                mass_1, mass_2, spin_1z, spin_2z, f_low
            )
        elif cond2 and duration is None:
            logger.warn(
                "Could not find samples for spin_1z and spin_2z. We will "
                "assume that spin_1z = spin_2z = chi_eff to estimate the "
                "maximum IMR duration. This is used to estimate delta_f for "
                "the PSD."
            )
            duration = pnutils.get_imr_duration(
                mass_1, mass_2, chi_eff, chi_eff, f_low
            )
        elif duration is None:
            raise ValueError(
                "Please provide either 'spin_1z' and 'spin_2z' or 'chi_eff' "
                "samples to estimate the maximum IMR duration. This is used to "
                "estimate delta_f for the PSD."
            )
        duration = np.max(duration)
        t_len = 2**np.ceil(np.log2(duration) + 1)
        df = 1. / t_len
    flen = int(f_final / df) + 1
    _psd = psd(flen, df, f_low)
    setattr(_psd, "__name__", psd.__name__)
    return {ifo: _psd for ifo in detectors}
