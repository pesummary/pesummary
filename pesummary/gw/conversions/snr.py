# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


@array_input
def _ifo_snr(IFO_abs_snr, IFO_snr_angle):
    """Return the matched filter SNR for a given IFO given samples for the
    absolute SNR and the angle
    """
    return IFO_abs_snr * np.cos(IFO_snr_angle)


@array_input
def _ifo_snr_from_real_and_imaginary(IFO_real_snr, IFO_imag_snr):
    """Return the matched filter SNR for a given IFO given samples for the
    real and imaginary SNR
    """
    _complex = IFO_real_snr + IFO_imag_snr * 1j
    _abs = np.abs(_complex)
    return _ifo_snr(_abs, np.angle(_complex))


@array_input
def network_snr(snrs):
    """Return the network SNR for N IFOs

    Parameters
    ----------
    snrs: list
        list of numpy.array objects containing the snrs samples for a particular
        IFO
    """
    squares = np.square(snrs)
    network_snr = np.sqrt(np.sum(squares, axis=0))
    return network_snr


@array_input
def network_matched_filter_snr(IFO_matched_filter_snrs, IFO_optimal_snrs):
    """Return the network matched filter SNR for a given detector network. Code
    adapted from Christopher Berry's python notebook

    Parameters
    ----------
    IFO_matched_filter_snrs: list
        list of matched filter SNRs for each IFO in the network
    IFO_optimal_snrs: list
        list of optimal SNRs
    """
    for num, det_snr in enumerate(IFO_matched_filter_snrs):
        complex_snr = np.iscomplex(det_snr)
        convert_mf_snr = False
        try:
            if complex_snr:
                convert_mf_snr = True
        except ValueError:
            if any(_complex for _complex in complex_snr):
                convert_mf_snr = True
        if convert_mf_snr:
            IFO_matched_filter_snrs[num] = np.real(det_snr)
    network_optimal_snr = network_snr(IFO_optimal_snrs)
    network_matched_filter_snr = np.sum(
        [
            mf_snr * opt_snr / network_optimal_snr for mf_snr, opt_snr in zip(
                IFO_matched_filter_snrs, IFO_optimal_snrs
            )
        ], axis=0
    )
    return network_matched_filter_snr
