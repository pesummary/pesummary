# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.decorators import array_input
from pesummary.utils.utils import logger
from pesummary.gw.pycbc import optimal_snr, compute_the_overlap
from pesummary.gw.conversions.angles import _dphi, _dpsi

__author__ = [
    "Stephen Fairhurst <stephen.fairhurst@ligo.org>",
    "Rhys Green <rhys.green@ligo.org>",
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


@array_input()
def _ifo_snr(IFO_abs_snr, IFO_snr_angle):
    """Return the matched filter SNR for a given IFO given samples for the
    absolute SNR and the angle
    """
    return IFO_abs_snr * np.cos(IFO_snr_angle)


@array_input()
def _ifo_snr_from_real_and_imaginary(IFO_real_snr, IFO_imag_snr):
    """Return the matched filter SNR for a given IFO given samples for the
    real and imaginary SNR
    """
    _complex = IFO_real_snr + IFO_imag_snr * 1j
    _abs = np.abs(_complex)
    return _ifo_snr(_abs, np.angle(_complex))


@array_input()
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


@array_input()
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


def _setup_psd(psd, psd_default, **psd_default_kwargs):
    """Setup the PSD dictionary. If the provided PSD is empty, construct a
    PSD based on the default, this could either be analytic or not. If the
    provided PSD is not empty, simply return the provided PSD unchanged.

    Parameters
    ----------
    psd: dict
        dictionary containing the psd. Keys are the IFO and items are a
        pycbc.frequencyseries.FrequencySeries object
    psd_default: str/dict
        The default PSD to use. This can either be a string describing the
        analytic PSD or a dictionary with keyes showing the IFO and items
        either a pesummary.gw.file.psd.PSD or a
        pycbc.frequencyseries.FrequencySeries object
    psd_default_kwargs: dict, optional
        kwargs to pass to pesummary.gw.file.psd.pycbc_default_psd when
        a PSD is constructured based on the analytic default
    """
    from pesummary.gw.file.psd import PSD
    ANALYTIC = False
    condition = isinstance(psd_default, dict) and len(psd_default) and (
        all(isinstance(value, PSD) for ifo, value in psd_default.items())
    )
    f_low = psd_default_kwargs.get("f_low", None)
    f_final = psd_default_kwargs.get("f_final", None)
    frequency_cond = any(param is None for param in [f_low, f_final])
    if psd == {} and condition:
        if frequency_cond:
            raise ValueError(
                "Please provide f_low and f_final as keyword arguments"
            )
        for ifo, data in psd_default.items():
            psd[ifo] = data.to_pycbc(
                f_low, f_high=f_final, f_high_override=True
            )
    elif psd == {} and isinstance(psd_default, dict) and len(psd_default):
        for ifo, data in psd_default.items():
            psd[ifo] = data
    elif psd == {}:
        from pesummary.gw.pycbc import pycbc_default_psd
        if isinstance(psd_default, dict):
            from pesummary import conf
            psd_default = conf.psd
        ANALYTIC = True
        psd_default_kwargs.update({"psd": psd_default})
        psd = pycbc_default_psd(**psd_default_kwargs)
    return psd, ANALYTIC


def _make_waveform(
    approx, theta_jn, phi_jl, phase, psi_J, mass_1, mass_2, tilt_1, tilt_2,
    phi_12, a_1, a_2, beta, distance, **kwargs
):
    """Generate a frequency domain waveform

    Parameters
    ----------
    approx: str
        Name of the approximant you wish to use when generating the waveform
    theta_jn: float
        Angle between the total angular momentum and the line of sight
    phi_jl: float
        Azimuthal angle of the total orbital angular momentum around the
        total angular momentum
    phase: float
        The phase of the binary at coaelescence
    mass_1: float
        Primary mass of the binary
    mass_2: float
        Secondary mass of the binary
    tilt_1: float
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: float
        The angle between the total orbital angular momentum and the primary
        spin
    phi_12: float
        The angle between the primary spin and the secondary spin
    a_1: float
        The spin magnitude on the larger object
    a_2: float
        The spin magnitude on the secondary object
    beta: float
        The opening angle of the system. Defined as the angle between the
        orbital angular momentum, L, and the total angular momentum J.
    **kwargs: dict
        All additional kwargs are passed to the
        pesummary.gw.waveform.fd_waveform function
    """
    from pesummary.gw.waveform import fd_waveform
    _samples = {
        "theta_jn": [theta_jn], "phi_jl": [phi_jl], "phase": [phase],
        "mass_1": [mass_1], "mass_2": [mass_2], "tilt_1": [tilt_1],
        "tilt_2": [tilt_2], "phi_12": [phi_12], "a_1": [a_1],
        "a_2": [a_2], "luminosity_distance": [distance]
    }
    waveforms = fd_waveform(
        _samples, "IMRPhenomPv2", kwargs.get("df", 1. / 256),
        kwargs.get("f_low", 20.), kwargs.get("f_final", 1024.),
        f_ref=kwargs.get("f_ref", 20.), ind=0, pycbc=True
    )
    hp, hc = waveforms["h_plus"], waveforms["h_cross"]
    if kwargs.get("flen", None) is not None:
        flen = kwargs.get("flen")
        hp.resize(flen)
        hc.resize(flen)
    dpsi = _dpsi(theta_jn, phi_jl, beta)
    fp = np.cos(2 * (psi_J - dpsi))
    fc = -1. * np.sin(2 * (psi_J - dpsi))
    h = (fp * hp + fc * hc)
    h *= np.exp(2j * _dphi(theta_jn, phi_jl, beta))
    return h


def _calculate_precessing_harmonics(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, beta, distance,
    harmonics=[0, 1], approx="IMRPhenomPv2", **kwargs
):
    """Decompose a precessing waveform into a series of harmonics as defined
    in Fairhurst et al. arXiv:1908.05707

    Parameters
    ----------
    mass_1: np.ndarray
        Primary mass of the bianry
    mass_2: np.ndarray
        Secondary mass of the binary
    a_1: np.ndarray
        The spin magnitude on the larger object
    a_2: np.ndarray
        The spin magnitude on the secondary object
    tilt_1: np.ndarray
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: np.ndarray
        The angle between the total orbital angular momentum and the secondary
        spin
    phi_12: np.ndarray
        The angle between the primary spin and the secondary spin
    beta: np.ndarray
        The angle between the total angular momentum and the total orbital
        angular momentum
    harmonics: list, optional
        List of harmonics which you wish to calculate. Default [0, 1]
    approximant: str, optional
        Approximant to use for the decomposition. Default IMRPhenomPv2
    """
    harm = {}
    if (0 in harmonics) or (4 in harmonics):
        h0 = _make_waveform(
            approx, 0., 0., 0., 0., mass_1, mass_2, tilt_1, tilt_2,
            phi_12, a_1, a_2, beta, distance, **kwargs
        )
        hpi4 = _make_waveform(
            approx, 0., 0., np.pi / 4, np.pi / 4, mass_1, mass_2,
            tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, **kwargs
        )
        if (0 in harmonics):
            harm[0] = (h0 - hpi4) / 2
        if (4 in harmonics):
            harm[4] = (h0 + hpi4) / 2
    if (1 in harmonics) or (3 in harmonics):
        h0 = _make_waveform(
            approx, np.pi / 2, 0., np.pi / 4, np.pi / 4, mass_1, mass_2,
            tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance,
            **kwargs
        )
        hpi2 = _make_waveform(
            approx, np.pi / 2, np.pi / 2, 0., np.pi / 4, mass_1, mass_2,
            tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance,
            **kwargs
        )
        if (1 in harmonics):
            harm[1] = -1. * (h0 + hpi2) / 4
        if (3 in harmonics):
            harm[3] = -1. * (h0 - hpi2) / 4
    if (2 in harmonics):
        h0 = _make_waveform(
            approx, np.pi / 2, 0., 0., 0., mass_1, mass_2,
            tilt_1, tilt_2, phi_12, a_1, a_2, beta,
            distance, **kwargs
        )
        hpi2 = _make_waveform(
            approx, np.pi / 2, np.pi / 2, 0., 0., mass_1, mass_2,
            tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance,
            **kwargs
        )
        harm[2] = (h0 + hpi2) / 6
    return harm


def _make_waveform_from_precessing_harmonics(
    harmonic_dict, theta_jn, phi_jl, phase, f_plus_j, f_cross_j
):
    """Generate waveform for a binary merger with given precessing harmonics and
    orientation

    Parameters
    ----------
    harmonic_dict: dict
        harmonics to include
    theta_jn: np.ndarray
        the angle between total angular momentum and line of sight
    phi_jl: np.ndarray
        the initial polarization phase
    phase: np.ndarray
        the initial orbital phase
    psi_J: np.ndarray
        the polarization angle in the J-aligned frame
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned
        frame
    """
    A = _harmonic_amplitudes(
        theta_jn, phi_jl, f_plus_j, f_cross_j, harmonic_dict
    )
    h_app = 0
    for k, harm in harmonic_dict.items():
        if h_app:
            h_app += A[k] * harm
        else:
            h_app = A[k] * harm
    h_app *= np.exp(2j * phase + 2j * phi_jl)
    return h_app


def _harmonic_amplitudes(
    theta_jn, phi_jl, f_plus_j, f_cross_j, harmonics=[0, 1]
):
    """Calculate the amplitudes of the precessing harmonics as a function of
    orientation

    Parameters
    ----------
    theta_jn: np.ndarray
        the angle between J and line of sight
    phi_jl: np.ndarray
        the precession phase
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned
        frame
    harmonics: list, optional
        The list of harmonics you wish to return. Default is [0, 1]
    """
    amp = {}
    if 0 in harmonics:
        amp[0] = (
            (1 + np.cos(theta_jn)**2) / 2 * f_plus_j
            - 1j * np.cos(theta_jn) * f_cross_j
        )
    if 1 in harmonics:
        amp[1] = 2 * np.exp(-1j * phi_jl) * (
            np.sin(theta_jn) * np.cos(theta_jn) * f_plus_j
            - 1j * np.sin(theta_jn) * f_cross_j
        )
    if 2 in harmonics:
        amp[2] = 3 * np.exp(-2j * phi_jl) * (np.sin(theta_jn)**2) * f_plus_j
    if 3 in harmonics:
        amp[3] = 2 * np.exp(-3j * phi_jl) * (
            -np.sin(theta_jn) * np.cos(theta_jn) * f_plus_j
            - 1j * np.sin(theta_jn) * f_cross_j
        )
    if 4 in harmonics:
        amp[4] = np.exp(-4j * phi_jl) * (
            (1 + np.cos(theta_jn)**2) / 2 * f_plus_j
            + 1j * np.cos(theta_jn) * f_cross_j
        )
    return amp


def _prec_ratio(theta_jn, phi_jl, psi_J, b_bar, ra, dec, time, detector):
    """Calculate the ratio between the leading and first precession terms: Zeta
    as defined in Fairhurst et al. arXiv:1908.05707

    Parameters
    ----------
    theta_jn: float
        The angle between the total angular momentum and the line of sight
    phi_jl: np.ndarray
        the precession phase
    psi_J: float
        The polarization of the binary defined with respect to the total
        angular momentum
    b_bar: float
        Tangent of the average angle between the total angular momentum and the
        total orbital angular momentum during inspiral
    ra: float
        The right ascension of the binary
    dec: float
        The declinartion of the binary
    time: float
        The merger time of the binary
    detector: str
        The name of the detector you wish to calculate the ratio for
    """
    from pesummary.gw.waveform import antenna_response
    samples = {"ra": [ra], "dec": [dec], "psi": [psi_J], "geocent_time": [time]}
    f_plus, f_cross = antenna_response(samples, detector)
    ratio = _prec_ratio_plus_cross(theta_jn, phi_jl, f_plus, f_cross, b_bar)
    return ratio


def _prec_ratio_plus_cross(theta_jn, phi_jl, f_plus, f_cross, b_bar):
    """Calculate the ratio between the leading and first precession harmonics
    given an antenna response pattern. Zeta as defined in Fairhurst et al.
    arXiv:1908.05707

    Parameters
    ----------
    theta_jn: float/np.ndarray
        The angle between the total angular momentum and the line of sight
    phi_jl: np.ndarray
        the precession phase
    f_plus: float/np.ndarray
        The plus polarization factor for a given sky location / orientation
    f_cross: float/np.ndarray
        The cross polarization factor for a given sky location / orientation
    b_bar: float/np.ndarray
        Tangent of the average angle between the total angular momentum and the
        total orbital angular momentum during inspiral
    """
    amplitudes = _harmonic_amplitudes(
        theta_jn, phi_jl, f_plus, f_cross, harmonics=[0, 1]
    )
    A0 = amplitudes[0]
    A1 = amplitudes[1]
    return b_bar * A1 / A0


@array_input(
    ignore_kwargs=[
        "f_low", "psd", "approx", "f_final", "f_ref", "return_data_used",
        "multi_process", "duration", "df", "psd_default", "debug"
    ]
)
def precessing_snr(
    mass_1, mass_2, beta, psi_J, a_1, a_2, tilt_1, tilt_2, phi_12, theta_jn,
    ra, dec, time, phi_jl, distance, phase, f_low=20., psd={}, spin_1z=None,
    spin_2z=None, chi_eff=None, approx="IMRPhenomPv2", f_final=1024.,
    f_ref=None, return_data_used=False, multi_process=6., duration=None,
    df=1. / 256, psd_default="aLIGOZeroDetHighPower", debug=True
):
    """Calculate the precessing snr as defined in Fairhurst et al.
    arXiv:1908.05707

    Parameters
    ----------
    mass_1: np.ndarray
        Primary mass of the bianry
    mass_2: np.ndarray
        Secondary mass of the binary
    beta: np.ndarray
        The angle between the total angular momentum and the total orbital
        angular momentum
    psi_J: np.ndarray
        The polarization angle as defined with respect to the total angular
        momentum
    a_1: np.ndarray
        The spin magnitude on the larger object
    a_2: np.ndarray
        The spin magnitude on the secondary object
    tilt_1: np.ndarray
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: np.ndarray
        The angle between the total orbital angular momentum and the secondary
        spin
    phi_12: np.ndarray
        The angle between the primary spin and the secondary spin
    theta_jn: np.ndarray
        The angle between the total orbital angular momentum and the line of
        sight
    ra: np.ndarray
        The right ascension of the source
    dec: np.ndarray
        The declination of the source
    time: np.ndarray
        The merger time of the binary
    phi_jl: np.ndarray
        the precession phase
    f_low: float, optional
        The low frequency cut-off to use for integration. Default is 20Hz
    psd: dict, optional
        Dictionary of pycbc.types.frequencyseries.FrequencySeries objects, one
        for each detector. Default is to use the aLIGOZeroDetHighPower PSD
    spin_1z: np.ndarray, optional
        The primary spin aligned with the total orbital angular momentum
    spin_2z: np.ndarray, optional
        The secondary spin aligned with the total orbital angular momentum
    chi_eff: np.ndarray, optional
        Effective spin of the binary
    approx: str, optional
        The approximant you wish to use. Default IMRPhenomPv2
    f_final: float, optional
        Final frequency to use for integration. Default 1024Hz
    f_ref: float, optional
        Reference frequency where the spins are defined. Default is f_low
    return_data_used: Bool, optional
        if True, return a dictionary containing information about what data was
        used. Default False
    multi_process: int, optional
        The number of cpus to use when computing the precessing_snr. Default 6
    duration: float, optional
        maximum IMR duration to use to estimate delta_f when PSD is not
        provided.
    debug: Bool, optional
        if True, return posteriors for b_bar and the overlap between the 0th
        and 1st harmonics. These are useful for debugging.
    """
    from pesummary.gw.file.psd import PSD
    from pesummary.gw.waveform import antenna_response
    from pesummary.utils.utils import iterator
    import multiprocessing

    if isinstance(f_low, (list, np.ndarray)):
        f_low = f_low[0]
    psd, ANALYTIC = _setup_psd(
        psd, psd_default, mass_1=mass_1, mass_2=mass_2, spin_1z=spin_1z,
        spin_2z=spin_2z, chi_eff=chi_eff, f_low=f_low, duration=duration,
        detectors=["H1", "L1"], f_final=f_final, df=df
    )
    detectors = list(psd.keys())
    if df != psd[detectors[0]].delta_f:
        from pycbc.psd import estimate
        logger.warn(
            "Provided PSD has df={} and {} has been specified. Interpolation "
            "will be used".format(psd[detectors[0]].delta_f, df)
        )
        psd = {
            ifo: estimate.interpolate(psd[ifo], df) for ifo in psd.keys()
        }
    _f_final = psd[detectors[0]].sample_frequencies[-1]
    if f_final is None:
        f_final = _f_final
    elif f_final != _f_final:
        logger.warn(
            "The provided final frequency: {} does not match the final "
            "frequency in the PSD: {}. Using the final frequency stored in the "
            "PSD to prevent interpolation errors".format(f_final, _f_final)
        )
        f_final = _f_final
    if f_ref is None:
        logger.warn("No reference frequency provided. Using f_low as default")
        f_ref = f_low
    elif isinstance(f_ref, (list, np.ndarray)):
        f_ref = f_ref[0]

    flen = int(f_final / df) + 1
    _samples = {"ra": ra, "dec": dec, "psi": psi_J, "geocent_time": time}
    antenna = {
        detector: antenna_response(_samples, detector) for detector in detectors
    }
    _f_plus_j = {key: value[0] for key, value in antenna.items()}
    _f_cross_j = {key: value[1] for key, value in antenna.items()}
    with multiprocessing.Pool(multi_process) as pool:
        f_plus_j = np.array(
            [dict(zip(_f_plus_j, item)) for item in zip(*_f_plus_j.values())]
        )
        f_cross_j = np.array(
            [dict(zip(_f_cross_j, item)) for item in zip(*_f_cross_j.values())]
        )
        dphi = _dphi(theta_jn, phi_jl, beta)
        args = np.array([
            mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12,
            theta_jn, beta, psi_J, ra, dec, time, [approx] * len(mass_1),
            [psd] * len(mass_1), [detectors] * len(mass_1), phi_jl, distance,
            phase - dphi, f_plus_j, f_cross_j, [f_low] * len(mass_1),
            [df] * len(mass_1), [f_final] * len(mass_1), [flen] * len(mass_1),
            [f_ref] * len(mass_1), [debug] * len(mass_1)
        ], dtype=object).T

        rho_p = np.array(
            list(
                iterator(
                    pool.imap(_wrapper_for_precessing_snr, args), tqdm=True,
                    desc="Calculating rho_p", logger=logger, total=len(mass_1)
                )
            ), dtype=object
        )

    if debug:
        rho_ps, b_bars, overlaps, snrs = {}, {}, {}, {}
        for num, _dict in enumerate([rho_ps, b_bars, overlaps, snrs]):
            for key in rho_p[0][0]:
                _dict[key] = np.nan_to_num(
                    [dictionary[num][key] for dictionary in rho_p], 0
                )
        _return = [
            np.sqrt(np.sum([_dict[i] for i in detectors], axis=0)) if num == 0
            or num == 3 else np.mean([_dict[i] for i in detectors], axis=0) for
            num, _dict in enumerate([rho_ps, b_bars, overlaps, snrs])
        ]
    else:
        rho_p = {
            key: np.nan_to_num([dictionary[key] for dictionary in rho_p], 0) for
            key in rho_p[0]
        }
        _return = np.sqrt(np.sum([rho_p[i] for i in detectors], axis=0))
    if return_data_used:
        psd_used = "stored" if not ANALYTIC else list(psd.values())[0].__name__
        data_used = {"psd": psd_used, "approximant": approx, "f_final": f_final}
        return _return, data_used
    return _return


def _wrapper_for_precessing_snr(args):
    """Wrapper function for _precessing_snr for a pool of workers

    Parameters
    ----------
    args: tuple
        All args passed to _precessing_snr
    """
    return _precessing_snr(*args)


def _calculate_b_bar(
    harmonic_dict, psd_dict, low_frequency_cutoff=20.,
    high_frequency_cutoff=1024., return_snrs=False
):
    """Calculate the tangent of half the opening angle as defined in
    Fairhurst et al. arXiv:1908.05707

    Parameters
    ----------
    harmonic_dict: dict
        dictionary of precessing harmonics. Key is the harmonic number
        (0, 1, 2...) and item is the
        `pycbc.types.frequencyseries.FrequencySeries` object
    psd_dict: dict
        dictionary of psds. Key is the IFO and item is the
        `pycbc.types.frequencyseries.FrequencySeries` object
    low_frequency_cutoff: float, optional
        Low frequency-cutoff to use for integration. Default 20Hz
    high_frequency_cutoff: float, optional
        The final frequency to use for integration. Default 1024Hz
    return_snrs: Bool, optional
        if True, return the snrs of each harmonic
    """
    _optimal_snr = lambda harmonic, psd: optimal_snr(
        harmonic, psd, low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff
    )
    rhos = {
        detector: {
            0: _optimal_snr(harmonic_dict[0], psd_dict[detector]),
            1: _optimal_snr(harmonic_dict[1], psd_dict[detector])
        } for detector in psd_dict.keys()
    }
    b_bar = {detector: snr[1] / snr[0] for detector, snr in rhos.items()}
    if return_snrs:
        return b_bar, rhos
    return b_bar


def _precessing_snr(
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, theta_jn,
    beta, psi_J, ra, dec, time, approx, psd, detectors, phi_jl, distance,
    phase, f_plus_j, f_cross_j, f_low, df, f_final, flen, f_ref, debug
):
    """Calculate the square of the precessing SNR for a given detector network

    Parameters
    ----------
    mass_1: float
        Primary mass of the bianry
    mass_2: float
        Secondary mass of the binary
    a_1: float
        The spin magnitude on the larger object
    a_2: float
        The spin magnitude on the secondary object
    tilt_1: float
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: float
        The angle between the total orbital angular momentum and the secondary
        spin
    phi_12: float
        The angle between the primary spin and the secondary spin
    theta_jn: float
        The angle between the total orbital angular momentum and the line of
        sight
    beta: float
        The angle between the total angular momentum and the total orbital
        angular momentum
    psi_J: float
        The polarization angle as defined with respect to the total angular
        momentum
    ra: float
        The right ascension of the source
    dec: float
        The declination of the source
    time: float
        The merger time of the binary
    flow: float
        Low frequency-cutoff to use for integration
    df: float
        The difference between consecutive frequency samples
    f_final: float
        The final frequency to use for integration
    flen: int
        Length of the frequency series in samples. Default is None
    f_ref: float, optional
        Reference frequency where the spins are defined. Default is f_low
    approx: str
        Name of the approximant to use when calculating the harmonic
        decomposition
    psd: dict
        Dictionary of PSDs for each detector
    detector: list
        List of detectors to analyse
    phi_jl: float
        the precession phase
    """
    rho_p_dict = {detector: 0 for detector in detectors}
    b_bar_dict = {detector: 0 for detector in detectors}
    overlap_dict = {detector: 0 for detector in detectors}
    snr_dict = {detector: 0 for detector in detectors}
    harmonics = _calculate_precessing_harmonics(
        mass_1, mass_2, a_1, a_2, tilt_1,
        tilt_2, phi_12, beta, distance, approx=approx, f_final=f_final,
        flen=flen, f_ref=f_ref, f_low=f_low, df=df
    )
    for detector in detectors:
        rho_0 = optimal_snr(
            harmonics[0], psd[detector], low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_final
        )
        rho_1 = optimal_snr(
            harmonics[1], psd[detector], low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_final
        )
        b_bar = rho_1 / rho_0
        pr = _prec_ratio_plus_cross(
            theta_jn, phi_jl, f_plus_j[detector], f_cross_j[detector], b_bar
        )
        overlap = compute_the_overlap(
            harmonics[0], harmonics[1], psd[detector],
            low_frequency_cutoff=f_low, high_frequency_cutoff=f_final,
            normalized=True
        )
        overlap_squared = np.abs(overlap)**2
        prec_squared = np.abs(pr)**2
        real_prec_overlap = 2 * (pr * overlap).real

        h_2harm = _make_waveform_from_precessing_harmonics(
            harmonics, theta_jn, phi_jl, phase, f_plus_j[detector],
            f_cross_j[detector]
        )
        snr = optimal_snr(
            h_2harm, psd[detector], low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_final
        )
        normalization = snr**2 / (1 + prec_squared + real_prec_overlap)
        rho_0 = (
            1 + np.abs(pr * overlap)**2 + real_prec_overlap
        ) * normalization
        rho_0_perp = (prec_squared * (1 - overlap_squared)) * normalization
        rho_1 = (
            overlap_squared + real_prec_overlap + prec_squared
        ) * normalization
        rho_1_perp = (1 - overlap_squared) * normalization
        _rho_p = np.min([rho_0_perp, rho_1_perp])
        if _rho_p > snr**2 / 2.:
            _rho_p = snr**2 - _rho_p
        rho_p_dict[detector] = _rho_p
        b_bar_dict[detector] = b_bar
        overlap_dict[detector] = np.sqrt(overlap_squared)
        snr_dict[detector] = snr**2
    if debug:
        return rho_p_dict, b_bar_dict, overlap_dict, snr_dict
    return rho_p_dict
