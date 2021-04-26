# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import lalsimulation as lalsim

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _lal_approximant_from_string(approximant):
    """Return the LAL approximant number given an approximant string

    Parameters
    ----------
    approximant: str
        approximant you wish to convert
    """
    return lalsim.GetApproximantFromString(approximant)


def _insert_mode_array(modes, LAL_parameters=None):
    """Add a mode array to a LAL dictionary

    Parameters
    ----------
    modes: 2d list
        2d list of modes you wish to add to a LAL dictionary. Must be of the
        form [[l1, m1], [l2, m2]]
    LAL_parameters: LALDict, optional
        An existing LAL dictionary to add mode array to. If not provided, a new
        LAL dictionary is created. Default None.
    """
    if LAL_parameters is None:
        import lal
        LAL_parameters = lal.CreateDict()
    _mode_array = lalsim.SimInspiralCreateModeArray()
    for l, m in modes:
        lalsim.SimInspiralModeArrayActivateMode(_mode_array, l, m)
    lalsim.SimInspiralWaveformParamsInsertModeArray(LAL_parameters, _mode_array)
    return LAL_parameters


def _waveform_args(samples, f_ref=20., ind=0, longAscNodes=0., eccentricity=0.):
    """Arguments to be passed to waveform generation

    Parameters
    ----------
    f_ref: float, optional
        reference frequency to use when converting spherical spins to
        cartesian spins
    ind: int, optional
        index for the sample you wish to plot
    longAscNodes: float, optional
        longitude of ascending nodes, degenerate with the polarization
        angle. Default 0.
    eccentricity: float, optional
        eccentricity at reference frequency. Default 0.
    """
    from lal import MSUN_SI, PC_SI

    key = list(samples.keys())[0]
    if isinstance(samples[key], (list, np.ndarray)):
        _samples = {key: value[ind] for key, value in samples.items()}
    else:
        _samples = samples.copy()
    required = [
        "mass_1", "mass_2", "luminosity_distance"
    ]
    if not all(param in _samples.keys() for param in required):
        raise ValueError(
            "Unable to generate a waveform. Please add samples for "
            + ", ".join(required)
        )
    waveform_args = [
        _samples["mass_1"] * MSUN_SI, _samples["mass_2"] * MSUN_SI
    ]
    spin_angles = [
        "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
        "phase"
    ]
    spin_angles_condition = all(
        spin in _samples.keys() for spin in spin_angles
    )
    cartesian_spins = [
        "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"
    ]
    cartesian_spins_condition = any(
        spin in _samples.keys() for spin in cartesian_spins
    )
    if spin_angles_condition and not cartesian_spins_condition:
        from pesummary.gw.conversions import component_spins
        data = component_spins(
            _samples["theta_jn"], _samples["phi_jl"], _samples["tilt_1"],
            _samples["tilt_2"], _samples["phi_12"], _samples["a_1"],
            _samples["a_2"], _samples["mass_1"], _samples["mass_2"],
            f_ref, _samples["phase"]
        )
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = data.T
        spins = [spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z]
    else:
        iota = _samples["iota"]
        spins = [
            _samples[param] if param in _samples.keys() else 0. for param in
            ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
        ]
    waveform_args += spins
    phase = _samples["phase"] if "phase" in _samples.keys() else 0.
    waveform_args += [
        _samples["luminosity_distance"] * PC_SI * 10**6, iota, phase
    ]
    waveform_args += [longAscNodes, eccentricity, 0.]
    return waveform_args, _samples


def antenna_response(samples, ifo):
    """
    """
    import importlib

    mod = importlib.import_module("pesummary.gw.plots.plot")
    func = getattr(mod, "__antenna_response")
    antenna = func(
        ifo, samples["ra"], samples["dec"], samples["psi"],
        samples["geocent_time"]
    )
    return antenna


def _project_waveform(ifo, hp, hc, ra, dec, psi, time):
    """Project a waveform onto a given detector

    Parameters
    ----------
    ifo: str
        name of the detector you wish to project the waveform onto
    hp: np.ndarray
        plus gravitational wave polarization
    hc: np.ndarray
        cross gravitational wave polarization
    ra: float
        right ascension to be passed to antenna response function
    dec: float
        declination to be passed to antenna response function
    psi: float
        polarization to be passed to antenna response function
    time: float
        time to be passed to antenna response function
    """
    samples = {
        "ra": ra, "dec": dec, "psi": psi, "geocent_time": time
    }
    antenna = antenna_response(samples, ifo)
    ht = hp * antenna[0] + hc * antenna[1]
    return ht


def fd_waveform(
    samples, approximant, delta_f, f_low, f_high, f_ref=20., project=None,
    ind=0, longAscNodes=0., eccentricity=0., LAL_parameters=None,
    mode_array=None, pycbc=False, flen=None
):
    """Generate a gravitational wave in the frequency domain

    Parameters
    ----------
    approximant: str
        name of the approximant to use when generating the waveform
    delta_f: float
        spacing between frequency samples
    f_low: float
        frequency to start evaluating the waveform
    f_high: float
        frequency to stop evaluating the waveform
    f_ref: float, optional
        reference frequency
    project: str, optional
        name of the detector to project the waveform onto. If None,
        the plus and cross polarizations are returned. Default None
    ind: int, optional
        index for the sample you wish to plot
    longAscNodes: float, optional
        longitude of ascending nodes, degenerate with the polarization
        angle. Default 0.
    eccentricity: float, optional
        eccentricity at reference frequency. Default 0.
    LAL_parameters: LALDict, optional
        LAL dictionary containing accessory parameters. Default None
    mode_array: 2d list
        2d list of modes you wish to include in waveform. Must be of the form
        [[l1, m1], [l2, m2]]
    pycbc: Bool, optional
        return a the waveform as a pycbc.frequencyseries.FrequencySeries
        object
    flen: int
        Length of the frequency series in samples. Default is None. Only used
        when pycbc=True
    """
    from gwpy.frequencyseries import FrequencySeries

    waveform_args, _samples = _waveform_args(
        samples, f_ref=f_ref, ind=ind, longAscNodes=longAscNodes,
        eccentricity=eccentricity
    )
    approx = _lal_approximant_from_string(approximant)
    if mode_array is not None:
        LAL_parameters = _insert_mode_array(
            mode_array, LAL_parameters=LAL_parameters
        )
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        *waveform_args, delta_f, f_low, f_high, f_ref, LAL_parameters, approx
    )
    hp = FrequencySeries(hp.data.data, df=hp.deltaF, f0=0.)
    hc = FrequencySeries(hc.data.data, df=hc.deltaF, f0=0.)
    if pycbc:
        hp, hc = hp.to_pycbc(), hc.to_pycbc()
        if flen is not None:
            hp.resize(flen)
            hc.resize(flen)
    if project is None:
        return {"h_plus": hp, "h_cross": hc}
    ht = _project_waveform(
        project, hp, hc, _samples["ra"], _samples["dec"], _samples["psi"],
        _samples["geocent_time"]
    )
    return ht


def td_waveform(
    samples, approximant, delta_t, f_low, f_ref=20., project=None, ind=0,
    longAscNodes=0., eccentricity=0., LAL_parameters=None, mode_array=None,
    pycbc=False
):
    """Generate a gravitational wave in the time domain

    Parameters
    ----------
    approximant: str
        name of the approximant to use when generating the waveform
    delta_t: float
        spacing between frequency samples
    f_low: float
        frequency to start evaluating the waveform
    f_ref: float, optional
        reference frequency
    project: str, optional
        name of the detector to project the waveform onto. If None,
        the plus and cross polarizations are returned. Default None
    ind: int, optional
        index for the sample you wish to plot
    longAscNodes: float, optional
        longitude of ascending nodes, degenerate with the polarization
        angle. Default 0.
    eccentricity: float, optional
        eccentricity at reference frequency. Default 0.
    LAL_parameters: LALDict, optional
        LAL dictionary containing accessory parameters. Default None
    mode_array: 2d list
        2d list of modes you wish to include in waveform. Must be of the form
        [[l1, m1], [l2, m2]]
    pycbc: Bool, optional
        return a the waveform as a pycbc.timeseries.TimeSeries object
    """
    from gwpy.timeseries import TimeSeries
    from astropy.units import Quantity

    waveform_args, _samples = _waveform_args(
        samples, ind=ind, longAscNodes=longAscNodes, eccentricity=eccentricity
    )
    approx = _lal_approximant_from_string(approximant)
    if mode_array is not None:
        LAL_parameters = _insert_mode_array(
            mode_array, LAL_parameters=LAL_parameters
        )
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        *waveform_args, delta_t, f_low, f_ref, LAL_parameters, approx
    )
    hp = TimeSeries(hp.data.data, dt=hp.deltaT, t0=hp.epoch)
    hc = TimeSeries(hc.data.data, dt=hc.deltaT, t0=hc.epoch)
    if pycbc:
        hp, hc = hp.to_pycbc(), hc.to_pycbc()
    if project is None:
        return {"h_plus": hp, "h_cross": hc}
    ht = _project_waveform(
        project, hp, hc, _samples["ra"], _samples["dec"], _samples["psi"],
        _samples["geocent_time"]
    )
    ht.times = (
        Quantity(ht.times, unit="s")
        + Quantity(_samples["{}_time".format(project)], unit="s")
    )
    return ht
