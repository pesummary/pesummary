# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import lalsimulation as lalsim
from lalsimulation import (
    SimInspiralGetSpinFreqFromApproximant, SIM_INSPIRAL_SPINS_CASEBYCASE,
    SIM_INSPIRAL_SPINS_FLOW
)
from pesummary.utils.utils import iterator, logger
from pesummary.utils.exceptions import EvolveSpinError

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_python_module_from_approximant_name = {
    "SEOBNRv5": "lalsimulation.gwsignal.models.pyseobnr_model"
}

def _get_spin_freq_from_approximant(approximant):
    """Determine whether the reference frequency is the starting frequency
    for a given approximant string.

    Parameters
    ----------
    approximant: str
        Name of the approximant you wish to check
    """
    try:
        # default to using LAL code
        approx = getattr(lalsim, approximant)
        return SimInspiralGetSpinFreqFromApproximant(approx)
    except AttributeError:
        from lalsimulation import (
            SIM_INSPIRAL_SPINS_NONPRECESSING, SIM_INSPIRAL_SPINS_F_REF
        )
        # check to see if approximant is in gwsignal
        approx = _gwsignal_generator_from_string(approximant)
        meta = approx.metadata
        if meta["type"] == "aligned_spin":
            return SIM_INSPIRAL_SPINS_NONPRECESSING
        elif meta["type"] == "precessing_spin":
            if meta["f_ref_spin"]:
                return SIM_INSPIRAL_SPINS_F_REF
            return SIM_INSPIRAL_SPINS_FLOW
        raise EvolveSpinError(
            "Unable to evolve spins as '{}' does not have a set frequency "
            "at which the spins are defined".format(approximant)
        )


def _get_start_freq_from_approximant(approximant, f_low, f_ref):
    """Determine the starting frequency to use when evolving the spins for
    a given approximant string.

    Parameters
    ----------
    approximant: str
        Name of the approximant you wish to check
    f_low: float
        Low frequency used when generating the posterior samples
    f_ref: float
        Reference frequency used when generating the posterior samples
    """
    try:
        spinfreq_enum = _get_spin_freq_from_approximant(approximant)
    except ValueError:  # raised when approximant is not in gwsignal
        raise EvolveSpinError(
            "Unable to evolve spins as '{}' is unknown to lalsimulation "
            "and gwsignal".format(approximant)
        )
    if spinfreq_enum == SIM_INSPIRAL_SPINS_CASEBYCASE:
        _msg = (
            "Unable to evolve spins as '{}' does not have a set frequency "
            "at which the spins are defined".format(approximant)
        )
        logger.warning(_msg)
        raise EvolveSpinError(_msg)
    return float(np.where(
        np.array(spinfreq_enum == SIM_INSPIRAL_SPINS_FLOW), f_low, f_ref
    ))


def _check_approximant_from_string(approximant):
    """Check to see if the approximant is known to lalsimulation and/or
    gwsignal

    Parameters
    ----------
    approximant: str
        approximant you wish to check
    """
    if hasattr(lalsim, approximant):
        return True
    else:
        try:
            _ = _gwsignal_generator_from_string(approximant)
        except (ValueError, NameError):
            return False
        return True


def _lal_approximant_from_string(approximant):
    """Return the LAL approximant number given an approximant string

    Parameters
    ----------
    approximant: str
        approximant you wish to convert
    """
    return lalsim.GetApproximantFromString(approximant)


def _gwsignal_generator_from_string(approximant, version="v1", **kwargs):
    """Return the GW signal generator given an approximant string

    Parameters
    ----------
    approximant: str
        approximant you wish to convert
    version: str, optional
        version of gwsignal_get_waveform_generator to use. 'v1' ignores
        all kwargs and simply passes the approximant name to
        gwsignal_get_waveform_generator. 'v2' passes the approximant name
        and python_module to gwsignal_get_waveform_generator. The python_module
        can either be provided to _gwsignal_generator_from_string directly or if
        not provided, PESummary will use the stored default values. Default 'v1'
    kwargs: dict, optional
        optional kwargs to pass to gwsignal_get_waveform_generator
    """
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
    if version == "v1":
        return gwsignal_get_waveform_generator(approximant)
    elif version == "v2":
        fallback = [
            value for key, value in _python_module_from_approximant_name.items()
            if key in approximant
        ]
        if len(fallback):
            fallback = fallback[0]
        else:
            fallback = None
        module = kwargs.get("python_module", fallback)
        if module is None:
            raise ValueError(
                "Please provide a python_module defining the gwsignal "
                "generator as pesummary does not have a default value"
            )
        return gwsignal_get_waveform_generator(
            approximant, python_module=module
        )
    else:
        raise ValueError(
            "Unknown version {}. Please use either v1 or v2".format(version)
        )


def _insert_flags(flags, LAL_parameters=None):
    """Insert waveform specific flags to a LAL dictionary

    Parameters
    ----------
    flags: dict
        dictionary of flags you wish to add to a LAL dictionary
    LAL_parameters: LALDict, optional
        An existing LAL dictionary to add mode array to. If not provided, a new
        LAL dictionary is created. Default None.
    """
    if LAL_parameters is None:
        import lal
        LAL_parameters = lal.CreateDict()
    for key, item in flags.items():
        func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
        if func is not None:
            func(LAL_parameters, int(item))
        else:
            logger.warning(
                "Unable to add flag {} to LAL dictionary as no function "
                "SimInspiralWaveformParamsInsert{} found in "
                "LALSimulation.".format(key, key)
            )
    return LAL_parameters


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
    _zero_spins = np.isclose(spins, 0.)
    if sum(_zero_spins):
        spins = np.array(spins)
        spins[_zero_spins] = 0.
        spins = list(spins)
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
    mode_array=None, pycbc=False, flen=None, flags={}, debug=False,
    **kwargs
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
    debug: bool, optional
        if True, return the model object used to generate the waveform. Only
        used when the waveform approximant is not in LALSimulation
    flags: dict, optional
        waveform specific flags to add to LAL dictionary
    **kwargs: dict, optional
        all kwargs passed to _calculate_hp_hc_fd
    """
    from gwpy.frequencyseries import FrequencySeries

    waveform_args, _samples = _waveform_args(
        samples, f_ref=f_ref, ind=ind, longAscNodes=longAscNodes,
        eccentricity=eccentricity
    )
    if len(flags):
        LAL_parameters = _insert_flags(
            flags, LAL_parameters=LAL_parameters
        )
    if mode_array is not None:
        LAL_parameters = _insert_mode_array(
            mode_array, LAL_parameters=LAL_parameters
        )
    (hp, hc), model = _calculate_hp_hc_fd(
        waveform_args, delta_f, f_low, f_high, f_ref, LAL_parameters, approximant,
        debug=True, **kwargs
    )
    hp = FrequencySeries(hp.data.data, df=hp.deltaF, f0=0.)
    hc = FrequencySeries(hc.data.data, df=hc.deltaF, f0=0.)
    if pycbc:
        hp, hc = hp.to_pycbc(), hc.to_pycbc()
        if flen is not None:
            hp.resize(flen)
            hc.resize(flen)
    if project is None and debug:
        return {"h_plus": hp, "h_cross": hc}, model
    elif project is None:
        return {"h_plus": hp, "h_cross": hc}
    ht = _project_waveform(
        project, hp, hc, _samples["ra"], _samples["dec"], _samples["psi"],
        _samples["geocent_time"]
    )
    if debug:
        return ht, model
    return ht


def _wrapper_for_td_waveform(args):
    """Wrapper function for td_waveform for a pool of workers

    Parameters
    ----------
    args: tuple
        All args passed to td_waveform
    """
    return td_waveform(*args[:-1], **args[-1])


def td_waveform(
    samples, approximant, delta_t, f_low, f_ref=20., project=None, ind=0,
    longAscNodes=0., eccentricity=0., LAL_parameters=None, mode_array=None,
    pycbc=False, level=None, multi_process=1, flags={}, debug=False,
    **kwargs
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
    level: list, optional
        the symmetric confidence interval of the time domain waveform. Level
        must be greater than 0 and less than 1
    multi_process: int, optional
        number of cores to run on when generating waveforms. Only used when
        level is not None,
    flags: dict, optional
        waveform specific flags to add to LAL dictionary. Default {}
    kwargs: dict, optional
        all kwargs passed to _td_waveform
    """
    if len(flags):
        LAL_parameters = _insert_flags(
            flags, LAL_parameters=LAL_parameters
        )
    if mode_array is not None:
        LAL_parameters = _insert_mode_array(
            mode_array, LAL_parameters=LAL_parameters
        )
    if level is not None:
        import multiprocessing
        from pesummary.utils.interpolate import BoundedInterp1d
        td_waveform_list = []
        _key = list(samples.keys())[0]
        N = len(samples[_key])
        with multiprocessing.Pool(multi_process) as pool:
            args = np.array([
                [samples] * N, [approximant] * N, [delta_t] * N, [f_low] * N,
                [f_ref] * N, [project] * N, np.arange(N), [longAscNodes] * N,
                [eccentricity] * N, [LAL_parameters] * N, [mode_array] * N,
                [pycbc] * N, [None] * N, [kwargs] * N
            ], dtype="object").T
            td_waveform_list = list(
                iterator(
                    pool.imap(_wrapper_for_td_waveform, args),
                    tqdm=True, logger=logger, total=N,
                    desc="Generating waveforms"
                )
            )
        td_waveform_array = np.array(td_waveform_list, dtype=object)
        _level = (1 + np.array(level)) / 2
        if project is None:
            mint = np.min(
                [
                    np.min([_.times[0].value for _ in waveform.values()]) for
                    waveform in td_waveform_array
                ]
            )
            maxt = np.max(
                [
                    np.max([_.times[-1].value for _ in waveform.values()]) for
                    waveform in td_waveform_array
                ]
            )
            new_t = np.arange(mint, maxt, delta_t)
            td_waveform_array = {
                polarization: np.array(
                    [
                        BoundedInterp1d(
                            np.array(waveform[polarization].times, dtype=np.float64),
                            waveform[polarization], xlow=mint, xhigh=maxt
                        )(new_t) for waveform in td_waveform_array
                    ]
                ) for polarization in ["h_plus", "h_cross"]
            }
        else:
            mint = np.min([_.times[0].value for _ in td_waveform_array])
            maxt = np.max([_.times[-1].value for _ in td_waveform_array])
            new_t = np.arange(mint, maxt, delta_t)
            td_waveform_array = {
                "h_t": [
                    BoundedInterp1d(
                        np.array(waveform.times, dtype=np.float64), waveform,
                        xlow=mint, xhigh=maxt
                    )(new_t) for waveform in td_waveform_array
                ]
            }

        upper = {
            polarization: np.percentile(
                td_waveform_array[polarization], _level * 100, axis=0
            ) for polarization in td_waveform_array.keys()
        }
        lower = {
            polarization: np.percentile(
                td_waveform_array[polarization], (1 - _level) * 100, axis=0
            ) for polarization in td_waveform_array.keys()
        }
        if len(upper) == 1:
            upper = upper["h_t"]
            lower = lower["h_t"]

    waveform_args, _samples = _waveform_args(
        samples, ind=ind, longAscNodes=longAscNodes, eccentricity=eccentricity,
        f_ref=f_ref
    )
    waveform, model = _td_waveform(
        waveform_args, approximant, delta_t, f_low, f_ref, LAL_parameters,
        _samples, pycbc=pycbc, project=project, debug=True, **kwargs
    )
    if level is not None and debug:
        return waveform, upper, lower, new_t, model
    elif level is not None:
        return waveform, upper, lower, new_t
    elif debug:
        return waveform, model
    return waveform


def _td_waveform(
    waveform_args, approximant, delta_t, f_low, f_ref, LAL_parameters, samples,
    pycbc=False, project=None, debug=False, **kwargs
):
    """Generate a gravitational wave in the time domain

    Parameters
    ----------
    waveform_args: tuple
        args to pass to lalsimulation.SimInspiralChooseTDWaveform
    approximant: str
        lalsimulation approximant number to use when generating a waveform
    delta_t: float
        spacing between time samples
    f_low: float
        frequency to start evaluating the waveform
    f_ref: float, optional
        reference frequency
    LAL_parameters: LALDict
        LAL dictionary containing accessory parameters. Default None
    samples: dict
        dictionary of posterior samples to use when projecting the waveform
        onto a given detector
    pycbc: Bool, optional
        return a the waveform as a pycbc.timeseries.TimeSeries object
    project: str, optional
        name of the detector to project the waveform onto. If None,
        the plus and cross polarizations are returned. Default None
    debug: bool, optional
        if True, return the model object used to generate the waveform. Only
        used when the waveform approximant is not in LALSimulation
    kwargs: dict, optional
        all kwargs passed to _calculate_hp_hc_td
    """
    from gwpy.timeseries import TimeSeries
    from astropy.units import Quantity

    (hp, hc), model = _calculate_hp_hc_td(
        waveform_args, delta_t, f_low, f_ref, LAL_parameters, approximant,
        debug=True, **kwargs
    )
    hp = TimeSeries(hp.data.data, dt=hp.deltaT, t0=hp.epoch)
    hc = TimeSeries(hc.data.data, dt=hc.deltaT, t0=hc.epoch)
    if pycbc:
        hp, hc = hp.to_pycbc(), hc.to_pycbc()
    if project is None and debug:
        return {"h_plus": hp, "h_cross": hc}, model
    elif project is None:
        return {"h_plus": hp, "h_cross": hc}
    ht = _project_waveform(
        project, hp, hc, samples["ra"], samples["dec"], samples["psi"],
        samples["geocent_time"]
    )
    if "{}_time".format(project) not in samples.keys():
        from pesummary.gw.conversions import time_in_each_ifo
        try:
            _detector_time = time_in_each_ifo(
                project, samples["ra"], samples["dec"], samples["geocent_time"]
            )
        except Exception:
            logger.warning(
                "Unable to calculate samples for '{}_time' using the provided "
                "posterior samples. Unable to shift merger to merger time in "
                "the detector".format(project)
            )
            if debug:
                return ht, model
            return ht
    else:
        _detector_time = samples["{}_time".format(project)]
    ht.times = (
        Quantity(ht.times, unit="s") + Quantity(_detector_time, unit="s")
    )
    if debug:
        return ht, model
    return ht


def _calculate_hp_hc_td(
    waveform_args, delta_t, f_low, f_ref, LAL_parameters, approximant,
    debug=False, **kwargs
):
    """Calculate the plus and cross polarizations in the time domain.
    If the approximant is in LALSimulation, the
    SimInspiralChooseTDWaveform function is used. If the approximant is
    not in LALSimulation, gwsignal is used

    Parameters
    ----------
    waveform_args: tuple
        args to pass to lalsimulation.SimInspiralChooseTDWaveform
    delta_t: float
        spacing between time samples
    f_low: float
        frequency to start evaluating the waveform
    f_ref: float, optional
        reference frequency
    LAL_parameters: LALDict
        LAL dictionary containing accessory parameters. Default None
    approximant: str
        lalsimulation approximant number to use when generating a waveform
    debug: bool, optional
        if True, return the model object used to generate the waveform. Only
        used when the waveform approximant is not in LALSimulation
    **kwargs: dict, optional
        all kwargs passed to _setup_gwsignal
    """
    if hasattr(lalsim, approximant):
        approx = _lal_approximant_from_string(approximant)
        if lalsim.SimInspiralImplementedFDApproximants(approx):
            # if the waveform is defined in the frequency domain, use
            # SimInspiralTD as it appropiately conditions the waveform so it
            # is correct at f_low in the time domain
            _func = lalsim.SimInspiralTD
        else:
            # if the waveform is defined in the time domain, use
            # SimInspiralChooseTDWaveform as it doesnt taper the start of the
            # waveform
            _func = lalsim.SimInspiralChooseTDWaveform
        result = _func(
            *waveform_args, delta_t, f_low, f_ref, LAL_parameters, approx
        )
        if debug:
            return result, None
        return result
    wfm_gen = _setup_gwsignal(
        waveform_args, f_low, f_ref, approximant, delta_t=delta_t, **kwargs
    )
    hpc = wfm_gen.generate_td_polarizations()
    if debug:
        return hpc, wfm_gen
    return hpc


def _calculate_hp_hc_fd(
    waveform_args, delta_f, f_low, f_high, f_ref, LAL_parameters, approximant,
    debug=False, **kwargs
):
    """Calculate the plus and cross polarizations in the frequency domain.
    If the approximant is in LALSimulation, the
    SimInspiralChooseFDWaveform function is used. If the approximant is
    not in LALSimulation, gwsignal is used

    Parameters
    ----------
    waveform_args: tuple
        args to pass to lalsimulation.SimInspiralChooseFDWaveform
    delta_f: float
        spacing between frequency samples
    f_low: float
        frequency to start evaluating the waveform
    f_high: float, optional
        frequency to stop evaluating the waveform
    f_ref: float, optional
        reference frequency
    LAL_parameters: LALDict
        LAL dictionary containing accessory parameters. Default None
    approximant: str
        lalsimulation approximant number to use when generating a waveform
    debug: bool, optional
        if True, return the model object used to generate the waveform. Only
        used when the waveform approximant is not in LALSimulation
    """
    if hasattr(lalsim, approximant):
        approx = _lal_approximant_from_string(approximant)
        for func in [lalsim.SimInspiralChooseFDWaveform, lalsim.SimInspiralFD]:
            try:
                result = func(
                    *waveform_args, delta_f, f_low, f_high, f_ref,
                    LAL_parameters, approx
                )
                if debug:
                    return result, None
                return result
            except Exception:
                continue
            break
    wfm_gen = _setup_gwsignal(
        waveform_args, f_low, f_ref, approximant, delta_f=delta_f,
        f_high=f_high, **kwargs
    )
    hpc = wfm_gen.generate_fd_polarizations()
    if debug:
        return hpc, wfm_gen
    return hpc


def _setup_gwsignal(
    waveform_args, f_low, f_ref, approximant, delta_t=None, delta_f=None,
    f_high=None, **kwargs
):
    """Setup a waveform generator with gwsignal

    Parameters
    ----------
    waveform_args: tuple
        args to pass to lalsimulation.SimInspiralChoose*DWaveform
    f_low: float
        frequency to start evaluating the waveform
    f_ref: float, optional
        reference frequency
    approximant: str
        lalsimulation approximant number to use when generating a waveform
    delta_t: float, optional
        spacing between frequency samples
    delta_f: float, optional
        spacing between frequency samples
    f_high: float, optional
        frequency to stop evaluating the waveform
    kwargs: dict, optional
        all kwargs passed to _gwsignal_generator_from_string
    """
    from lalsimulation.gwsignal.core.parameter_conventions import (
        common_units_dictionary
    )
    from astropy import units
    names = [
        "mass1", "mass2", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y",
        "spin2z", "distance", "inclination", "phi_ref", "longAscNodes",
        "eccentricity", "meanPerAno"
    ]
    params_dict = {name: waveform_args[ii] for ii, name in enumerate(names)}
    if delta_t is not None:
        params_dict["deltaT"] = delta_t
    if delta_f is not None:
        params_dict["deltaF"] = delta_f
    params_dict["f22_start"] = f_low
    params_dict["f22_ref"] = f_ref
    if f_high is None:
        params_dict["f_max"] = 0.5 / params_dict["deltaT"]
    else:
        params_dict["f_max"] = f_high
    wf_gen = _gwsignal_generator_from_string(approximant, **kwargs)
    for key, item in params_dict.items():
        params_dict[key] = item * common_units_dictionary.get(key, 1)
    params_dict["mass1"] *= units.kg
    params_dict["mass2"] *= units.kg
    params_dict["distance"] *= units.m
    wfm_gen = wf_gen._generate_waveform_class(**params_dict)
    return wfm_gen
