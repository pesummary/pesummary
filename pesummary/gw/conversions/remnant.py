# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

from pesummary.utils.utils import logger, iterator
from pesummary.utils.decorators import array_input
from .spins import chi_p

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    import lalsimulation
    from lalsimulation import (
        FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL,
        FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION,
        FLAG_SEOBNRv4P_ZFRAME_L
    )
    from lal import MSUN_SI
except ImportError:
    pass

DEFAULT_SEOBFLAGS = {
    "SEOBNRv4P_SpinAlignedEOBversion": 4,
    "SEOBNRv4P_SymmetrizehPlminusm": 1,
    "SEOBNRv4P_HamiltonianDerivative": FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL,
    "SEOBNRv4P_euler_extension": FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION,
    "SEOBNRv4P_Zframe": FLAG_SEOBNRv4P_ZFRAME_L,
    "SEOBNRv4P_debug": 0
}


@array_input()
def final_mass_of_merger_from_NSBH(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the final mass resulting from an NSBH merger using NSBH
    waveform models given samples for mass_1, mass_2, spin_1z and lambda_2.
    mass_1 and mass_2 should be in solar mass units.
    """
    from .tidal import _check_NSBH_approximant
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[4]


@array_input()
def final_spin_of_merger_from_NSBH(
    mass_1, mass_2, spin_1z, lambda_2, approximant="IMRPhenomNSBH"
):
    """Calculate the final spin resulting from an NSBH merger using NSBH
    waveform models given samples for mass_1, mass_2, spin_1z and lambda_2.
    mass_1 and mass_2 should be in solar mass units.
    """
    from .tidal import _check_NSBH_approximant
    return _check_NSBH_approximant(
        approximant, mass_1, mass_2, spin_1z, lambda_2
    )[5]


@array_input()
def _final_from_initial_NSBH(*args, **kwargs):
    """Calculate the final mass and final spin given the initial parameters
    of the binary using the approximant directly
    """
    return [
        final_mass_of_merger_from_NSBH(*args, **kwargs),
        final_spin_of_merger_from_NSBH(*args, **kwargs)
    ]


def _wrapper_return_final_mass_and_final_spin_from_waveform(args):
    """Wrapper function to calculate the remnant properties for a given waveform
    for a pool of workers

    Parameters
    ----------
    args: np.ndarray
        2 dimensional array giving arguments to pass to
        _return_final_mass_and_final_spin_from_waveform. The first argument
        in each sublist is the keyword and the second argument in each sublist
        is the item you wish to pass
    """
    kwargs = {arg[0]: arg[1] for arg in args}
    return _return_final_mass_and_final_spin_from_waveform(**kwargs)


def _return_final_mass_and_final_spin_from_waveform(
    mass_function=None, spin_function=None, mass_function_args=[],
    spin_function_args=[], mass_function_return_function=None,
    mass_function_return_index=None, spin_function_return_function=None,
    spin_function_return_index=None, mass_1_index=0, mass_2_index=1,
    nsamples=0, approximant=None, default_SEOBNRv4P_kwargs=False
):
    """Return the final mass and final spin given functions to use

    Parameters
    ----------
    mass_function: func
        function you wish to use to calculate the final mass
    spin_function: func
        function you wish to use to calculate the final spin
    mass_function_args: list
        list of arguments you wish to pass to mass_function
    spin_function_args: list
        list of arguments you wish to pass to spin_function
    mass_function_return_function: str, optional
        function used to extract the final mass from the quantity returned from
        mass_function. For example, if mass_function returns a list and the
        final_mass is a property of the 3 arg of this list,
        mass_function_return_function='[3].final_mass'
    mass_function_return_index: str, optional
        if mass_function returns a list of parameters,
        mass_function_return_index indicates the index of `final_mass` in the
        list
    spin_function_return_function: str, optional
        function used to extract the final spin from the quantity returned from
        spin_function. For example, if spin_function returns a list and the
        final_spin is a property of the 3 arg of this list,
        spin_function_return_function='[3].final_spin'
    spin_function_return_index: str, optional
        if spin_function returns a list of parameters,
        spin_function_return_index indicates the index of `final_spin` in the
        list
    mass_1_index: int, optional
        the index of mass_1 in mass_function_args. Default is 0
    mass_2_index: int, optional
        the index of mass_2 in mass_function_args. Default is 1
    nsamples: int, optional
        the total number of samples
    approximant: str, optional
        the approximant used
    default_SEOBNRv4P_kwargs: Bool, optional
        if True, use the default SEOBNRv4P flags
    """
    if default_SEOBNRv4P_kwargs:
        mode_array, seob_flags = _setup_SEOBNRv4P_args()
        mass_function_args += [mode_array, seob_flags]
        spin_function_args += [mode_array, seob_flags]
    fm = mass_function(*mass_function_args)
    if mass_function_return_function is not None:
        fm = eval("fm{}".format(mass_function_return_function))
    elif mass_function_return_index is not None:
        fm = fm[mass_function_return_index]
    fs = spin_function(*spin_function_args)
    if spin_function_return_function is not None:
        fs = eval("fs{}".format(spin_function_return_function))
    elif spin_function_return_index is not None:
        fs = fs[spin_function_return_index]
    final_mass = fm * (
        mass_function_args[mass_1_index] + mass_function_args[mass_2_index]
    ) / MSUN_SI
    final_spin = fs
    return final_mass, final_spin


def _setup_SEOBNRv4P_args(mode=[2, 2], seob_flags=DEFAULT_SEOBFLAGS):
    """Setup the SEOBNRv4P[HM] kwargs
    """
    from lalsimulation import (
        SimInspiralCreateModeArray, SimInspiralModeArrayActivateMode
    )
    from lal import DictInsertINT4Value, CreateDict

    mode_array = SimInspiralCreateModeArray()
    SimInspiralModeArrayActivateMode(mode_array, mode[0], mode[1])
    _seob_flags = CreateDict()
    for key, item in seob_flags.items():
        DictInsertINT4Value(_seob_flags, key, item)
    return mode_array, _seob_flags


@array_input()
def _final_from_initial_BBH(
    mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
    approximant="SEOBNRv4", iota=None, luminosity_distance=None, f_ref=None,
    phi_ref=None, mode=[2, 2], delta_t=1. / 4096, seob_flags=DEFAULT_SEOBFLAGS,
    return_fits_used=False, multi_process=None
):
    """Calculate the final mass and final spin given the initial parameters
    of the binary using the approximant directly

    Parameters
    ----------
    mass_1: float/np.ndarray
        primary mass of the binary
    mass_2: float/np.ndarray
        secondary mass of the binary
    spin_1x: float/np.ndarray
        x component of the primary spin
    spin_1y: float/np.ndarray
        y component of the primary spin
    spin_1z: float/np.ndarray
        z component of the primary spin
    spin_2x: float/np.ndarray
        x component of the secondary spin
    spin_2y: float/np.ndarray
        y component of the secondary spin
    spin_2z: float/np.ndarray
        z component of the seconday spin
    approximant: str
        name of the approximant you wish to use for the remnant fits
    iota: float/np.ndarray, optional
        the angle between the total orbital angular momentum and the line of
        sight of the source. Used when calculating the remnant fits for
        SEOBNRv4PHM. Since we only need the EOB dynamics here it does not matter
        what we pass
    luminosity_distance: float/np.ndarray, optional
        the luminosity distance of the source. Used when calculating the
        remnant fits for SEOBNRv4PHM. Since we only need the EOB dynamics here
        it does not matter what we pass.
    f_ref: float/np.ndarray, optional
        the reference frequency at which the spins are defined
    phi_ref: float/np.ndarray, optional
        the coalescence phase of the binary
    mode: list, optional
        specific mode to use when calculating the remnant fits for SEOBNRv4PHM.
        Since we only need the EOB dynamics here it does not matter what we
        pass.
    delta_t: float, optional
        the sampling rate used in the analysis, Used when calculating the
        remnant fits for SEOBNRv4PHM
    seob_flags: dict, optional
        dictionary containing the SEOB flags. Used when calculating the remnant
        fits for SEOBNRv4PHM
    return_fits_used: Bool, optional
        if True, return the approximant that was used.
    multi_process: int, optional
        the number of cores to use when calculating the remnant fits
    """
    from lalsimulation import (
        SimIMREOBFinalMassSpin, SimIMREOBFinalMassSpinPrec,
        SimInspiralGetSpinSupportFromApproximant,
        SimIMRSpinPrecEOBWaveformAll, SimPhenomUtilsIMRPhenomDFinalMass,
        SimPhenomUtilsPhenomPv2FinalSpin
    )
    import multiprocessing

    def convert_args_for_multi_processing(kwargs):
        args = []
        for n in range(kwargs["nsamples"]):
            _args = []
            for key, item in kwargs.items():
                if key == "mass_function_args" or key == "spin_function_args":
                    _args.append([key, [arg[n] for arg in item]])
                else:
                    _args.append([key, item])
            args.append(_args)
        return args

    try:
        approx = getattr(lalsimulation, approximant)
    except AttributeError:
        raise ValueError(
            "The waveform '{}' is not supported by lalsimulation"
        )

    m1 = mass_1 * MSUN_SI
    m2 = mass_2 * MSUN_SI
    kwargs = {"nsamples": len(mass_1), "approximant": approximant}
    if approximant.lower() in ["seobnrv4p", "seobnrv4phm"]:
        if any(i is None for i in [iota, luminosity_distance, f_ref, phi_ref]):
            raise ValueError(
                "The approximant '{}' requires samples for iota, f_ref, "
                "phi_ref and luminosity_distance. Please pass these "
                "samples.".format(approximant)
            )
        if len(delta_t) == 1:
            delta_t = [delta_t[0]] * len(mass_1)
        elif len(delta_t) != len(mass_1):
            raise ValueError(
                "Please provide either a single 'delta_t' that is is used for "
                "all samples, or a single 'delta_t' for each sample"
            )
        mode_array, _seob_flags = _setup_SEOBNRv4P_args(
            mode=mode, seob_flags=seob_flags
        )
        args = np.array([
            phi_ref, delta_t, m1, m2, f_ref, luminosity_distance, iota,
            spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
            [mode_array] * len(mass_1), [_seob_flags] * len(mass_1)
        ])
        kwargs.update(
            {
                "mass_function": SimIMRSpinPrecEOBWaveformAll,
                "spin_function": SimIMRSpinPrecEOBWaveformAll,
                "mass_function_args": args,
                "spin_function_args": args,
                "mass_function_return_function": "[21].data[6]",
                "spin_function_return_function": "[21].data[7]",
                "mass_1_index": 2,
                "mass_2_index": 3,
            }
        )
    elif approximant.lower() in ["seobnrv4"]:
        spin1 = np.array([spin_1x, spin_1y, spin_1z]).T
        spin2 = np.array([spin_2x, spin_2y, spin_2z]).T
        app = np.array([approx] * len(mass_1))
        kwargs.update(
            {
                "mass_function": SimIMREOBFinalMassSpin,
                "spin_function": SimIMREOBFinalMassSpin,
                "mass_function_args": [m1, m2, spin1, spin2, app],
                "spin_function_args": [m1, m2, spin1, spin2, app],
                "mass_function_return_index": 1,
                "spin_function_return_index": 2
            }
        )
    elif "phenompv3" in approximant.lower():
        kwargs.update(
            {
                "mass_function": SimPhenomUtilsIMRPhenomDFinalMass,
                "spin_function": SimPhenomUtilsPhenomPv2FinalSpin,
                "mass_function_args": [m1, m2, spin_1z, spin_2z],
                "spin_function_args": [m1, m2, spin_1z, spin_2z]
            }
        )
        if SimInspiralGetSpinSupportFromApproximant(approx) > 2:
            # matches the waveform's internal usage as corrected in
            # https://git.ligo.org/lscsoft/lalsuite/-/merge_requests/1270
            _chi_p = chi_p(mass_1, mass_2, spin_1x, spin_1y, spin_2x, spin_2y)
            kwargs["spin_function_args"].append(_chi_p)
        else:
            kwargs["spin_function_args"].append(np.zeros_like(mass_1))
    else:
        raise ValueError(
            "The waveform '{}' is not support by this function.".format(
                approximant
            )
        )

    args = convert_args_for_multi_processing(kwargs)
    if multi_process is not None and multi_process[0] != 1:
        _multi_process = multi_process[0]
        if approximant.lower() in ["seobnrv4p", "seobnrv4phm"]:
            logger.warning(
                "Ignoring passed 'mode' and 'seob_flags' options. Defaults "
                "must be used with multiprocessing. If you wish to use custom "
                "options, please set `multi_process=None`"
            )
            _kwargs = kwargs.copy()
            _kwargs["mass_function_args"] = kwargs["mass_function_args"][:-2]
            _kwargs["spin_function_args"] = kwargs["spin_function_args"][:-2]
            _kwargs["default_SEOBNRv4P_kwargs"] = True
            args = convert_args_for_multi_processing(_kwargs)
        with multiprocessing.Pool(_multi_process) as pool:
            data = np.array(list(
                iterator(
                    pool.imap(
                        _wrapper_return_final_mass_and_final_spin_from_waveform,
                        args
                    ), tqdm=True, desc="Evaluating {} fit".format(approximant),
                    logger=logger, total=len(mass_1)
                )
            )).T
    else:
        final_mass, final_spin = [], []
        _iterator = iterator(
            range(kwargs["nsamples"]), tqdm=True, total=len(mass_1),
            desc="Evaluating {} fit".format(approximant), logger=logger
        )
        for i in _iterator:
            data = _wrapper_return_final_mass_and_final_spin_from_waveform(
                args[i]
            )
            final_mass.append(data[0])
            final_spin.append(data[1])
        data = [final_mass, final_spin]
    if return_fits_used:
        return data, [approximant]
    return data


def final_remnant_properties_from_NRSurrogate(
    *args, f_low=20., f_ref=20., model="NRSur7dq4Remnant", return_fits_used=False,
    properties=["final_mass", "final_spin", "final_kick"], approximant="SEOBNRv4PHM"
):
    """Return the properties of the final remnant resulting from a BBH merger using
    NRSurrogate fits

    Parameters
    ---------
    f_low: float/np.ndarray
        The low frequency cut-off used in the analysis. Default is 20Hz
    f_ref: float/np.ndarray
        The reference frequency used in the analysis. Default is 20Hz
    model: str, optional
        The name of the NRSurrogate model you wish to use
    return_fits_used: Bool, optional
        if True, return the approximant that was used.
    properties: list, optional
        The list of properties you wish to calculate
    approximant: str, optional
        The approximant that was used to generate the posterior samples
    """
    from .nrutils import NRSur_fit

    fit = NRSur_fit(
        *args, f_low=f_low, f_ref=f_ref, model=model, fits=properties,
        approximant=approximant
    )
    if return_fits_used:
        return fit, [model]
    return fit


def final_mass_of_merger_from_NR(
    *args, NRfit="average", final_spin=None, return_fits_used=False
):
    """Return the final mass resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
    final_spin: float/np.ndarray, optional
        precomputed final spin of the remnant.
    return_fits_used: Bool, optional
        if True, return the fits that were used. Only used when NRfit='average'
    """
    from pesummary.gw.conversions import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_final_mass_average")
    elif "panetal" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_mass_non_spinning_Panetal"
        )
    else:
        func = getattr(
            nrutils, "bbh_final_mass_non_precessing_{}".format(NRfit)
        )
    if "healy" in NRfit.lower():
        return func(*args, final_spin=final_spin)
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)


def final_mass_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final mass resulting from a BBH merger using NRSurrogate
    fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_mass"],
        return_fits_used=return_fits_used,
        approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_mass"], data[1]
    return data["final_mass"]


def final_mass_of_merger_from_waveform(*args, NSBH=False, **kwargs):
    """Return the final mass resulting from a BBH/NSBH merger using a given
    approximant

    Parameters
    ----------
    NSBH: Bool, optional
        if True, use NSBH waveform fits. Default False
    """
    if NSBH or "nsbh" in kwargs.get("approximant", "").lower():
        return _final_from_initial_NSBH(*args, **kwargs)[1]
    return _final_from_initial_BBH(*args, **kwargs)[0]


def final_spin_of_merger_from_NR(
    *args, NRfit="average", return_fits_used=False
):
    """Return the final spin resulting from a BBH merger using NR fits

    Parameters
    ----------
    NRfit: str
        Name of the fit you wish to use. If you wish to use a precessing fit
        please use the syntax 'precessing_{}'.format(fit_name). If you wish
        to have an average NR fit, then pass 'average'
    return_fits_used: Bool, optional
        if True, return the fits that were used. Only used when NRfit='average'
    """
    from pesummary.gw.conversions import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_final_spin_average_precessing")
    elif "pan" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_spin_non_spinning_Panetal"
        )
    elif "precessing" in NRfit.lower():
        func = getattr(
            nrutils, "bbh_final_spin_precessing_{}".format(
                NRfit.split("precessing_")[1]
            )
        )
    else:
        func = getattr(
            nrutils, "bbh_final_spin_non_precessing_{}".format(NRfit)
        )
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)


def final_spin_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final spin resulting from a BBH merger using NRSurrogate
    fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_spin"],
        return_fits_used=return_fits_used, approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_spin"], data[1]
    return data["final_spin"]


def final_spin_of_merger_from_waveform(*args, NSBH=False, **kwargs):
    """Return the final spin resulting from a BBH/NSBH merger using a given
    approximant.

    Parameters
    ----------
    NSBH: Bool, optional
        if True, use NSBH waveform fits. Default False
    """
    if NSBH or "nsbh" in kwargs.get("approximant", "").lower():
        return _final_from_initial_NSBH(*args, **kwargs)[1]
    return _final_from_initial_BBH(*args, **kwargs)[1]


def final_kick_of_merger_from_NRSurrogate(
    *args, model="NRSur7dq4Remnant", return_fits_used=False, approximant="SEOBNRv4PHM"
):
    """Return the final kick of the remnant resulting from a BBH merger
    using NRSurrogate fits
    """
    data = final_remnant_properties_from_NRSurrogate(
        *args, model=model, properties=["final_kick"],
        return_fits_used=return_fits_used, approximant=approximant
    )
    if return_fits_used:
        return data[0]["final_kick"], data[1]
    return data["final_kick"]


def final_mass_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    final_spin=None, return_fits_used=False, model="NRSur7dq4Remnant"
):
    """Return the final mass resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    spin_1z: float/np.ndarray
        float/array of primary spin aligned with the orbital angular momentum
    spin_2z: float/np.ndarray
        float/array of secondary spin aligned with the orbital angular momentum
    method: str
        The method you wish to use to calculate the final mass of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if method.lower() == "nr":
        mass_func = final_mass_of_merger_from_NR
        kwargs = {
            "NRfit": NRfit, "final_spin": final_spin,
            "return_fits_used": return_fits_used
        }
    elif "nrsur" in method.lower():
        mass_func = final_mass_of_merger_from_NRSurrogate
        kwargs = {
            "approximant": approximant, "return_fits_used": return_fits_used,
            "model": model
        }
    else:
        mass_func = final_mass_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return mass_func(*args, **kwargs)


def final_spin_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    return_fits_used=False, model="NRSur7dq4Remnant"
):
    """Return the final mass resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular
        momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    method: str
        The method you wish to use to calculate the final mass of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if method.lower() == "nr":
        spin_func = final_spin_of_merger_from_NR
        kwargs = {"NRfit": NRfit, "return_fits_used": return_fits_used}
    elif "nrsur" in method.lower():
        spin_func = final_spin_of_merger_from_NRSurrogate
        kwargs = {
            "approximant": approximant, "return_fits_used": return_fits_used,
            "model": model
        }
    else:
        spin_func = final_spin_of_merger_from_waveform
        kwargs = {"approximant": approximant}

    return spin_func(*args, **kwargs)


def final_kick_of_merger(
    *args, method="NR", approximant="SEOBNRv4", NRfit="average",
    return_fits_used: False, model="NRSur7dq4Remnant"
):
    """Return the final kick velocity of the remnant resulting from a BBH merger

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    a_1: float/np.ndarray
        float/array of primary spin magnitudes
    a_2: float/np.ndarray
        float/array of secondary spin magnitudes
    tilt_1: float/np.ndarray
        float/array of primary spin tilt angle from the orbital angular momentum
    tilt_2: float/np.ndarray
        float/array of secondary spin tilt angle from the orbital angular
        momentum
    phi_12: float/np.ndarray
        float/array of samples for the angle between the in-plane spin
        components
    method: str
        The method you wish to use to calculate the final kick of merger. Either
        NR, NRSurrogate or waveform
    approximant: str
        Name of the approximant you wish to use if the chosen method is waveform
        or NRSurrogate
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average' or when method='NRSurrogate'
    model: str, optional
        The NRSurrogate model to use when evaluating the fits
    """
    if "nrsur" not in method.lower():
        raise NotImplementedError(
            "Currently you can only work out the final kick velocity using "
            "NRSurrogate fits."
        )
    velocity_func = final_kick_of_merger_from_NRSurrogate
    kwargs = {
        "approximant": approximant, "return_fits_used": return_fits_used,
        "model": model
    }
    return velocity_func(*args, **kwargs)


def peak_luminosity_of_merger(*args, NRfit="average", return_fits_used=False):
    """Return the peak luminosity of an aligned-spin BBH using NR fits

    Parameters
    ----------
    mass_1: float/np.ndarray
        float/array of masses for the primary object
    mass_2: float/np.ndarray
        float/array of masses for the secondary object
    spin_1z: float/np.ndarray
        float/array of primary spin aligned with the orbital angular momentum
    spin_2z: float/np.ndarray
        float/array of secondary spin aligned with the orbital angular momentum
    NRFit: str
        Name of the NR fit you wish to use if chosen method is NR
    return_fits_used: Bool, optional
        if True, return the NR fits that were used. Only used when
        NRFit='average'
    """
    from pesummary.gw.conversions import nrutils

    if NRfit.lower() == "average":
        func = getattr(nrutils, "bbh_peak_luminosity_average")
    else:
        func = getattr(
            nrutils, "bbh_peak_luminosity_non_precessing_{}".format(NRfit)
        )
    if NRfit.lower() == "average":
        return func(*args, return_fits_used=return_fits_used)
    return func(*args)
