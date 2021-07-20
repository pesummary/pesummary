# Licensed under an MIT style license -- see LICENSE.md

error_msg = (
    "Unable to install '{}'. You will not be able to use some of the inbuilt "
    "functions."
)
import copy
import numpy as np
from pathlib import Path

from pesummary import conf
from pesummary.utils.decorators import set_docstring
from pesummary.utils.exceptions import EvolveSpinError
from pesummary.utils.utils import logger

try:
    import lalsimulation
except ImportError:
    logger.warn(error_msg.format("lalsimulation"))
try:
    import astropy
except ImportError:
    logger.warn(error_msg.format("astropy"))

from .angles import *
from .cosmology import *
from .cosmology import _source_from_detector
from .mass import *
from .remnant import *
from .remnant import _final_from_initial_BBH
from .snr import *
from .snr import _ifo_snr
from .spins import *
from .tidal import *
from .tidal import _check_NSBH_approximant
from .time import *

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_conversion_doc = """
    Class to calculate all possible derived quantities

    Parameters
    ----------
    data: dict, list
        either a dictionary or samples or a list of parameters and a list of
        samples. See the examples below for details
    extra_kwargs: dict, optional
        dictionary of kwargs associated with this set of posterior samples.
    f_low: float, optional
        the low frequency cut-off to use when evolving the spins
    f_ref: float, optional
        the reference frequency when spins are defined
    f_final: float, optional
        the final frequency to use when integrating over frequencies
    approximant: str, optional
        the approximant to use when evolving the spins
    evolve_spins_forwards: float/str, optional
        the final velocity to evolve the spins up to.
    evolve_spins_backwards: str, optional
        method to use when evolving the spins backwards to an infinite separation
    return_kwargs: Bool, optional
        if True, return a modified dictionary of kwargs containing information
        about the conversion
    NRSur_fits: float/str, optional
        the NRSurrogate model to use to calculate the remnant fits. If nothing
        passed, the average NR fits are used instead
    precessing_snr: Bool, optional
        if True, the precessing SNR is calculated from the posterior samples.
    psd: dict, optional
        dictionary containing a psd frequency series for each detector you wish
        to include in calculations
    waveform_fits: Bool, optional
        if True, the approximant is used to calculate the remnant fits. Default
        is False which means that the average NR fits are used
    multi_process: int, optional
        number of cores to use to parallelize the computationally expensive
        conversions
    redshift_method: str, optional
        method you wish to use when calculating the redshift given luminosity
        distance samples. If redshift samples already exist, this method is not
        used. Default is 'approx' meaning that interpolation is used to calculate
        the redshift given N luminosity distance points.
    cosmology: str, optional
        cosmology you wish to use when calculating the redshift given luminosity
        distance samples.
    force_non_evolved: Bool, optional
        force non evolved remnant quantities to be calculated when evolved quantities
        already exist in the input. Default False
    force_BBH_remnant_computation: Bool, optional
        force BBH remnant quantities to be calculated for systems that include
        tidal deformability parameters where BBH fits may not be applicable.
        Default False.
    force_BH_spin_evolution: Bool, optional
        force BH spin evolution methods to be applied for systems that include
        tidal deformability parameters where these methods may not be applicable.
        Default False.
    disable_remnant: Bool, optional
        disable all remnant quantities from being calculated. Default False.
    add_zero_spin: Bool, optional
        if no spins are present in the posterior table, add spins with 0 value.
        Default False.
    psd_default: str/pycbc.psd obj, optional
        Default PSD to use for conversions when no other PSD is provided.
    regenerate: list, optional
        list of posterior distributions that you wish to regenerate
    return_dict: Bool, optional
        if True, return a pesummary.utils.utils.SamplesDict object
    resume_file: str, optional
        path to file to use for checkpointing. If not provided, checkpointing
        is not used. Default None

    Examples
    --------
    There are two ways of passing arguments to this conversion class, either
    a dictionary of samples or a list of parameters and a list of samples. See
    the examples below:

    >>> samples = {"mass_1": 10, "mass_2": 5}
    >>> converted_samples = %(function)s(samples)

    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [10, 5]
    >>> converted_samples = %(function)s(parameters, samples)

    >>> samples = {"mass_1": [10, 20], "mass_2": [5, 8]}
    >>> converted_samples = %(function)s(samples)

    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [[10, 5], [20, 8]]
    """


@set_docstring(_conversion_doc % {"function": "convert"})
def convert(*args, restart_from_checkpoint=False, resume_file=None, **kwargs):
    import os
    if resume_file is not None:
        if os.path.isfile(resume_file) and restart_from_checkpoint:
            return _Conversion.load_current_state(resume_file)
        logger.info(
            "Unable to find resume file for conversion. Not restarting from "
            "checkpoint"
        )
    return _Conversion(*args, resume_file=resume_file, **kwargs)


class _PickledConversion(object):
    pass


@set_docstring(_conversion_doc % {"function": "_Conversion"})
class _Conversion(object):
    @classmethod
    def load_current_state(cls, resume_file):
        """Load current state from a resume file

        Parameters
        ----------
        resume_file: str
            path to a resume file to restart conversion
        """
        from pesummary.io import read
        logger.info(
            "Reading checkpoint file: {}".format(resume_file)
        )
        state = read(resume_file, checkpoint=True)
        return cls(
            state.parameters, state.samples, extra_kwargs=state.extra_kwargs,
            evolve_spins_forwards=state.evolve_spins_forwards,
            evolve_spins_backwards=state.evolve_spins_backwards,
            NRSur_fits=state.NRSurrogate,
            waveform_fits=state.waveform_fit, multi_process=state.multi_process,
            redshift_method=state.redshift_method, cosmology=state.cosmology,
            force_non_evolved=state.force_non_evolved,
            force_BBH_remnant_computation=state.force_remnant,
            disable_remnant=state.disable_remnant,
            add_zero_spin=state.add_zero_spin, regenerate=state.regenerate,
            return_kwargs=state.return_kwargs, return_dict=state.return_dict,
            resume_file=state.resume_file
        )

    def write_current_state(self):
        """Write the current state of the conversion class to file
        """
        from pesummary.io import write
        state = _PickledConversion()
        for key, value in vars(self).items():
            setattr(state, key, value)

        _path = Path(self.resume_file)
        write(
            state, outdir=_path.parent, file_format="pickle",
            filename=_path.name, overwrite=True
        )
        logger.debug(
            "Written checkpoint file: {}".format(self.resume_file)
        )

    def __new__(cls, *args, **kwargs):
        from pesummary.utils.samples_dict import SamplesDict
        from pesummary.utils.parameters import Parameters

        obj = super(_Conversion, cls).__new__(cls)
        base_replace = (
            "'{}': {} already found in the result file. Overwriting with "
            "the passed {}"
        )
        if len(args) > 2:
            raise ValueError(
                "The _Conversion module only takes as arguments a dictionary "
                "of samples or a list of parameters and a list of samples"
            )
        elif isinstance(args[0], dict):
            parameters = Parameters(args[0].keys())
            samples = np.atleast_2d(
                np.array([args[0][i] for i in parameters]).T
            ).tolist()
        else:
            if not isinstance(args[0], Parameters):
                parameters = Parameters(args[0])
            else:
                parameters = args[0]
            samples = args[1]
            samples = np.atleast_2d(samples).tolist()
        extra_kwargs = kwargs.get("extra_kwargs", {"sampler": {}, "meta_data": {}})
        f_low = kwargs.get("f_low", None)
        f_ref = kwargs.get("f_ref", None)
        f_final = kwargs.get("f_final", None)
        delta_f = kwargs.get("delta_f", None)

        for param, value in {"f_final": f_final, "delta_f": delta_f}.items():
            if value is not None and param in extra_kwargs["meta_data"].keys():
                logger.warn(
                    base_replace.format(
                        param, extra_kwargs["meta_data"][param], value
                    )
                )
                extra_kwargs["meta_data"][param] = value
            elif value is not None:
                extra_kwargs["meta_data"][param] = value
            else:
                logger.warn(
                    "Could not find {} in input file and one was not passed "
                    "from the command line. Using {}Hz as default".format(
                        param, getattr(conf, "default_{}".format(param))
                    )
                )
                extra_kwargs["meta_data"][param] = getattr(
                    conf, "default_{}".format(param)
                )

        approximant = kwargs.get("approximant", None)
        NRSurrogate = kwargs.get("NRSur_fits", False)
        redshift_method = kwargs.get("redshift_method", "approx")
        cosmology = kwargs.get("cosmology", "Planck15")
        force_non_evolved = kwargs.get("force_non_evolved", False)
        force_remnant = kwargs.get("force_BBH_remnant_computation", False)
        force_evolve = kwargs.get("force_BH_spin_evolution", False)
        disable_remnant = kwargs.get("disable_remnant", False)
        if redshift_method not in ["approx", "exact"]:
            raise ValueError(
                "'redshift_method' can either be 'approx' corresponding to "
                "an approximant method, or 'exact' corresponding to an exact "
                "method of calculating the redshift"
            )
        if isinstance(NRSurrogate, bool) and NRSurrogate:
            raise ValueError(
                "'NRSur_fits' must be a string corresponding to the "
                "NRSurrogate model you wish to use to calculate the remnant "
                "quantities"
            )
        waveform_fits = kwargs.get("waveform_fits", False)
        evolve_spins_forwards = kwargs.get("evolve_spins_forwards", False)
        evolve_spins_backwards = kwargs.get("evolve_spins_backwards", False)
        if disable_remnant and (
                force_non_evolved or force_remnant
                or NRSurrogate or waveform_fits or evolve_spins_forwards
        ):
            _disable = []
            if force_non_evolved:
                _disable.append("force_non_evolved")
                force_non_evolved = False
            if force_remnant:
                _disable.append("force_BBH_remnant_computation")
                force_remnant = False
            if NRSurrogate:
                _disable.append("NRSur_fits")
                NRSurrogate = False
            if waveform_fits:
                _disable.append("waveform_fits")
                waveform_fits = False
            if evolve_spins_forwards:
                _disable.append("evolve_spins_forwards")
                evolve_spins_forwards = False
            logger.warn(
                "Unable to use 'disable_remnant' and {}. Setting "
                "{} and disabling all remnant quantities from being "
                "calculated".format(
                    " or ".join(_disable),
                    " and ".join(["{}=False".format(_p) for _p in _disable])
                )
            )
        if NRSurrogate and waveform_fits:
            raise ValueError(
                "Unable to use both the NRSurrogate and {} to calculate "
                "remnant quantities. Please select only one option".format(
                    approximant
                )
            )
        if isinstance(evolve_spins_forwards, bool) and evolve_spins_forwards:
            raise ValueError(
                "'evolve_spins_forwards' must be a float, the final velocity to "
                "evolve the spins up to, or a string, 'ISCO', meaning "
                "evolve the spins up to the ISCO frequency"
            )
        if not evolve_spins_forwards and (NRSurrogate or waveform_fits):
            if (approximant is not None and "eob" in approximant) or NRSurrogate:
                logger.warning(
                    "Only evolved spin remnant quantities are returned by the "
                    "{} fits.".format(
                        "NRSurrogate" if NRSurrogate else approximant
                    )
                )
        elif evolve_spins_forwards and (NRSurrogate or waveform_fits):
            if (approximant is not None and "eob" in approximant) or NRSurrogate:
                logger.warning(
                    "The {} fits already evolve the spins. Therefore "
                    "additional spin evolution will not be performed.".format(
                        "NRSurrogate" if NRSurrogate else approximant
                    )
                )
            else:
                logger.warning(
                    "The {} fits are not applied with spin evolution.".format(
                        approximant
                    )
                )
            evolve_spins_forwards = False

        precessing_snr = kwargs.get("precessing_snr", False)
        if f_low is not None and "f_low" in extra_kwargs["meta_data"].keys():
            logger.warning(
                base_replace.format(
                    "f_low", extra_kwargs["meta_data"]["f_low"], f_low
                )
            )
            extra_kwargs["meta_data"]["f_low"] = f_low
        elif f_low is not None:
            extra_kwargs["meta_data"]["f_low"] = f_low
        else:
            logger.warn(
                "Could not find minimum frequency in input file and "
                "one was not passed from the command line. Using {}Hz "
                "as default".format(conf.default_flow)
            )
            extra_kwargs["meta_data"]["f_low"] = conf.default_flow
        if approximant is not None and "approximant" in extra_kwargs["meta_data"].keys():
            logger.warning(
                base_replace.format(
                    "approximant", extra_kwargs["meta_data"]["approximant"],
                    approximant
                )
            )
            extra_kwargs["meta_data"]["approximant"] = approximant
        elif approximant is not None:
            extra_kwargs["meta_data"]["approximant"] = approximant
        if f_ref is not None and "f_ref" in extra_kwargs["meta_data"].keys():
            logger.warning(
                base_replace.format(
                    "f_ref", extra_kwargs["meta_data"]["f_ref"], f_ref
                )
            )
            extra_kwargs["meta_data"]["f_ref"] = f_ref
        elif f_ref is not None:
            extra_kwargs["meta_data"]["f_ref"] = f_ref
        regenerate = kwargs.get("regenerate", None)
        multi_process = kwargs.get("multi_process", None)
        if multi_process is not None:
            multi_process = int(multi_process)
        psd_default = kwargs.get("psd_default", "aLIGOZeroDetHighPower")
        psd = kwargs.get("psd", {})
        if psd is None:
            psd = {}
        elif psd is not None and not isinstance(psd, dict):
            raise ValueError(
                "'psd' must be a dictionary of frequency series for each detector"
            )
        ifos = list(psd.keys())
        pycbc_psd = copy.deepcopy(psd)
        if psd != {}:
            from pesummary.gw.file.psd import PSD
            if isinstance(psd[ifos[0]], PSD):
                for ifo in ifos:
                    try:
                        pycbc_psd[ifo] = pycbc_psd[ifo].to_pycbc(
                            extra_kwargs["meta_data"]["f_low"],
                            f_high=extra_kwargs["meta_data"]["f_final"],
                            f_high_override=True
                        )
                    except (ImportError, IndexError, ValueError):
                        pass
        obj.__init__(
            parameters, samples, extra_kwargs, evolve_spins_forwards, NRSurrogate,
            waveform_fits, multi_process, regenerate, redshift_method,
            cosmology, force_non_evolved, force_remnant,
            kwargs.get("add_zero_spin", False), disable_remnant,
            kwargs.get("return_kwargs", False), kwargs.get("return_dict", True),
            kwargs.get("resume_file", None), precessing_snr, pycbc_psd,
            psd_default, evolve_spins_backwards, force_evolve
        )
        return_kwargs = kwargs.get("return_kwargs", False)
        if kwargs.get("return_dict", True) and return_kwargs:
            return [
                SamplesDict(obj.parameters, np.array(obj.samples).T),
                obj.extra_kwargs
            ]
        elif kwargs.get("return_dict", True):
            return SamplesDict(obj.parameters, np.array(obj.samples).T)
        elif return_kwargs:
            return obj.parameters, obj.samples, obj.extra_kwargs
        else:
            return obj.parameters, obj.samples

    def __init__(
        self, parameters, samples, extra_kwargs, evolve_spins_forwards, NRSurrogate,
        waveform_fits, multi_process, regenerate, redshift_method,
        cosmology, force_non_evolved, force_remnant, add_zero_spin,
        disable_remnant, return_kwargs, return_dict, resume_file,
        precessing_snr, psd, psd_default, evolve_spins_backwards, force_evolve
    ):
        self.parameters = parameters
        self.samples = samples
        self.extra_kwargs = extra_kwargs
        self.evolve_spins_forwards = evolve_spins_forwards
        self.evolve_spins_backwards = evolve_spins_backwards
        self.NRSurrogate = NRSurrogate
        self.waveform_fit = waveform_fits
        self.multi_process = multi_process
        self.regenerate = regenerate
        self.redshift_method = redshift_method
        self.cosmology = cosmology
        self.force_non_evolved = force_non_evolved
        self.force_remnant = force_remnant
        self.force_evolve = force_evolve
        self.disable_remnant = disable_remnant
        self.return_kwargs = return_kwargs
        self.return_dict = return_dict
        self.resume_file = resume_file
        self.precessing_snr = precessing_snr
        self.psd = psd
        self.psd_default = psd_default
        self.non_precessing = False
        if not any(param in self.parameters for param in conf.precessing_angles):
            self.non_precessing = True
        if "chi_p" in self.parameters:
            _chi_p = self.specific_parameter_samples(["chi_p"])
            if not np.any(_chi_p):
                logger.info(
                    "chi_p = 0 for all samples. Treating this as a "
                    "non-precessing system"
                )
                self.non_precessing = True
        cond1 = self.non_precessing and evolve_spins_forwards
        cond2 = self.non_precessing and evolve_spins_backwards
        if cond1 or cond2:
            logger.info(
                "Spin evolution is trivial for a non-precessing system. No additional "
                "transformation required."
            )
            self.evolve_spins_forwards = False
            self.evolve_spins_backwards = False
        if self.non_precessing and precessing_snr:
            logger.info(
                "Precessing SNR is 0 for a non-precessing system. No additional "
                "conversion required."
            )
            self.precessing_snr = False
        self.has_tidal = self._check_for_tidal_parameters()
        self.NSBH = self._check_for_NSBH_system()
        self.compute_remnant = not self.disable_remnant
        if self.has_tidal:
            if force_evolve and (self.evolve_spins_forwards or self.evolve_spins_backwards):
                logger.warning(
                    "Posterior samples for tidal deformability found in the "
                    "posterior table. 'force_evolve' provided so using BH spin "
                    "evolution methods for this system. This may not give "
                    "sensible results"
                )
            elif self.evolve_spins_forwards or self.evolve_spins_backwards:
                logger.warning(
                    "Tidal deformability parameters found in the posterior table. "
                    "Skipping spin evolution as current methods are only valid "
                    "for BHs."
                )
                self.evolve_spins_forwards = False
                self.evolve_spins_backwards = False

            if force_remnant and self.NSBH and self.compute_remnant:
                logger.warning(
                    "Posterior samples for lambda_2 found in the posterior table "
                    "but unable to find samples for lambda_1. Assuming this "
                    "is an NSBH system. 'force_remnant' provided so using BBH remnant "
                    "fits for this system. This may not give sensible results"
                )
            elif self.NSBH and self.compute_remnant:
                logger.warning(
                    "Posterior samples for lambda_2 found in the posterior table "
                    "but unable to find samples for lambda_1. Applying NSBH "
                    "fits to this system."
                )
                self.waveform_fit = True
            elif force_remnant and self.compute_remnant:
                logger.warning(
                    "Posterior samples for tidal deformability found in the "
                    "posterior table. Applying BBH remnant fits to this system. "
                    "This may not give sensible results."
                )
            elif self.compute_remnant:
                logger.info(
                    "Skipping remnant calculations as tidal deformability "
                    "parameters found in the posterior table."
                )
                self.compute_remnant = False
        if self.regenerate is not None:
            for param in self.regenerate:
                self.remove_posterior(param)
        self.add_zero_spin = add_zero_spin
        self.generate_all_posterior_samples(evolve_spins_forwards=evolve_spins_forwards)

    def _check_for_tidal_parameters(self):
        """Check to see if any tidal parameters are stored in the table
        """
        from pesummary.gw.file.standard_names import tidal_params

        if any(param in self.parameters for param in tidal_params):
            return True
        return False

    def _check_for_NSBH_system(self):
        """Check to see if the posterior samples correspond to an NSBH
        system
        """
        if "lambda_2" in self.parameters and "lambda_1" not in self.parameters:
            return True
        elif "lambda_2" in self.parameters and "lambda_1" in self.parameters:
            _lambda_1 = self.specific_parameter_samples(["lambda_1"])
            if not np.any(_lambda_1):
                return True
        return False

    def remove_posterior(self, parameter):
        if parameter in self.parameters:
            logger.info(
                "Removing the posterior samples for '{}'".format(parameter)
            )
            ind = self.parameters.index(parameter)
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        else:
            logger.info(
                "'{}' is not in the table of posterior samples. Unable to "
                "remove".format(parameter)
            )

    def _specific_parameter_samples(self, param):
        """Return the samples for a specific parameter

        Parameters
        ----------
        param: str
            the parameter that you would like to return the samples for
        """
        if param == "empty":
            return np.array(np.zeros(len(self.samples)))
        ind = self.parameters.index(param)
        samples = np.array([i[ind] for i in self.samples])
        return samples

    def specific_parameter_samples(self, param):
        """Return the samples for either a list or a single parameter

        Parameters
        ----------
        param: list/str
            the parameter/parameters that you would like to return the samples
            for
        """
        if type(param) == list:
            samples = [self._specific_parameter_samples(i) for i in param]
        else:
            samples = self._specific_parameter_samples(param)
        return samples

    def append_data(self, parameter, samples):
        """Add a list of samples to the existing samples data object

        Parameters
        ----------
        parameter: str
            the name of the parameter you would like to append
        samples: list
            the list of samples that you would like to append
        """
        if parameter not in self.parameters:
            self.parameters.append(parameter)
            for num, i in enumerate(self.samples):
                self.samples[num].append(samples[num])
        if self.resume_file is not None:
            self.write_current_state()

    def _mchirp_from_mchirp_source_z(self):
        samples = self.specific_parameter_samples(["chirp_mass_source", "redshift"])
        chirp_mass = mchirp_from_mchirp_source_z(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _q_from_eta(self):
        samples = self.specific_parameter_samples("symmetric_mass_ratio")
        mass_ratio = q_from_eta(samples)
        self.append_data("mass_ratio", mass_ratio)

    def _q_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        mass_ratio = q_from_m1_m2(samples[0], samples[1])
        self.append_data("mass_ratio", mass_ratio)

    def _invert_q(self):
        ind = self.parameters.index("mass_ratio")
        for num, i in enumerate(self.samples):
            self.samples[num][ind] = 1. / self.samples[num][ind]

    def _invq_from_q(self):
        samples = self.specific_parameter_samples("mass_ratio")
        inverted_mass_ratio = invq_from_q(samples)
        self.append_data("inverted_mass_ratio", inverted_mass_ratio)

    def _mchirp_from_mtotal_q(self):
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        chirp_mass = mchirp_from_mtotal_q(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _m1_from_mchirp_q(self):
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_1 = m1_from_mchirp_q(samples[0], samples[1])
        self.append_data("mass_1", mass_1)

    def _m2_from_mchirp_q(self):
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_2 = m2_from_mchirp_q(samples[0], samples[1])
        self.append_data("mass_2", mass_2)

    def _m1_from_mtotal_q(self):
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        mass_1 = m1_from_mtotal_q(samples[0], samples[1])
        self.append_data("mass_1", mass_1)

    def _m2_from_mtotal_q(self):
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        mass_2 = m2_from_mtotal_q(samples[0], samples[1])
        self.append_data("mass_2", mass_2)

    def _reference_frequency(self):
        nsamples = len(self.samples)
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "f_ref" in list(extra_kwargs.keys()):
            self.append_data(
                "reference_frequency", [float(extra_kwargs["f_ref"])] * nsamples
            )
        else:
            logger.warning(
                "Could not find reference_frequency in input file. Using 20Hz "
                "as default")
            self.append_data("reference_frequency", [20.] * nsamples)

    def _mtotal_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        m_total = m_total_from_m1_m2(samples[0], samples[1])
        self.append_data("total_mass", m_total)

    def _mchirp_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        chirp_mass = mchirp_from_m1_m2(samples[0], samples[1])
        self.append_data("chirp_mass", chirp_mass)

    def _eta_from_m1_m2(self):
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        eta = eta_from_m1_m2(samples[0], samples[1])
        self.append_data("symmetric_mass_ratio", eta)

    def _phi_12_from_phi1_phi2(self):
        samples = self.specific_parameter_samples(["phi_1", "phi_2"])
        phi_12 = phi_12_from_phi1_phi2(samples[0], samples[1])
        self.append_data("phi_12", phi_12)

    def _phi1_from_spins(self):
        samples = self.specific_parameter_samples(["spin_1x", "spin_1y"])
        phi_1 = phi1_from_spins(samples[0], samples[1])
        self.append_data("phi_1", phi_1)

    def _phi2_from_spins(self):
        samples = self.specific_parameter_samples(["spin_2x", "spin_2y"])
        phi_2 = phi2_from_spins(samples[0], samples[1])
        self.append_data("phi_2", phi_2)

    def _spin_angles(self):
        _spin_angles = ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12",
                        "a_1", "a_2"]
        spin_angles_to_calculate = [
            i for i in _spin_angles if i not in self.parameters]
        spin_components = [
            "mass_1", "mass_2", "iota", "spin_1x", "spin_1y", "spin_1z",
            "spin_2x", "spin_2y", "spin_2z", "reference_frequency"]
        samples = self.specific_parameter_samples(spin_components)
        if "phase" in self.parameters:
            spin_components.append("phase")
            samples.append(self.specific_parameter_samples("phase"))
        else:
            logger.warning(
                "Phase it not given, we will be assuming that a "
                "reference phase of 0 to calculate all the spin angles"
            )
            samples.append([0] * len(samples[0]))
        angles = spin_angles(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spin_angles_to_calculate:
            ind = _spin_angles.index(i)
            data = np.array([i[ind] for i in angles])
            self.append_data(i, data)

    def _non_precessing_component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        angles = ["a_1", "a_2", "theta_jn", "tilt_1", "tilt_2"]
        if all(i in self.parameters for i in angles):
            samples = self.specific_parameter_samples(angles)
            cond1 = all(i in [0, np.pi] for i in samples[3])
            cond2 = all(i in [0, np.pi] for i in samples[4])
            spins_to_calculate = [
                i for i in spins if i not in self.parameters]
            if cond1 and cond1:
                spin_1x = np.array([0.] * len(samples[0]))
                spin_1y = np.array([0.] * len(samples[0]))
                spin_1z = samples[0] * np.cos(samples[3])
                spin_2x = np.array([0.] * len(samples[0]))
                spin_2y = np.array([0.] * len(samples[0]))
                spin_2z = samples[1] * np.cos(samples[4])
                iota = np.array(samples[2])
                spin_components = [
                    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z]

                for i in spins_to_calculate:
                    ind = spins.index(i)
                    data = spin_components[ind]
                    self.append_data(i, data)

    def _component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        angles = [
            "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
            "mass_1", "mass_2", "reference_frequency"]
        samples = self.specific_parameter_samples(angles)
        if "phase" in self.parameters:
            angles.append("phase")
            samples.append(self.specific_parameter_samples("phase"))
        else:
            logger.warning(
                "Phase it not given, we will be assuming that a "
                "reference phase of 0 to calculate all the spin angles"
            )
            samples.append([0] * len(samples[0]))
        spin_components = component_spins(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spins_to_calculate:
            ind = spins.index(i)
            data = np.array([i[ind] for i in spin_components])
            self.append_data(i, data)

    def _component_spins_from_azimuthal_and_polar_angles(self):
        spins = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        angles = [
            "a_1", "a_2", "a_1_azimuthal", "a_1_polar", "a_2_azimuthal",
            "a_2_polar"]
        samples = self.specific_parameter_samples(angles)
        spin_components = spin_angles_from_azimuthal_and_polar_angles(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        for i in spins_to_calculate:
            ind = spins.index(i)
            data = np.array([i[ind] for i in spin_components])
            self.append_data(i, data)

    def _chi_p(self):
        parameters = [
            "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]
        samples = self.specific_parameter_samples(parameters)
        chi_p_samples = chi_p(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        self.append_data("chi_p", chi_p_samples)

    def _chi_p_from_tilts(self, suffix=""):
        parameters = [
            "mass_1", "mass_2", "a_1", "tilt_1{}".format(suffix), "a_2",
            "tilt_2{}".format(suffix)
        ]
        samples = self.specific_parameter_samples(parameters)
        chi_p_samples = chi_p_from_tilts(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5]
        )
        self.append_data("chi_p{}".format(suffix), chi_p_samples)

    def _chi_p_2spin(self):
        parameters = [
            "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]
        samples = self.specific_parameter_samples(parameters)
        chi_p_2spin_samples = chi_p_2spin(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        self.append_data("chi_p_2spin", chi_p_2spin_samples)

    def _chi_eff(self, suffix=""):
        parameters = [
            "mass_1", "mass_2", "spin_1z{}".format(suffix),
            "spin_2z{}".format(suffix)
        ]
        samples = self.specific_parameter_samples(parameters)
        chi_eff_samples = chi_eff(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("chi_eff{}".format(suffix), chi_eff_samples)

    def _aligned_spin_from_magnitude_tilts(
        self, primary=False, secondary=False, suffix=""
    ):
        if primary:
            parameters = ["a_1", "tilt_1{}".format(suffix)]
            param_to_add = "spin_1z{}".format(suffix)
        elif secondary:
            parameters = ["a_2", "tilt_2{}".format(suffix)]
            param_to_add = "spin_2z{}".format(suffix)
        samples = self.specific_parameter_samples(parameters)
        spin_samples = samples[0] * np.cos(samples[1])
        self.append_data(param_to_add, spin_samples)

    def _cos_tilt_1_from_tilt_1(self):
        samples = self.specific_parameter_samples("tilt_1")
        cos_tilt_1 = np.cos(samples)
        self.append_data("cos_tilt_1", cos_tilt_1)

    def _cos_tilt_2_from_tilt_2(self):
        samples = self.specific_parameter_samples("tilt_2")
        cos_tilt_2 = np.cos(samples)
        self.append_data("cos_tilt_2", cos_tilt_2)

    def _viewing_angle(self):
        samples = self.specific_parameter_samples("theta_jn")
        viewing_angle = viewing_angle_from_inclination(samples)
        self.append_data("viewing_angle", viewing_angle)

    def _dL_from_z(self):
        samples = self.specific_parameter_samples("redshift")
        distance = dL_from_z(samples, cosmology=self.cosmology)
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("luminosity_distance", distance)

    def _z_from_dL(self):
        samples = self.specific_parameter_samples("luminosity_distance")
        func = getattr(Redshift, self.redshift_method)
        redshift = func(
            samples, cosmology=self.cosmology, multi_process=self.multi_process
        )
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("redshift", redshift)

    def _comoving_distance_from_z(self):
        samples = self.specific_parameter_samples("redshift")
        distance = comoving_distance_from_z(samples, cosmology=self.cosmology)
        self.extra_kwargs["meta_data"]["cosmology"] = self.cosmology
        self.append_data("comoving_distance", distance)

    def _m1_source_from_m1_z(self):
        samples = self.specific_parameter_samples(["mass_1", "redshift"])
        mass_1_source = m1_source_from_m1_z(samples[0], samples[1])
        self.append_data("mass_1_source", mass_1_source)

    def _m1_from_m1_source_z(self):
        samples = self.specific_parameter_samples(["mass_1_source", "redshift"])
        mass_1 = m1_from_m1_source_z(samples[0], samples[1])
        self.append_data("mass_1", mass_1)

    def _m2_source_from_m2_z(self):
        samples = self.specific_parameter_samples(["mass_2", "redshift"])
        mass_2_source = m2_source_from_m2_z(samples[0], samples[1])
        self.append_data("mass_2_source", mass_2_source)

    def _m2_from_m2_source_z(self):
        samples = self.specific_parameter_samples(["mass_2_source", "redshift"])
        mass_2 = m2_from_m2_source_z(samples[0], samples[1])
        self.append_data("mass_2", mass_2)

    def _mtotal_source_from_mtotal_z(self):
        samples = self.specific_parameter_samples(["total_mass", "redshift"])
        total_mass_source = m_total_source_from_mtotal_z(samples[0], samples[1])
        self.append_data("total_mass_source", total_mass_source)

    def _mtotal_from_mtotal_source_z(self):
        samples = self.specific_parameter_samples(["total_mass_source", "redshift"])
        total_mass = mtotal_from_mtotal_source_z(samples[0], samples[1])
        self.append_data("total_mass", total_mass)

    def _mchirp_source_from_mchirp_z(self):
        samples = self.specific_parameter_samples(["chirp_mass", "redshift"])
        chirp_mass_source = mchirp_source_from_mchirp_z(samples[0], samples[1])
        self.append_data("chirp_mass_source", chirp_mass_source)

    def _beta(self):
        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", "phi_jl", "tilt_1", "tilt_2", "phi_12",
            "a_1", "a_2", "reference_frequency", "phase"
        ])
        beta = opening_angle(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9]
        )
        self.append_data("beta", beta)

    def _psi_J(self):
        samples = self.specific_parameter_samples([
            "psi", "theta_jn", "phi_jl", "beta"
        ])
        psi = psi_J(samples[0], samples[1], samples[2], samples[3])
        self.append_data("psi_J", psi)

    def _time_in_each_ifo(self):
        detectors = []
        if "IFOs" in list(self.extra_kwargs["meta_data"].keys()):
            detectors = self.extra_kwargs["meta_data"]["IFOs"].split(" ")
        else:
            for i in self.parameters:
                if "optimal_snr" in i and i != "network_optimal_snr":
                    det = i.split("_optimal_snr")[0]
                    detectors.append(det)

        samples = self.specific_parameter_samples(["ra", "dec", "geocent_time"])
        for i in detectors:
            time = time_in_each_ifo(i, samples[0], samples[1], samples[2])
            self.append_data("%s_time" % (i), time)

    def _lambda1_from_lambda_tilde(self):
        samples = self.specific_parameter_samples([
            "lambda_tilde", "mass_1", "mass_2"])
        lambda_1 = lambda1_from_lambda_tilde(samples[0], samples[1], samples[2])
        self.append_data("lambda_1", lambda_1)

    def _lambda2_from_lambda1(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "mass_1", "mass_2"])
        lambda_2 = lambda2_from_lambda1(samples[0], samples[1], samples[2])
        self.append_data("lambda_2", lambda_2)

    def _lambda_tilde_from_lambda1_lambda2(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        lambda_tilde = lambda_tilde_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("lambda_tilde", lambda_tilde)

    def _delta_lambda_from_lambda1_lambda2(self):
        samples = self.specific_parameter_samples([
            "lambda_1", "lambda_2", "mass_1", "mass_2"])
        delta_lambda = delta_lambda_from_lambda1_lambda2(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data("delta_lambda", delta_lambda)

    def _NS_compactness_from_lambda(self, parameter="lambda_1"):
        if parameter not in ["lambda_1", "lambda_2"]:
            logger.warn(
                "Can only use Love-compactness relation for 'lambda_1' and/or "
                "'lambda_2'. Skipping conversion"
            )
            return
        ind = parameter.split("lambda_")[1]
        samples = self.specific_parameter_samples([parameter])
        compactness = NS_compactness_from_lambda(samples[0])
        self.append_data("compactness_{}".format(ind), compactness)
        self.extra_kwargs["meta_data"]["compactness_fits"] = (
            "YagiYunes2017_with_BBHlimit"
        )

    def _NS_baryonic_mass(self, primary=True):
        if primary:
            params = ["compactness_1", "mass_1"]
        else:
            params = ["compactness_2", "mass_2"]
        samples = self.specific_parameter_samples(params)
        mass = NS_baryonic_mass(samples[0], samples[1])
        if primary:
            self.append_data("baryonic_mass_1", mass)
        else:
            self.append_data("baryonic_mass_2", mass)
        self.extra_kwargs["meta_data"]["baryonic_mass_fits"] = "Breu2016"

    def _lambda1_lambda2_from_polytrope_EOS(self):
        samples = self.specific_parameter_samples([
            "log_pressure", "gamma_1", "gamma_2", "gamma_3", "mass_1", "mass_2"
        ])
        lambda_1, lambda_2 = \
            lambda1_lambda2_from_4_parameter_piecewise_polytrope_equation_of_state(
                *samples, multi_process=self.multi_process
            )
        if "lambda_1" not in self.parameters:
            self.append_data("lambda_1", lambda_1)
        if "lambda_2" not in self.parameters:
            self.append_data("lambda_2", lambda_2)

    def _lambda1_lambda2_from_spectral_decomposition_EOS(self):
        samples = self.specific_parameter_samples([
            "spectral_decomposition_gamma_0", "spectral_decomposition_gamma_1",
            "spectral_decomposition_gamma_2", "spectral_decomposition_gamma_3",
            "mass_1", "mass_2"
        ])
        lambda_1, lambda_2 = lambda1_lambda2_from_spectral_decomposition(
            *samples, multi_process=self.multi_process
        )
        if "lambda_1" not in self.parameters:
            self.append_data("lambda_1", lambda_1)
        if "lambda_2" not in self.parameters:
            self.append_data("lambda_2", lambda_2)

    def _ifo_snr(self):
        abs_snrs = [
            i for i in self.parameters if "_matched_filter_abs_snr" in i
        ]
        angle_snrs = [
            i for i in self.parameters if "_matched_filter_snr_angle" in i
        ]
        for ifo in [snr.split("_matched_filter_abs_snr")[0] for snr in abs_snrs]:
            if "{}_matched_filter_snr".format(ifo) not in self.parameters:
                samples = self.specific_parameter_samples(
                    [
                        "{}_matched_filter_abs_snr".format(ifo),
                        "{}_matched_filter_snr_angle".format(ifo)
                    ]
                )
                snr = _ifo_snr(samples[0], samples[1])
                self.append_data("{}_matched_filter_snr".format(ifo), snr)

    def _optimal_network_snr(self):
        snrs = [i for i in self.parameters if "_optimal_snr" in i]
        samples = self.specific_parameter_samples(snrs)
        snr = network_snr(samples)
        self.append_data("network_optimal_snr", snr)

    def _matched_filter_network_snr(self):
        mf_snrs = sorted([
            i for i in self.parameters if "_matched_filter_snr" in i
            and "_angle" not in i and "_abs" not in i
        ])
        opt_snrs = sorted([
            i for i in self.parameters if "_optimal_snr" in i and "network" not
            in i
        ])
        _mf_detectors = [
            param.split("_matched_filter_snr")[0] for param in mf_snrs
        ]
        _opt_detectors = [
            param.split("_optimal_snr")[0] for param in opt_snrs
        ]
        if _mf_detectors == _opt_detectors:
            mf_samples = self.specific_parameter_samples(mf_snrs)
            opt_samples = self.specific_parameter_samples(opt_snrs)
            snr = network_matched_filter_snr(mf_samples, opt_samples)
            self.append_data("network_matched_filter_snr", snr)
        else:
            logger.warn(
                "Unable to generate 'network_matched_filter_snr' as "
                "there is an inconsistency in the detector network based on "
                "the 'optimal_snrs' and the 'matched_filter_snrs'. We find "
                "that from the 'optimal_snrs', the detector network is: {} "
                "while we find from the 'matched_filter_snrs', the detector "
                "network is: {}".format(_opt_detectors, _mf_detectors)
            )

    def _rho_p(self):
        required = [
            "mass_1", "mass_2", "beta", "psi_J", "a_1", "a_2", "tilt_1",
            "tilt_2", "phi_12", "theta_jn", "ra", "dec", "geocent_time",
            "phi_jl", "reference_frequency", "luminosity_distance", "phase"
        ]
        samples = self.specific_parameter_samples(required)
        try:
            spins = self.specific_parameter_samples(["spin_1z", "spin_2z"])
        except ValueError:
            spins = [None, None]
        _f_low = self._retrieve_f_low()
        if isinstance(_f_low, (np.ndarray)):
            f_low = _f_low() * len(samples[0])
        else:
            f_low = [_f_low] * len(samples[0])
        [rho_p, b_bar, overlap, snrs], data_used = precessing_snr(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9], samples[10],
            samples[11], samples[12], samples[13], samples[15], samples[16], f_low=f_low,
            spin_1z=spins[0], spin_2z=spins[1], psd=self.psd, return_data_used=True,
            f_final=self.extra_kwargs["meta_data"]["f_final"], f_ref=samples[14],
            multi_process=self.multi_process, psd_default=self.psd_default,
            df=self.extra_kwargs["meta_data"]["delta_f"], debug=True
        )
        self.append_data("network_precessing_snr", rho_p)
        self.append_data("_b_bar", b_bar)
        self.append_data("_precessing_harmonics_overlap", overlap)
        nbreakdown = len(np.argwhere(b_bar > 0.3))
        if nbreakdown > 0:
            logger.warning(
                "{}/{} ({}%) samples have b_bar greater than 0.3. For these "
                "samples, the two-harmonic approximation used to calculate "
                "the precession SNR may not be valid".format(
                    nbreakdown, len(b_bar),
                    np.round((nbreakdown / len(b_bar)) * 100, 2)
                )
            )
        try:
            _samples = self.specific_parameter_samples("network_optimal_snr")
            if np.logical_or(
                    np.median(snrs) > 1.1 * np.median(_samples),
                    np.median(snrs) < 0.9 * np.median(_samples)
            ):
                logger.warn(
                    "The two-harmonic SNR is different from the stored SNR. "
                    "This indicates that the provided PSD may be different "
                    "from the one used in the sampling."
                )
        except Exception:
            pass
        self.extra_kwargs["meta_data"]["precessing_snr"] = data_used

    def _retrieve_f_low(self):
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "f_low" in list(extra_kwargs.keys()):
            f_low = extra_kwargs["f_low"]
        else:
            raise ValueError(
                "Could not find f_low in input file. Please either modify the "
                "input file or pass it from the command line"
            )
        return f_low

    def _retrieve_approximant(self):
        extra_kwargs = self.extra_kwargs["meta_data"]
        if extra_kwargs != {} and "approximant" in list(extra_kwargs.keys()):
            approximant = extra_kwargs["approximant"]
        else:
            raise ValueError(
                "Unable to find the approximant used to generate the posterior "
                "samples in the result file."
            )
        return approximant

    def _evolve_spins(self, final_velocity="ISCO", forward=True):
        from .evolve import evolve_spins

        parameters = ["tilt_1", "tilt_2", "phi_12", "spin_1z", "spin_2z"]
        samples = self.specific_parameter_samples(
            ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2",
             "phi_12", "reference_frequency"]
        )
        if not forward:
            [tilt_1_evolved, tilt_2_evolved, phi_12_evolved], fits_used = evolve_spins(
                samples[0], samples[1], samples[2], samples[3], samples[4],
                samples[5], samples[6], samples[7][0],
                evolve_limit="infinite_separation", multi_process=self.multi_process,
                return_fits_used=True, method=self.evolve_spins_backwards
            )
            suffix = ""
            if self.evolve_spins_backwards.lower() == "precession_averaged":
                suffix = "_only_prec_avg"
            self.append_data("tilt_1_infinity{}".format(suffix), tilt_1_evolved)
            self.append_data("tilt_2_infinity{}".format(suffix), tilt_2_evolved)
            self.extra_kwargs["meta_data"]["backward_spin_evolution"] = fits_used
            return
        else:
            f_low = self._retrieve_f_low()
            approximant = self._retrieve_approximant()
            if not hasattr(lalsimulation, approximant):
                _msg = (
                    'Not evolving spins: approximant {0} unknown to '
                    'lalsimulation'.format(approximant)
                )
                logger.warning(_msg)
                raise EvolveSpinError(_msg)
            tilt_1_evolved, tilt_2_evolved, phi_12_evolved = evolve_spins(
                samples[0], samples[1], samples[2], samples[3], samples[4],
                samples[5], samples[6], f_low, samples[7][0],
                approximant, final_velocity=final_velocity,
                multi_process=self.multi_process
            )
            self.extra_kwargs["meta_data"]["forward_spin_evolution"] = final_velocity
        spin_1z_evolved = samples[2] * np.cos(tilt_1_evolved)
        spin_2z_evolved = samples[3] * np.cos(tilt_2_evolved)
        self.append_data("tilt_1_evolved", tilt_1_evolved)
        self.append_data("tilt_2_evolved", tilt_2_evolved)
        self.append_data("phi_12_evolved", phi_12_evolved)
        self.append_data("spin_1z_evolved", spin_1z_evolved)
        self.append_data("spin_2z_evolved", spin_2z_evolved)

    @staticmethod
    def _evolved_vs_non_evolved_parameter(
        parameter, evolved=False, core_param=False, non_precessing=False
    ):
        if non_precessing:
            base_string = ""
        elif evolved and core_param:
            base_string = "_evolved"
        elif evolved:
            base_string = ""
        elif core_param:
            base_string = ""
        else:
            base_string = "_non_evolved"
        return "{}{}".format(parameter, base_string)

    def _precessing_vs_non_precessing_parameters(
        self, non_precessing=False, evolved=False
    ):
        if not non_precessing:
            tilt_1 = self._evolved_vs_non_evolved_parameter(
                "tilt_1", evolved=evolved, core_param=True
            )
            tilt_2 = self._evolved_vs_non_evolved_parameter(
                "tilt_2", evolved=evolved, core_param=True
            )
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", "a_1", "a_2", tilt_1, tilt_2
            ])
            if "phi_12" in self.parameters and evolved:
                phi_12_samples = self.specific_parameter_samples([
                    self._evolved_vs_non_evolved_parameter(
                        "phi_12", evolved=True, core_param=True
                    )
                ])[0]
            elif "phi_12" in self.parameters:
                phi_12_samples = self.specific_parameter_samples(["phi_12"])[0]
            else:
                phi_12_samples = np.zeros_like(samples[0])
            samples.append(phi_12_samples)
            if self.NRSurrogate:
                NRSurrogate_samples = self.specific_parameter_samples([
                    "phi_jl", "theta_jn", "phase"
                ])
                for ss in NRSurrogate_samples:
                    samples.append(ss)
        else:
            spin_1z = self._evolved_vs_non_evolved_parameter(
                "spin_1z", evolved=evolved, core_param=True, non_precessing=True
            )
            spin_2z = self._evolved_vs_non_evolved_parameter(
                "spin_2z", evolved=evolved, core_param=True, non_precessing=True
            )
            samples = self.specific_parameter_samples([
                "mass_1", "mass_2", spin_1z, spin_2z
            ])
            samples = [
                samples[0], samples[1], np.abs(samples[2]), np.abs(samples[3]),
                0.5 * np.pi * (1 - np.sign(samples[2])),
                0.5 * np.pi * (1 - np.sign(samples[3])),
                np.zeros_like(samples[0])
            ]
        return samples

    def _peak_luminosity_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "peak_luminosity", evolved=evolved, non_precessing=self.non_precessing
        )
        spin_1z_param = self._evolved_vs_non_evolved_parameter(
            "spin_1z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )
        spin_2z_param = self._evolved_vs_non_evolved_parameter(
            "spin_2z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )

        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", spin_1z_param, spin_2z_param
        ])
        peak_luminosity, fits = peak_luminosity_of_merger(
            samples[0], samples[1], samples[2], samples[3],
            return_fits_used=True
        )
        self.append_data(param, peak_luminosity)
        self.extra_kwargs["meta_data"]["peak_luminosity_NR_fits"] = fits

    def _final_remnant_properties_from_NRSurrogate(
        self, non_precessing=False,
        parameters=["final_mass", "final_spin", "final_kick"]
    ):
        f_low = self._retrieve_f_low()
        approximant = self._retrieve_approximant()
        samples = self._precessing_vs_non_precessing_parameters(
            non_precessing=non_precessing, evolved=False
        )
        frequency_samples = self.specific_parameter_samples([
            "reference_frequency"
        ])
        data, fits = final_remnant_properties_from_NRSurrogate(
            *samples, f_low=f_low, f_ref=frequency_samples[0],
            properties=parameters, return_fits_used=True,
            approximant=approximant
        )
        for param in parameters:
            self.append_data(param, data[param])
            self.extra_kwargs["meta_data"]["{}_NR_fits".format(param)] = fits

    def _final_remnant_properties_from_NSBH_waveform(
        self, source=False, parameters=[
            "baryonic_torus_mass", "final_mass", "final_spin"
        ]
    ):
        approximant = self._retrieve_approximant()
        if source:
            sample_params = [
                "mass_1_source", "mass_2_source", "spin_1z", "lambda_2"
            ]
        else:
            sample_params = ["mass_1", "mass_2", "spin_1z", "lambda_2"]
        samples = self.specific_parameter_samples(sample_params)
        _data = _check_NSBH_approximant(
            approximant, samples[0], samples[1], samples[2], samples[3],
            _raise=False
        )
        if _data is None:
            return
        _mapping = {
            "220_quasinormal_mode_frequency": 0, "tidal_disruption_frequency": 1,
            "baryonic_torus_mass": 2, "compactness_2": 3,
            "final_mass": 4, "final_spin": 5
        }
        for param in parameters:
            self.append_data(param, _data[_mapping[param]])
        if "final_mass" in parameters:
            self.extra_kwargs["meta_data"]["final_mass_NR_fits"] = "Zappa2019"
        if "final_spin" in parameters:
            self.extra_kwargs["meta_data"]["final_spin_NR_fits"] = "Zappa2019"
        if "baryonic_torus_mass" in parameters:
            self.extra_kwargs["meta_data"]["baryonic_torus_mass_fits"] = (
                "Foucart2012"
            )
        if "220_quasinormal_mode_frequency" in parameters:
            self.extra_kwargs["meta_data"]["quasinormal_mode_fits"] = (
                "London2019"
            )
        if "tidal_disruption_frequency" in parameters:
            probabilities = NSBH_merger_type(
                samples[0], samples[1], samples[2], samples[3],
                approximant=approximant,
                _ringdown=_data[_mapping["220_quasinormal_mode_frequency"]],
                _disruption=_data[_mapping["tidal_disruption_frequency"]],
                _torus=_data[_mapping["baryonic_torus_mass"]], percentages=True
            )
            self.extra_kwargs["meta_data"]["NSBH_merger_type_probabilities"] = (
                probabilities
            )
            self.extra_kwargs["meta_data"]["tidal_disruption_frequency_fits"] = (
                "Pannarale2018"
            )
            ratio = (
                _data[_mapping["tidal_disruption_frequency"]]
                / _data[_mapping["220_quasinormal_mode_frequency"]]
            )
            self.append_data(
                "tidal_disruption_frequency_ratio", ratio
            )

    def _final_remnant_properties_from_waveform(
        self, non_precessing=False, parameters=["final_mass", "final_spin"],
    ):
        f_low = self._retrieve_f_low()
        approximant = self._retrieve_approximant()
        if "delta_t" in self.extra_kwargs["meta_data"].keys():
            delta_t = self.extra_kwargs["meta_data"]["delta_t"]
        else:
            delta_t = 1. / 4096
            if "seob" in approximant.lower():
                logger.warning(
                    "Could not find 'delta_t' in the meta data. Using {} as "
                    "default.".format(delta_t)
                )
        if non_precessing:
            sample_params = [
                "mass_1", "mass_2", "empty", "empty", "spin_1z", "empty",
                "empty", "spin_2z", "iota", "luminosity_distance",
                "phase"
            ]
        else:
            sample_params = [
                "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z",
                "spin_2x", "spin_2y", "spin_2z", "iota", "luminosity_distance",
                "phase"
            ]
        samples = self.specific_parameter_samples(sample_params)
        ind = self.parameters.index("spin_1x")
        _data, fits = _final_from_initial_BBH(
            *samples[:8], iota=samples[8], luminosity_distance=samples[9],
            f_ref=[f_low] * len(samples[0]), phi_ref=samples[10],
            delta_t=1. / 4096, approximant=approximant, return_fits_used=True,
            multi_process=self.multi_process
        )
        data = {"final_mass": _data[0], "final_spin": _data[1]}
        for param in parameters:
            self.append_data(param, data[param])
            self.extra_kwargs["meta_data"]["{}_NR_fits".format(param)] = fits

    def _final_mass_of_merger(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved, non_precessing=self.non_precessing
        )
        spin_1z_param = self._evolved_vs_non_evolved_parameter(
            "spin_1z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )
        spin_2z_param = self._evolved_vs_non_evolved_parameter(
            "spin_2z", evolved=evolved, core_param=True,
            non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([
            "mass_1", "mass_2", spin_1z_param, spin_2z_param
        ])
        final_mass, fits = final_mass_of_merger(
            *samples, return_fits_used=True
        )
        self.append_data(param, final_mass)
        self.extra_kwargs["meta_data"]["final_mass_NR_fits"] = fits

    def _final_mass_source(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_mass", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([param, "redshift"])
        final_mass_source = _source_from_detector(
            samples[0], samples[1]
        )
        self.append_data(param.replace("mass", "mass_source"), final_mass_source)

    def _final_spin_of_merger(self, non_precessing=False, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "final_spin", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self._precessing_vs_non_precessing_parameters(
            non_precessing=non_precessing, evolved=evolved
        )
        final_spin, fits = final_spin_of_merger(
            *samples, return_fits_used=True
        )
        self.append_data(param, final_spin)
        self.extra_kwargs["meta_data"]["final_spin_NR_fits"] = fits

    def _radiated_energy(self, evolved=False):
        param = self._evolved_vs_non_evolved_parameter(
            "radiated_energy", evolved=evolved, non_precessing=self.non_precessing
        )
        final_mass_param = self._evolved_vs_non_evolved_parameter(
            "final_mass_source", evolved=evolved, non_precessing=self.non_precessing
        )
        samples = self.specific_parameter_samples([
            "total_mass_source", final_mass_param
        ])
        radiated_energy = samples[0] - samples[1]
        self.append_data(param, radiated_energy)

    def _cos_angle(self, parameter_to_add, reverse=False):
        if reverse:
            samples = self.specific_parameter_samples(
                ["cos_" + parameter_to_add])
            cos_samples = np.arccos(samples[0])
        else:
            samples = self.specific_parameter_samples(
                [parameter_to_add.split("cos_")[1]]
            )
            cos_samples = np.cos(samples[0])
        self.append_data(parameter_to_add, cos_samples)

    def source_frame_from_detector_frame(self, detector_frame_parameter):
        samples = self.specific_parameter_samples(
            [detector_frame_parameter, "redshift"]
        )
        source_frame = _source_from_detector(samples[0], samples[1])
        self.append_data(
            "{}_source".format(detector_frame_parameter), source_frame
        )

    def _check_parameters(self):
        params = ["mass_1", "mass_2", "a_1", "a_2", "mass_1_source", "mass_2_source",
                  "mass_ratio", "total_mass", "chirp_mass"]
        for i in params:
            if i in self.parameters:
                samples = self.specific_parameter_samples([i])
                if "mass" in i:
                    cond = any(np.array(samples[0]) <= 0.)
                else:
                    cond = any(np.array(samples[0]) < 0.)
                if cond:
                    if "mass" in i:
                        ind = np.argwhere(np.array(samples[0]) <= 0.)
                    else:
                        ind = np.argwhere(np.array(samples[0]) < 0.)
                    logger.warning(
                        "Removing %s samples because they have unphysical "
                        "values (%s < 0)" % (len(ind), i)
                    )
                    for i in np.arange(len(ind) - 1, -1, -1):
                        self.samples.remove(list(np.array(self.samples)[ind[i][0]]))

    def generate_all_posterior_samples(self, evolve_spins_forwards=False):
        logger.debug("Starting to generate all derived posteriors")
        evolve_condition = (
            True if evolve_spins_forwards and self.compute_remnant else False
        )
        if "cos_theta_jn" in self.parameters and "theta_jn" not in self.parameters:
            self._cos_angle("theta_jn", reverse=True)
        if "cos_iota" in self.parameters and "iota" not in self.parameters:
            self._cos_angle("iota", reverse=True)
        if "cos_tilt_1" in self.parameters and "tilt_1" not in self.parameters:
            self._cos_angle("tilt_1", reverse=True)
        if "cos_tilt_2" in self.parameters and "tilt_2" not in self.parameters:
            self._cos_angle("tilt_2", reverse=True)
        angles = [
            "a_1", "a_2", "a_1_azimuthal", "a_1_polar", "a_2_azimuthal",
            "a_2_polar"
        ]
        if all(i in self.parameters for i in angles):
            self._component_spins_from_azimuthal_and_polar_angles()
        spin_magnitudes = ["a_1", "a_2"]
        angles = ["phi_jl", "tilt_1", "tilt_2", "phi_12"]
        cartesian = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
        cond1 = all(i in self.parameters for i in spin_magnitudes)
        cond2 = all(i in self.parameters for i in angles)
        cond3 = all(i in self.parameters for i in cartesian)
        for _param in spin_magnitudes:
            if _param in self.parameters and not cond2 and not cond3:
                _index = _param.split("a_")[1]
                _spin = self.specific_parameter_samples(_param)
                _tilt = np.arccos(np.sign(_spin))
                self.append_data("tilt_{}".format(_index), _tilt)
                _spin_ind = self.parameters.index(_param)
                for num, i in enumerate(self.samples):
                    self.samples[num][_spin_ind] = abs(self.samples[num][_spin_ind])

        if not cond2 and not cond3 and self.add_zero_spin:
            for _param in spin_magnitudes:
                if _param not in self.parameters:
                    _spin = np.zeros(len(self.samples))
                    self.append_data(_param, _spin)
                    _index = _param.split("a_")[1]
                    self.append_data("spin_{}z".format(_index), _spin)
        self._check_parameters()
        if "cos_theta_jn" in self.parameters and "theta_jn" not in self.parameters:
            self._cos_angle("theta_jn", reverse=True)
        if "cos_iota" in self.parameters and "iota" not in self.parameters:
            self._cos_angle("iota", reverse=True)
        if "cos_tilt_1" in self.parameters and "tilt_1" not in self.parameters:
            self._cos_angle("tilt_1", reverse=True)
        if "cos_tilt_2" in self.parameters and "tilt_2" not in self.parameters:
            self._cos_angle("tilt_2", reverse=True)
        if "luminosity_distance" not in self.parameters:
            if "redshift" in self.parameters:
                self._dL_from_z()
        if "redshift" not in self.parameters:
            if "luminosity_distance" in self.parameters:
                self._z_from_dL()
        if "comoving_distance" not in self.parameters:
            if "redshift" in self.parameters:
                self._comoving_distance_from_z()

        if "mass_ratio" not in self.parameters and "symmetric_mass_ratio" in \
                self.parameters:
            self._q_from_eta()
        if "mass_ratio" not in self.parameters and "mass_1" in self.parameters \
                and "mass_2" in self.parameters:
            self._q_from_m1_m2()
        if "mass_ratio" in self.parameters:
            ind = self.parameters.index("mass_ratio")
            median = np.median([i[ind] for i in self.samples])
            if median > 1.:
                self._invert_q()
        if "inverted_mass_ratio" not in self.parameters and "mass_ratio" in \
                self.parameters:
            self._invq_from_q()
        if "chirp_mass" not in self.parameters and "total_mass" in self.parameters:
            self._mchirp_from_mtotal_q()
        if "mass_1" not in self.parameters and "chirp_mass" in self.parameters:
            self._m1_from_mchirp_q()
        if "mass_2" not in self.parameters and "chirp_mass" in self.parameters:
            self._m2_from_mchirp_q()
        if "mass_1" not in self.parameters and "total_mass" in self.parameters:
            self._m1_from_mtotal_q()
        if "mass_2" not in self.parameters and "total_mass" in self.parameters:
            self._m2_from_mtotal_q()
        if "mass_1" in self.parameters and "mass_2" in self.parameters:
            if "total_mass" not in self.parameters:
                self._mtotal_from_m1_m2()
            if "chirp_mass" not in self.parameters:
                self._mchirp_from_m1_m2()
            if "symmetric_mass_ratio" not in self.parameters:
                self._eta_from_m1_m2()
        if "redshift" in self.parameters:
            if "mass_1_source" not in self.parameters:
                if "mass_1" in self.parameters:
                    self._m1_source_from_m1_z()
            if "mass_1_source" in self.parameters:
                if "mass_1" not in self.parameters:
                    self._m1_from_m1_source_z()
            if "mass_2_source" not in self.parameters:
                if "mass_2" in self.parameters:
                    self._m2_source_from_m2_z()
            if "mass_2_source" in self.parameters:
                if "mass_2" not in self.parameters:
                    self._m2_from_m2_source_z()
            if "total_mass_source" not in self.parameters:
                if "total_mass" in self.parameters:
                    self._mtotal_source_from_mtotal_z()
            if "total_mass_source" in self.parameters:
                if "total_mass" not in self.parameters:
                    self._mtotal_from_mtotal_source_z()
            if "chirp_mass_source" not in self.parameters:
                if "chirp_mass" in self.parameters:
                    self._mchirp_source_from_mchirp_z()
            if "chirp_mass_source" in self.parameters:
                if "chirp_mass" not in self.parameters:
                    self._mchirp_from_mchirp_source_z()

        if "reference_frequency" not in self.parameters:
            self._reference_frequency()
        condition1 = "phi_12" not in self.parameters
        condition2 = "phi_1" in self.parameters and "phi_2" in self.parameters
        if condition1 and condition2:
            self._phi_12_from_phi1_phi2()

        check_for_evolved_parameter = lambda suffix, param, params: (
            param not in params and param + suffix not in params if
            len(suffix) else param not in params
        )

        if "mass_1" in self.parameters and "mass_2" in self.parameters:
            spin_components = [
                "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z",
                "iota"
            ]
            angles = ["a_1", "a_2", "tilt_1", "tilt_2", "theta_jn"]
            if all(i in self.parameters for i in spin_components):
                self._spin_angles()
            if all(i in self.parameters for i in angles):
                samples = self.specific_parameter_samples(["tilt_1", "tilt_2"])
                cond1 = all(i in [0, np.pi] for i in samples[0])
                cond2 = all(i in [0, np.pi] for i in samples[1])
                if cond1 and cond1:
                    self._non_precessing_component_spins()
                else:
                    angles = [
                        "phi_jl", "phi_12", "reference_frequency"]
                    if all(i in self.parameters for i in angles):
                        self._component_spins()
            cond1 = "spin_1x" in self.parameters and "spin_1y" in self.parameters
            if "phi_1" not in self.parameters and cond1:
                self._phi1_from_spins()
            cond1 = "spin_2x" in self.parameters and "spin_2y" in self.parameters
            if "phi_2" not in self.parameters and cond1:
                self._phi2_from_spins()
            evolve_spins_params = ["tilt_1", "tilt_2", "phi_12"]
            if self.evolve_spins_backwards:
                if all(i in self.parameters for i in evolve_spins_params):
                    self._evolve_spins(forward=False)
            for suffix in ["_infinity", "_infinity_only_prec_avg", ""]:
                if "spin_1z{}".format(suffix) not in self.parameters:
                    _params = ["a_1", "tilt_1{}".format(suffix)]
                    if all(i in self.parameters for i in _params):
                        self._aligned_spin_from_magnitude_tilts(
                            primary=True, suffix=suffix
                        )
                if "spin_2z{}".format(suffix) not in self.parameters:
                    _params = ["a_2", "tilt_2{}".format(suffix)]
                    if all(i in self.parameters for i in _params):
                        self._aligned_spin_from_magnitude_tilts(
                            secondary=True, suffix=suffix
                        )
                if "chi_eff{}".format(suffix) not in self.parameters:
                    _params = ["spin_1z{}".format(suffix), "spin_2z{}".format(suffix)]
                    if all(i in self.parameters for i in _params):
                        self._chi_eff(suffix=suffix)
                if any(
                        _p.format(suffix) not in self.parameters for _p in
                        ["chi_p{}", "chi_p_2spin"]
                ):
                    _params = [
                        "a_1", "tilt_1{}".format(suffix), "a_2",
                        "tilt_2{}".format(suffix)
                    ]
                    _cartesian_params = ["spin_1x", "spin_1y", "spin_2x", "spin_2y"]
                    if "chi_p{}".format(suffix) not in self.parameters:
                        if all(i in self.parameters for i in _params):
                            self._chi_p_from_tilts(suffix=suffix)
                        elif all(i in self.parameters for i in _cartesian_params):
                            self._chi_p()
                    if "chi_p_2spin" not in self.parameters:
                        if all(i in self.parameters for i in _cartesian_params):
                            self._chi_p_2spin()
            if "beta" not in self.parameters:
                beta_components = [
                    "mass_1", "mass_2", "phi_jl", "tilt_1", "tilt_2", "phi_12",
                    "a_1", "a_2", "reference_frequency", "phase"
                ]
                if all(i in self.parameters for i in beta_components):
                    self._beta()
            polytrope_params = ["log_pressure", "gamma_1", "gamma_2", "gamma_3"]
            if all(param in self.parameters for param in polytrope_params):
                if "lambda_1" not in self.parameters or "lambda_2" not in self.parameters:
                    self._lambda1_lambda2_from_polytrope_EOS()
            spectral_params = [
                "spectral_decomposition_gamma_{}".format(num) for num in
                np.arange(4)
            ]
            if all(param in self.parameters for param in spectral_params):
                if "lambda_1" not in self.parameters or "lambda_2" not in self.parameters:
                    self._lambda1_lambda2_from_spectral_decomposition_EOS()
            if "lambda_tilde" in self.parameters and "lambda_1" not in self.parameters:
                self._lambda1_from_lambda_tilde()
            if "lambda_2" not in self.parameters and "lambda_1" in self.parameters:
                self._lambda2_from_lambda1()
            if "lambda_1" in self.parameters and "lambda_2" in self.parameters:
                if "lambda_tilde" not in self.parameters:
                    self._lambda_tilde_from_lambda1_lambda2()
                if "delta_lambda" not in self.parameters:
                    self._delta_lambda_from_lambda1_lambda2()
            if "psi" in self.parameters:
                dpsi_parameters = ["theta_jn", "phi_jl", "beta"]
                if all(i in self.parameters for i in dpsi_parameters):
                    if "psi_J" not in self.parameters:
                        self._psi_J()

            evolve_suffix = "_non_evolved"
            final_spin_params = ["a_1", "a_2"]
            non_precessing_NR_params = ["spin_1z", "spin_2z"]
            if evolve_condition:
                final_spin_params += [
                    "tilt_1_evolved", "tilt_2_evolved", "phi_12_evolved"
                ]
                non_precessing_NR_params = [
                    "{}_evolved".format(i) for i in non_precessing_NR_params
                ]
                evolve_suffix = "_evolved"
                if all(i in self.parameters for i in evolve_spins_params):
                    try:
                        self._evolve_spins(final_velocity=evolve_spins_forwards)
                    except EvolveSpinError:
                        # Raised when approximant is unknown to lalsimulation or
                        # lalsimulation.SimInspiralGetSpinFreqFromApproximant is
                        # equal to lalsimulation.SIM_INSPIRAL_SPINS_CASEBYCASE
                        evolve_condition = False
                else:
                    evolve_condition = False
            else:
                final_spin_params += ["tilt_1", "tilt_2", "phi_12"]

            condition_peak_luminosity = check_for_evolved_parameter(
                evolve_suffix, "peak_luminosity", self.parameters
            )
            condition_final_spin = check_for_evolved_parameter(
                evolve_suffix, "final_spin", self.parameters
            )
            condition_final_mass = check_for_evolved_parameter(
                evolve_suffix, "final_mass", self.parameters
            )
            if (self.NRSurrogate or self.waveform_fit) and self.compute_remnant:
                parameters = []
                _default = ["final_mass", "final_spin"]
                if self.NRSurrogate:
                    _default.append("final_kick")
                    function = self._final_remnant_properties_from_NRSurrogate
                else:
                    final_spin_params = [
                        "spin_1x", "spin_1y", "spin_1z", "spin_2x",
                        "spin_2y", "spin_2z"
                    ]
                    function = self._final_remnant_properties_from_waveform

                for param in _default:
                    if param not in self.parameters:
                        parameters.append(param)
                # We already know that lambda_2 is in the posterior table if
                # self.NSBH = True
                if self.NSBH and "spin_1z" in self.parameters:
                    self._final_remnant_properties_from_NSBH_waveform()
                elif all(i in self.parameters for i in final_spin_params):
                    function(non_precessing=False, parameters=parameters)
                elif all(i in self.parameters for i in non_precessing_NR_params):
                    function(non_precessing=True, parameters=parameters)
                if all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_peak_luminosity or self.force_non_evolved:
                        if not self.NSBH:
                            self._peak_luminosity_of_merger(evolved=evolve_condition)
            elif self.compute_remnant:
                if all(i in self.parameters for i in final_spin_params):
                    if condition_final_spin or self.force_non_evolved:
                        self._final_spin_of_merger(evolved=evolve_condition)
                elif all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_final_spin or self.force_non_evolved:
                        self._final_spin_of_merger(
                            non_precessing=True, evolved=False
                        )
                if all(i in self.parameters for i in non_precessing_NR_params):
                    if condition_peak_luminosity or self.force_non_evolved:
                        self._peak_luminosity_of_merger(evolved=evolve_condition)
                    if condition_final_mass or self.force_non_evolved:
                        self._final_mass_of_merger(evolved=evolve_condition)

            # if NSBH system and self.compute_remnant = False and/or BBH fits
            # fits used, only calculate baryonic_torus_mass
            if self.NSBH and "spin_1z" in self.parameters:
                if "baryonic_torus_mass" not in self.parameters:
                    self._final_remnant_properties_from_NSBH_waveform(
                        parameters=["baryonic_torus_mass"]
                    )
        # calculate compactness from Love-compactness relation
        if "lambda_1" in self.parameters and "compactness_1" not in self.parameters:
            self._NS_compactness_from_lambda(parameter="lambda_1")
            if "mass_1" in self.parameters and "baryonic_mass_1" not in self.parameters:
                self._NS_baryonic_mass(primary=True)
        if "lambda_2" in self.parameters and "compactness_2" not in self.parameters:
            self._NS_compactness_from_lambda(parameter="lambda_2")
            if "mass_2" in self.parameters and "baryonic_mass_2" not in self.parameters:
                self._NS_baryonic_mass(primary=False)
        for suffix in ["_infinity", "_infinity_only_prec_avg", ""]:
            for tilt in ["tilt_1", "tilt_2"]:
                cond1 = "cos_{}{}".format(tilt, suffix) not in self.parameters
                cond2 = "{}{}".format(tilt, suffix) in self.parameters
                if cond1 and cond2:
                    self._cos_angle("cos_{}{}".format(tilt, suffix))
        evolve_suffix = "_non_evolved"
        if evolve_condition or self.NRSurrogate or self.waveform_fit or self.non_precessing:
            evolve_suffix = ""
            evolve_condition = True
        if "redshift" in self.parameters:
            condition_final_mass_source = check_for_evolved_parameter(
                evolve_suffix, "final_mass_source", self.parameters
            )
            if condition_final_mass_source or self.force_non_evolved:
                if "final_mass{}".format(evolve_suffix) in self.parameters:
                    self._final_mass_source(evolved=evolve_condition)
            if "baryonic_torus_mass" in self.parameters:
                if "baryonic_torus_mass_source" not in self.parameters:
                    self.source_frame_from_detector_frame(
                        "baryonic_torus_mass"
                    )
            if "baryonic_mass_1" in self.parameters:
                if "baryonic_mass_1_source" not in self.parameters:
                    self.source_frame_from_detector_frame(
                        "baryonic_mass_1"
                    )
            if "baryonic_mass_2" in self.parameters:
                if "baryonic_mass_2_source" not in self.parameters:
                    self.source_frame_from_detector_frame(
                        "baryonic_mass_2"
                    )
        if "total_mass_source" in self.parameters:
            if "final_mass_source{}".format(evolve_suffix) in self.parameters:
                condition_radiated_energy = check_for_evolved_parameter(
                    evolve_suffix, "radiated_energy", self.parameters
                )
                if condition_radiated_energy or self.force_non_evolved:
                    self._radiated_energy(evolved=evolve_condition)
        if self.NSBH and "spin_1z" in self.parameters:
            if all(_p in self.parameters for _p in ["mass_1_source", "mass_2_source"]):
                _NSBH_parameters = []
                if "tidal_disruption_frequency" not in self.parameters:
                    _NSBH_parameters.append("tidal_disruption_frequency")
                if "220_quasinormal_mode_frequency" not in self.parameters:
                    _NSBH_parameters.append("220_quasinormal_mode_frequency")
                if len(_NSBH_parameters):
                    self._final_remnant_properties_from_NSBH_waveform(
                        parameters=_NSBH_parameters, source=True
                    )
        location = ["geocent_time", "ra", "dec"]
        if all(i in self.parameters for i in location):
            try:
                self._time_in_each_ifo()
            except Exception as e:
                logger.warning(
                    "Failed to generate posterior samples for the time in each "
                    "detector because %s" % (e)
                )
        if any("_matched_filter_snr_angle" in i for i in self.parameters):
            if any("_matched_filter_abs_snr" in i for i in self.parameters):
                self._ifo_snr()
        if any("_optimal_snr" in i for i in self.parameters):
            if "network_optimal_snr" not in self.parameters:
                self._optimal_network_snr()
            if any("_matched_filter_snr" in i for i in self.parameters):
                if "network_matched_filter_snr" not in self.parameters:
                    self._matched_filter_network_snr()
        if "network_precessing_snr" not in self.parameters and self.precessing_snr:
            rho_p_parameters = [
                "mass_1", "mass_2", "beta", "psi_J", "a_1", "a_2", "tilt_1",
                "tilt_2", "phi_12", "theta_jn", "phi_jl", "ra", "dec", "geocent_time",
                "phi_jl"
            ]
            if all(i in self.parameters for i in rho_p_parameters):
                try:
                    logger.warn(
                        "Starting to calculate the precessing SNR. This may take "
                        "some time"
                    )
                    self._rho_p()
                except ImportError as e:
                    logger.warn(e)
            else:
                logger.warn(
                    "Unable to calculate the precessing SNR because requires "
                    "samples for {}".format(
                        ", ".join(
                            [i for i in rho_p_parameters if i not in self.parameters]
                        )
                    )
                )
        if "theta_jn" in self.parameters and "cos_theta_jn" not in self.parameters:
            self._cos_angle("cos_theta_jn")
        if "theta_jn" in self.parameters and "viewing_angle" not in self.parameters:
            self._viewing_angle()
        if "iota" in self.parameters and "cos_iota" not in self.parameters:
            self._cos_angle("cos_iota")
        remove_parameters = [
            "tilt_1_evolved", "tilt_2_evolved", "phi_12_evolved",
            "spin_1z_evolved", "spin_2z_evolved", "reference_frequency",
            "minimum_frequency"
        ]
        for param in remove_parameters:
            if param in self.parameters:
                ind = self.parameters.index(param)
                self.parameters.remove(self.parameters[ind])
                for i in self.samples:
                    del i[ind]
