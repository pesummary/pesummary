# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.gw.file.standard_names import standard_names
from pesummary.core.file.formats.base_read import (
    Read, SingleAnalysisRead, MultiAnalysisRead
)
from pesummary.utils.utils import logger
from pesummary.utils.parameters import Parameters
from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.decorators import open_config
from pesummary.gw.conversions import convert

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False


def _translate_parameters(parameters, samples):
    """Translate parameters to a standard names

    Parameters
    ----------
    parameters: list
        list of parameters used in the analysis
    samples: list
        list of samples for each parameters
    """
    path = ("https://git.ligo.org/lscsoft/pesummary/blob/master/pesummary/"
            "gw/file/standard_names.py")
    parameters_not_included = [
        i for i in parameters if i not in standard_names.keys()
    ]
    if len(parameters_not_included) > 0:
        logger.debug(
            "PESummary does not have a 'standard name' for the following "
            "parameters: {}. This means that comparison plots between "
            "different codes may not show these parameters. If you want to "
            "assign a standard name for these parameters, please add an MR "
            "which edits the following file: {}. These parameters will be "
            "added to the result pages and meta file as is.".format(
                ", ".join(parameters_not_included), path
            )
        )
    standard_params = [i for i in parameters if i in standard_names.keys()]
    converted_params = [
        standard_names[i] if i in standard_params else i for i in
        parameters
    ]
    return converted_params, samples


def _add_log_likelihood(parameters, samples):
    """Add zero log_likelihood samples to the posterior table

    Parameters
    ----------
    parameters: list
        list of parameters stored in the table
    samples: 2d list
        list of samples for each parameter. Columns correspond to a given
        parameter
    """
    if "log_likelihood" not in parameters:
        parameters.append("log_likelihood")
        samples = np.vstack(
            [np.array(samples).T, np.zeros(len(samples))]
        ).T
    return parameters, samples


def convert_injection_parameters(
    data, extra_kwargs={"sampler": {}, "meta_data": {}}, disable_convert=False,
    sampled_parameters=None
):
    """Apply the conversion module to the injection data

    Parameters
    ----------
    data: dict
        dictionary of injection data keyed by the parameter
    extra_kwargs: dict, optional
        optional kwargs to pass to the conversion module
    disable_convert: Bool, optional
        if True, do not convert injection parameters
    sampled_parameters: list, optional
        optional list of sampled parameters. If there is no injection value for
        a given sampled parameter, add a 'nan'
    """
    import math

    if disable_convert:
        return data
    if all(math.isnan(data[i]) for i in data.keys()):
        return data
    parameters = list(data.keys())
    samples = [[data[i] for i in parameters]]
    if "waveform_approximant" in parameters:
        ind = parameters.index("waveform_approximant")
        parameters.remove(parameters[ind])
        samples[0].remove(samples[0][ind])
    nan_inds = []
    for num, i in enumerate(parameters):
        if math.isnan(samples[0][num]):
            nan_inds.append(num)
    for i in nan_inds[::-1]:
        parameters.remove(parameters[i])
        samples[0].remove(samples[0][i])
    inj_samples = convert(parameters, samples, extra_kwargs=extra_kwargs)
    if sampled_parameters is not None:
        for i in sampled_parameters:
            if i not in list(inj_samples.keys()):
                inj_samples[i] = float("nan")
    return inj_samples


class GWRead(Read):
    """Base class to read in a results file

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    converted_parameters: list
        list of parameters that have been derived from the sampled distributions
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    converted_parameters: list
        list of parameters that have been added

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    to_lalinference:
        convert the posterior samples to a lalinference result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(GWRead, self).__init__(path_to_results_file, **kwargs)

    @property
    def calibration_spline_posterior(self):
        return None

    Read.attrs.update({"approximant": "approximant"})

    def load(self, function, _data=None, **kwargs):
        """Load a results file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your results file
        """
        data = _data
        if _data is None:
            data = self.load_from_function(
                function, self.path_to_results_file, **kwargs
            )
        parameters, samples = self.translate_parameters(
            data["parameters"], data["samples"]
        )
        _add_likelihood = kwargs.get("add_zero_likelihood", True)
        if not self.check_for_log_likelihood(parameters) and _add_likelihood:
            logger.warning(
                "Failed to find 'log_likelihood' in result file. Setting "
                "every sample to have log_likelihood 0"
            )
            parameters, samples = self.add_log_likelihood(parameters, samples)
        data.update(
            {
                "parameters": parameters, "samples": samples,
                "injection": data["injection"]
            }
        )
        super(GWRead, self).load(function, _data=data, **kwargs)
        if self.injection_parameters is not None:
            self.injection_parameters = self.convert_injection_parameters(
                self.injection_parameters, extra_kwargs=self.extra_kwargs,
                disable_convert=kwargs.get("disable_injection_conversion", False)
            )
        if self.priors is not None and len(self.priors):
            if self.priors["samples"] != {}:
                priors = self.priors["samples"]
                self.priors["samples"] = self.convert_and_translate_prior_samples(
                    priors, disable_convert=kwargs.get(
                        "disable_prior_conversion", False
                    )
                )

    def convert_and_translate_prior_samples(self, priors, disable_convert=False):
        """
        """
        default_parameters = list(priors.keys())
        default_samples = [
            [priors[parameter][i] for parameter in default_parameters] for i
            in range(len(priors[default_parameters[0]]))
        ]
        parameters, samples = self.translate_parameters(
            default_parameters, default_samples
        )
        if not disable_convert:
            return convert(
                parameters, samples, extra_kwargs=self.extra_kwargs
            )
        return SamplesDict(parameters, samples)

    def write(self, package="core", **kwargs):
        """Save the data to file

        Parameters
        ----------
        package: str, optional
            package you wish to use when writing the data
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        return super(GWRead, self).write(package="gw", **kwargs)

    def _grab_injection_parameters_from_file(self, injection_file, **kwargs):
        from pesummary.gw.file.injection import GWInjection

        inj_samples = GWInjection.read(injection_file, **kwargs).samples_dict
        for i in self.parameters:
            if i not in list(inj_samples.keys()):
                inj_samples[i] = float("nan")
        return inj_samples

    def _unzip_injection_file(self, injection_file):
        """Unzip the injection file and extract injection parameters from
        the file.
        """
        from pesummary.utils.utils import unzip

        out_file = unzip(injection_file)
        return self._grab_injection_parameters_from_file(out_file)

    def _grab_injection_data_from_hdf5_file(self):
        """Grab the data from an hdf5 injection file
        """
        pass

    def interpolate_calibration_spline_posterior(self, **kwargs):
        from pesummary.gw.file.calibration import Calibration
        from pesummary.utils.utils import iterator
        if self.calibration_spline_posterior is None:
            return
        total = []
        log_frequencies, amplitudes, phases = self.calibration_spline_posterior
        keys = list(log_frequencies.keys())
        _iterator = iterator(
            None, desc="Interpolating calibration posterior", logger=logger,
            tqdm=True, total=len(self.samples) * 2 * len(keys)
        )
        with _iterator as pbar:
            for key in keys:
                total.append(
                    Calibration.from_spline_posterior_samples(
                        np.array(log_frequencies[key]),
                        np.array(amplitudes[key]), np.array(phases[key]),
                        pbar=pbar, **kwargs
                    )
                )
        return total, log_frequencies.keys()

    @staticmethod
    def translate_parameters(parameters, samples):
        """Translate parameters to a standard names

        Parameters
        ----------
        parameters: list
            list of parameters used in the analysis
        samples: list
            list of samples for each parameters
        """
        return _translate_parameters(parameters, samples)

    @staticmethod
    def _check_definition_of_inclination(parameters):
        """Check the definition of inclination given the other parameters

        Parameters
        ----------
        parameters: list
            list of parameters used in the study
        """
        theta_jn = False
        spin_angles = ["tilt_1", "tilt_2", "a_1", "a_2"]
        names = [
            standard_names[i] for i in parameters if i in standard_names.keys()]
        if all(i in names for i in spin_angles):
            theta_jn = True
        if theta_jn:
            if "theta_jn" not in names and "inclination" in parameters:
                logger.warning("Because the spin angles are in your list of "
                               "parameters, the angle 'inclination' probably "
                               "refers to 'theta_jn'. If this is a mistake, "
                               "please change the definition of 'inclination' to "
                               "'iota' in your results file")
                index = parameters.index("inclination")
                parameters[index] = "theta_jn"
        else:
            if "inclination" in parameters:
                index = parameters.index("inclination")
                parameters[index] = "iota"
        return parameters

    def add_fixed_parameters_from_config_file(self, config_file):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        self._add_fixed_parameters_from_config_file(
            config_file, self._add_fixed_parameters)

    @staticmethod
    @open_config(index=2)
    def _add_fixed_parameters(parameters, samples, config_file):
        """Open a LALInference configuration file and add the fixed parameters
        to the list of parameters and samples

        Parameters
        ----------
        parameters: list
            list of existing parameters
        samples: list
            list of existing samples
        config_file: str
            path to the configuration file
        """
        from pesummary.gw.file.standard_names import standard_names

        config = config_file
        if not config.error:
            fixed_data = {}
            if "engine" in config.sections():
                fixed_data = {
                    key.split("fix-")[1]: item for key, item in
                    config.items("engine") if "fix" in key}
            for i in fixed_data.keys():
                fixed_parameter = i
                fixed_value = fixed_data[i]
                try:
                    param = standard_names[fixed_parameter]
                    if param in parameters:
                        pass
                    else:
                        parameters.append(param)
                        for num in range(len(samples)):
                            samples[num].append(float(fixed_value))
                except Exception:
                    if fixed_parameter == "logdistance":
                        if "luminosity_distance" not in parameters:
                            parameters.append(standard_names["distance"])
                            for num in range(len(samples)):
                                samples[num].append(float(fixed_value))
                    if fixed_parameter == "costheta_jn":
                        if "theta_jn" not in parameters:
                            parameters.append(standard_names["theta_jn"])
                            for num in range(len(samples)):
                                samples[num].append(float(fixed_value))
            return parameters, samples
        return parameters, samples


class GWSingleAnalysisRead(GWRead, SingleAnalysisRead):
    """Base class to read in a results file which contains a single analysis

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    converted_parameters: list
        list of parameters that have been derived from the sampled distributions
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    converted_parameters: list
        list of parameters that have been added

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    to_lalinference:
        convert the posterior samples to a lalinference result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
    """
    def __init__(self, *args, **kwargs):
        super(GWSingleAnalysisRead, self).__init__(*args, **kwargs)

    def check_for_log_likelihood(self, parameters):
        """Return True if 'log_likelihood' is in a list of sampled parameters

        Parameters
        ----------
        parameters: list
            list of sampled parameters
        """
        if "log_likelihood" in parameters:
            return True
        return False

    def add_log_likelihood(self, parameters, samples):
        """Add log_likelihood samples to a posterior table

        Parameters
        ----------
        parameters: list
            list of parameters stored in the table
        samples: 2d list
            list of samples for each parameter. Columns correspond to a given
            parameter
        """
        return _add_log_likelihood(parameters, samples)

    def generate_all_posterior_samples(self, **kwargs):
        """Generate all posterior samples via the conversion module

        Parameters
        ----------
        **kwargs: dict
            all kwargs passed to the conversion module
        """
        if "no_conversion" in kwargs.keys():
            no_conversion = kwargs.pop("no_conversion")
        else:
            no_conversion = False
        if not no_conversion:
            from pesummary.gw.conversions import convert

            data = convert(
                self.parameters, self.samples, extra_kwargs=self.extra_kwargs,
                return_dict=False, **kwargs
            )
            self.parameters = data[0]
            self.converted_parameters = self.parameters.added
            self.samples = data[1]
            if kwargs.get("return_kwargs", False):
                self.extra_kwargs = data[2]

    def convert_injection_parameters(
        self, data, extra_kwargs={"sampler": {}, "meta_data": {}},
        disable_convert=False
    ):
        """Apply the conversion module to the injection data

        Parameters
        ----------
        data: dict
            dictionary of injection data keyed by the parameter
        extra_kwargs: dict, optional
            optional kwargs to pass to the conversion module
        disable_convert: Bool, optional
            if True, do not convert injection parameters
        """
        return convert_injection_parameters(
            data, extra_kwargs=extra_kwargs, disable_convert=disable_convert,
            sampled_parameters=self.parameters
        )

    def to_lalinference(self, **kwargs):
        """Save the PESummary results file object to a lalinference hdf5 file

        Parameters
        ----------
        kwargs: dict
            all kwargs are passed to the pesummary.io.write.write function
        """
        return self.write(file_format="lalinference", package="gw", **kwargs)


class GWMultiAnalysisRead(GWRead, MultiAnalysisRead):
    """Base class to read in a results file which contains multiple analyses

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load
    """
    def __init__(self, *args, **kwargs):
        super(GWMultiAnalysisRead, self).__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        super(GWMultiAnalysisRead, self).load(*args, **kwargs)
        if "psd" in self.data.keys():
            from pesummary.gw.file.psd import PSDDict

            try:
                self.psd = {
                    label: PSDDict(
                        {ifo: value for ifo, value in psd_data.items()}
                    ) for label, psd_data in self.data["psd"].items()
                }
            except (KeyError, AttributeError):
                self.psd = self.data["psd"]
        if "calibration" in self.data.keys():
            from pesummary.gw.file.calibration import Calibration

            try:
                self.calibration = {
                    label: {
                        ifo: Calibration(value) for ifo, value in
                        calibration_data.items()
                    } for label, calibration_data in
                    self.data["calibration"].items()
                }
            except (KeyError, AttributeError):
                self.calibration = self.data["calibration"]
        if "prior" in self.data.keys() and "calibration" in self.data["prior"].keys():
            from pesummary.gw.file.calibration import CalibrationDict

            try:
                self.priors["calibration"] = {
                    label: CalibrationDict(calibration_data) for
                    label, calibration_data in
                    self.data["prior"]["calibration"].items()
                }
            except (KeyError, AttributeError):
                pass
        if "skymap" in self.data.keys():
            from pesummary.gw.file.skymap import SkyMapDict, SkyMap

            try:
                self.skymap = SkyMapDict({
                    label: SkyMap(skymap["data"], skymap["meta_data"])
                    for label, skymap in self.data["skymap"].items()
                })
            except (KeyError, AttributeError):
                self.skymap = self.data["skymap"]
        if "gwdata" in self.data.keys():
            try:
                from pesummary.gw.file.strain import StrainDataDict, StrainData
                from pesummary.utils.dict import Dict
                mydict = {}
                for IFO, value in self.data["gwdata"].items():
                    channel = [ch for ch in value.keys() if "_attrs" not in ch][0]
                    if "{}_attrs".format(channel) in value.keys():
                        _attrs = value["{}_attrs".format(channel)]
                    else:
                        _attrs = {}
                    mydict[IFO] = StrainData(value[channel], **_attrs)
                self.gwdata = StrainDataDict(mydict)
            except (KeyError, AttributeError):
                pass

    def convert_and_translate_prior_samples(self, priors, disable_convert=False):
        """
        """
        from pesummary.utils.samples_dict import MultiAnalysisSamplesDict

        mydict = {}
        for num, label in enumerate(self.labels):
            if label in priors.keys() and len(priors[label]):
                default_parameters = list(priors[label].keys())
                default_samples = np.array(
                    [priors[label][_param] for _param in default_parameters]
                ).T
                parameters, samples = self.translate_parameters(
                    [default_parameters], [default_samples]
                )
                if not disable_convert:
                    mydict[label] = convert(
                        parameters[0], samples[0], extra_kwargs=self.extra_kwargs[num]
                    )
                else:
                    mydict[label] = SamplesDict(parameters[0], samples[0])
            else:
                mydict[label] = {}
        return MultiAnalysisSamplesDict(mydict)

    def check_for_log_likelihood(self, parameters):
        if all("log_likelihood" in p for p in parameters):
            return True
        return False

    @staticmethod
    def translate_parameters(parameters, samples):
        """Translate parameters to a standard names

        Parameters
        ----------
        parameters: list
            list of parameters used in the analysis
        samples: list
            list of samples for each parameters
        """
        converted_params = []
        for _parameters, _samples in zip(parameters, samples):
            converted_params.append(
                _translate_parameters(_parameters, _samples)[0]
            )
        return converted_params, samples

    def add_log_likelihood(self, parameters, samples):
        """
        """
        parameters_logl, samples_logl = [], []
        for _parameters, _samples in zip(parameters, samples):
            pp, ss = _add_log_likelihood(_parameters, _samples)
            parameters_logl.append(pp)
            samples_logl.append(ss)
        return parameters_logl, samples_logl

    def generate_all_posterior_samples(self, labels=None, **conversion_kwargs):
        if "no_conversion" in conversion_kwargs.keys():
            no_conversion = conversion_kwargs.pop("no_conversion")
        else:
            no_conversion = False
        if no_conversion:
            return
        from pesummary.gw.conversions import convert

        converted_params, converted_samples, converted_kwargs = [], [], []
        _converted_params = []
        for label, param, samples, kwargs in zip(
                self.labels, self.parameters, self.samples, self.extra_kwargs
        ):
            if labels is not None and label not in labels:
                converted_params.append(param)
                _converted_params.append([])
                converted_samples.append(samples)
                if kwargs.get("return_kwargs", False):
                    converted_kwargs.append(kwargs)
                continue
            if label in conversion_kwargs.keys():
                _conversion_kwargs = conversion_kwargs[label]
            else:
                _conversion_kwargs = conversion_kwargs
            if _conversion_kwargs.get("evolve_spins", False):
                if not _conversion_kwargs.get("return_kwargs", False):
                    _conversion_kwargs["return_kwargs"] = True
            data = convert(
                param, samples, extra_kwargs=kwargs, return_dict=False,
                **_conversion_kwargs
            )
            converted_params.append(data[0])
            _converted_params.append(data[0].added)
            converted_samples.append(data[1])
            if kwargs.get("return_kwargs", False):
                converted_kwargs.append(data[2])
        self.parameters = converted_params
        self.converted_parameters = _converted_params
        self.samples = converted_samples
        if converted_kwargs != []:
            self.extra_kwargs = {
                label: converted_kwargs[num] for num, label in enumerate(
                    self.labels
                )
            }

    def convert_injection_parameters(
        self, data, extra_kwargs={"sampler": {}, "meta_data": {}},
        disable_convert=False
    ):
        """Apply the conversion module to the injection data
        """
        for num, label in enumerate(self.labels):
            _identifier = label
            if isinstance(data, dict):
                _data = data[label]
            else:
                _data = data[num]
                _identifier = num
            data[_identifier] = convert_injection_parameters(
                _data, extra_kwargs=extra_kwargs[num],
                disable_convert=disable_convert,
                sampled_parameters=self.parameters[num]
            )
        return data
