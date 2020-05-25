# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy as np
from scipy.interpolate import interp1d
from pesummary.gw.file.standard_names import standard_names
from pesummary.core.file.formats.base_read import Read
from pesummary.utils.utils import logger
from pesummary.utils.decorators import open_config
from pesummary.gw.file import conversions as con

try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False


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
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file

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
    def __init__(self, path_to_results_file):
        super(GWRead, self).__init__(path_to_results_file)

    def load(self, function, **kwargs):
        """Load a results file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your results file
        """
        data = self.load_from_function(
            function, self.path_to_results_file, **kwargs)
        if "mcmc_samples" in data.keys():
            self.mcmc_samples = data["mcmc_samples"]
        parameters, samples = self.translate_parameters(
            data["parameters"], data["samples"]
        )
        if "log_likelihood" not in parameters:
            logger.warn(
                "Failed to find 'log_likelihood' in result file. Setting "
                "every sample to have log_likelihood 0"
            )
            parameters.append("log_likelihood")
            for num, i in enumerate(samples):
                samples[num].append(0)
        self.data = {
            "parameters": parameters, "samples": samples
        }
        self.parameters = self.data["parameters"]
        self.samples = self.data["samples"]
        self.data["injection"] = data["injection"]
        if "version" in data.keys() and data["version"] is not None:
            self.input_version = data["version"]
        else:
            self.input_version = "No version information found"
        if "kwargs" in data.keys():
            self.extra_kwargs = data["kwargs"]
        else:
            self.extra_kwargs = {"sampler": {}, "meta_data": {}}
            self.extra_kwargs["sampler"]["nsamples"] = len(self.data["samples"])
        if data["injection"] is not None:
            self.injection_parameters = self.convert_injection_parameters(
                data["injection"]
            )
        else:
            self.injection_parameters = data["injection"]
        if isinstance(self.injection_parameters, dict):
            self.injection_parameters = {
                key.decode("utf-8") if isinstance(key, bytes) else key: val
                for key, val in self.injection_parameters.items()
            }
        elif isinstance(self.injection_parameters, list):
            self.injection_parameters = [
                {
                    key.decode("utf-8") if isinstance(key, bytes) else
                    key: val for key, val in i.items()
                } for i in self.injection_parameters
            ]
        if "prior" in data.keys() and data["prior"] != {}:
            priors = data["prior"]
            default_parameters = list(priors.keys())
            default_samples = [
                [priors[parameter][i] for parameter in default_parameters] for i
                in range(len(priors[default_parameters[0]]))
            ]
            parameters, samples = self.translate_parameters(
                default_parameters, default_samples
            )
            self.priors = con._Conversion(
                parameters, samples, extra_kwargs=self.extra_kwargs
            )
        if "weights" in self.data.keys():
            self.weights = self.data["weights"]
        else:
            self.weights = self.check_for_weights(
                self.data["parameters"], self.data["samples"]
            )

    def convert_injection_parameters(self, data):
        """
        """
        import math

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
        inj_samples = con._Conversion(
            parameters, samples, extra_kwargs=self.extra_kwargs
        )
        for i in self.parameters:
            if i not in list(inj_samples.keys()):
                inj_samples[i] = float("nan")
        return inj_samples

    def _grab_injection_parameters_from_file(self, injection_file):
        extension = injection_file.split(".")[-1]
        func_map = {"xml": self._grab_injection_data_from_xml_file,
                    "hdf5": self._grab_injection_data_from_hdf5_file,
                    "h5": self._grab_injection_data_from_hdf5_file,
                    "gz": self._unzip_injection_file}
        data = func_map[extension](injection_file)
        return self.convert_injection_parameters(data)

    def _unzip_injection_file(self, injection_file):
        """Unzip the injection file and extract injection parameters from
        the file.
        """
        from pesummary.utils.utils import unzip

        out_file = unzip(injection_file)
        return self._grab_injection_parameters_from_file(out_file)

    def _grab_injection_data_from_xml_file(self, injection_file):
        """Grab the data from an xml injection file
        """
        if GLUE:
            xmldoc = ligolw_utils.load_filename(
                injection_file, contenthandler=lsctables.use_in(
                    ligolw.LIGOLWContentHandler))
            try:
                table = lsctables.SimInspiralTable.get_table(xmldoc)[0]
            except Exception:
                table = lsctables.SnglInspiralTable.get_table(xmldoc)[0]
            injection_values = self._return_all_injection_parameters(
                self.parameters, table)
        else:
            injection_values = [float("nan")] * len(self.parameters)
        return {i: j for i, j in zip(self.parameters, injection_values)}

    def _return_all_injection_parameters(self, parameters, table):
        """Return tlhe full list of injection parameters

        Parameters
        ----------
        parameters: list
            full list of parameters being used in the analysis
        table: glue.ligolw.lsctables.SnglInspiral
            table containing the trigger values
        """
        func_map = {
            "chirp_mass": lambda inj: inj.mchirp,
            "luminosity_distance": lambda inj: inj.distance,
            "mass_1": lambda inj: inj.mass1,
            "mass_2": lambda inj: inj.mass2,
            "dec": lambda inj: inj.latitude,
            "spin_1x": lambda inj: inj.spin1x,
            "spin_1y": lambda inj: inj.spin1y,
            "spin_1z": lambda inj: inj.spin1z,
            "spin_2x": lambda inj: inj.spin2x,
            "spin_2y": lambda inj: inj.spin2y,
            "spin_2z": lambda inj: inj.spin2z,
            "iota": lambda inj: inj.inclination,
            "psi": lambda inj: inj.polarization,
            "mass_ratio": lambda inj: con.q_from_m1_m2(
                inj.mass1, inj.mass2),
            "symmetric_mass_ratio": lambda inj: con.eta_from_m1_m2(
                inj.mass1, inj.mass2),
            "inversed_mass_ratio": lambda inj: con.invq_from_m1_m2(
                inj.mass1, inj.mass2),
            "total_mass": lambda inj: inj.mass1 + inj.mass2,
            "chi_p": lambda inj: con._chi_p(
                inj.mass1, inj.mass2, inj.spin1x, inj.spin1y, inj.spin2x,
                inj.spin2y),
            "chi_eff": lambda inj: con._chi_eff(
                inj.mass1, inj.mass2, inj.spin1z, inj.spin2z)}

        injection_values = []
        for i in parameters:
            try:
                if func_map[i](table) is not None:
                    injection_values.append(func_map[i](table))
                else:
                    injection_values.append(float("nan"))
            except Exception:
                injection_values.append(float("nan"))
        return injection_values

    def _grab_injection_data_from_hdf5_file(self):
        """Grab the data from an hdf5 injection file
        """
        pass

    @staticmethod
    def check_for_calibration_data(function, path_to_results_file):
        """Check to see if there is any calibration data in the results file

        Parameters
        ----------
        function: func
            callable function that will check to see if calibration data is in
            the results file
        path_to_results_file: str
            path to the results file
        """
        return function(path_to_results_file)

    @staticmethod
    def grab_calibration_data(function, path_to_results_file):
        """Grab the calibration data from the results file

        Parameters
        ----------
        function: func
            callable function that will grab the calibration data from the
            results file
        path_to_results_file: str
            path to the results file
        """
        log_frequencies, amp_params, phase_params = function(path_to_results_file)
        total = []
        for key in log_frequencies.keys():
            f = np.exp(log_frequencies[key])
            fs = np.linspace(np.min(f), np.max(f), 100)
            data = [interp1d(log_frequencies[key], samp, kind="cubic",
                             fill_value=0, bounds_error=False)(np.log(fs)) for samp
                    in np.column_stack(amp_params[key])]
            amplitude_upper = 1. - np.mean(data, axis=0) + np.std(data, axis=0)
            amplitude_lower = 1. - np.mean(data, axis=0) - np.std(data, axis=0)
            amplitude_median = 1 - np.median(data, axis=0)

            data = [interp1d(log_frequencies[key], samp, kind="cubic",
                             fill_value=0, bounds_error=False)(np.log(fs)) for samp
                    in np.column_stack(phase_params[key])]

            phase_upper = np.mean(data, axis=0) + np.std(data, axis=0)
            phase_lower = np.mean(data, axis=0) - np.std(data, axis=0)
            phase_median = np.median(data, axis=0)
            total.append(np.column_stack(
                [fs, amplitude_median, phase_median, amplitude_lower,
                 phase_lower, amplitude_upper, phase_upper]))
        return total, log_frequencies.keys()

    @staticmethod
    def load_strain_data(strain_data):
        """Load the strain data

        Parameters
        ----------
        strain_data: dict
            strain data with key equal to the channel and value the path to
            the strain data
        """
        func_map = {"lcf": GWRead._timeseries_from_cache_file,
                    "pickle": GWRead._timeseries_from_pickle_file}
        timeseries = {}
        if isinstance(strain_data, str):
            ext = GWRead.extension_from_path(strain_data)
            function = func_map[ext]
            try:
                timeseries = GWRead.load_from_function(function, strain_data)
            except Exception as e:
                logger.info("Failed to load in {} because {}".format(strain_data, e))
                timeseries = None
            return timeseries

        for key in list(strain_data.keys()):
            ext = GWRead.extension_from_path(strain_data[key])
            function = func_map[ext]
            reduced_dict = {key: strain_data[key]}
            if "H1" in key:
                ifo = "H1"
            elif "L1" in key:
                ifo = "L1"
            elif "V1" in key:
                ifo = "V1"
            else:
                ifo = key
            try:
                timeseries[ifo] = GWRead.load_from_function(function, reduced_dict)
            except Exception as e:
                logger.info("Failed to load {} because {}".format(strain_data[key], e))
            if timeseries == {}:
                timeseries = None
        return timeseries

    @staticmethod
    def _timeseries_from_cache_file(strain_dictionary):
        """Return a time series from a cache file

        Parameters
        ----------
        strain_dictionary: dict
            dictionary containing one key (channel) and one value (path to
            cache file)
        """
        from gwpy.timeseries import TimeSeries

        try:
            from glue.lal import Cache
            GLUE = True
        except ImportError:
            GLUE = False

        if not GLUE:
            raise Exception("lscsoft-glue is required to read from a cached "
                            "file. Please install this package")
        channel = list(strain_dictionary.keys())[0]
        cached_file = strain_dictionary[channel]
        with open(cached_file, "r") as f:
            data = Cache.fromfile(f)
        try:
            strain_data = TimeSeries.read(data, channel)
        except Exception as e:
            raise Exception("Failed to read in the cached file because %s" % (
                            e))
        return strain_data

    @staticmethod
    def _timeseries_from_pickle_file(pickle_file):
        """Return a time series from a pickle file
        """
        try:
            from pesummary.gw.file.formats.bilby import Bilby

            data = Bilby._timeseries_from_bilby_pickle(pickle_file)
        except Exception as e:
            raise Exception("Failed to read pickle file because %s" % (e))
        return data

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
                logger.warn("Because the spin angles are in your list of "
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

    def _specific_parameter_samples(self, param):
        """Return the samples for a specific parameter

        Parameters
        ----------
        param: str
            the parameter that you would like to return the samples for
        """
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

    def append_data(self, samples):
        """Add a list of samples to the existing samples data object

        Parameters
        ----------
        samples: list
            the list of samples that you would like to append
        """
        for num, i in enumerate(self.samples):
            self.samples[num].append(samples[num])

    def generate_all_posterior_samples(self, **kwargs):
        if "no_conversion" in kwargs.keys():
            no_conversion = kwargs.pop("no_conversion")
        else:
            no_conversion = False
        if not no_conversion:
            from pesummary.gw.file.conversions import _Conversion

            data = _Conversion(
                self.parameters, self.samples, extra_kwargs=self.extra_kwargs,
                return_dict=False, **kwargs
            )
            self.parameters = data[0]
            self.samples = data[1]
            if kwargs.get("return_kwargs", False):
                self.extra_kwargs = data[2]

    def to_lalinference(self, **kwargs):
        """Save the PESummary results file object to a lalinference hdf5 file

        Parameters
        ----------
        kwargs: dict
            all kwargs are passed to the pesummary.io.write.write function
        """
        return self.write(file_format="lalinference", package="gw", **kwargs)
