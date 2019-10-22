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

import os

import h5py
import numpy as np
try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False

from pesummary.gw.file.formats.base_read import GWRead
from pesummary.gw.file import conversions as con
from pesummary.utils.utils import logger


class LALInference(GWRead):
    """PESummary wrapper of `lalinference`
    (https://git.ligo.org/lscsoft/lalsuite/lalinference).

    Attributes
    ----------
    path_to_results_file: str
        path to the results file you wish to load in with `LALInference`
    """
    def __init__(self, path_to_results_file, injection_file=None):
        super(LALInference, self).__init__(path_to_results_file)
        self.load(self._grab_data_from_lalinference_file)

    @classmethod
    def load_file(cls, path, injection_file=None):
        if not os.path.isfile(path):
            raise IOError("%s does not exist" % (path))
        if injection_file and not os.path.isfile(injection_file):
            raise IOError("%s does not exist" % (path))
        return cls(path, injection_file=injection_file)

    @staticmethod
    def guess_path_to_sampler(path):
        """Guess the path to the sampler group in a LALInference results file

        Parameters
        ----------
        path: str
            path to the LALInference results file
        """
        def _find_name(name):
            c1 = "lalinference_nest" in name or "lalinference_mcmc" in name
            c2 = "lalinference_nest/" not in name and "lalinference_mcmc/" not in name
            if c1 and c2:
                return name

        f = h5py.File(path, 'r')
        _path = f.visit(_find_name)
        f.close()
        return _path

    @staticmethod
    def _parameters_in_lalinference_file(path):
        """Return the parameter names stored in the LALInference results file

        Parameters
        ----------
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        parameters = list(f[path_to_samples].dtype.names)
        f.close()
        return parameters

    @staticmethod
    def _samples_in_lalinference_file(path):
        """
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        samples = [list(i) for i in f[path_to_samples]]
        return samples

    @staticmethod
    def _check_for_calibration_data_in_lalinference_file(path):
        """
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        lalinference_names = list(f[path_to_samples].dtype.names)
        if any("_spcal_amp" in i for i in lalinference_names):
            return True
        return False

    @property
    def calibration_data_in_results_file(self):
        """
        """
        check = LALInference._check_for_calibration_data_in_lalinference_file
        grab = LALInference._grab_calibration_data_from_lalinference_file
        if self.check_for_calibration_data(check, self.path_to_results_file):
            return self.grab_calibration_data(grab, self.path_to_results_file)
        return None

    @staticmethod
    def grab_extra_kwargs(path):
        """Grab any additional information stored in the lalinference file
        """
        kwargs = {"sampler": {}, "meta_data": {}}
        path_to_samples = GWRead.guess_path_to_samples(path)
        path_to_sampler = LALInference.guess_path_to_sampler(path)
        f = h5py.File(path, 'r')
        attributes = dict(f[path_to_samples].attrs.items())
        if "flow" in list(attributes.keys()):
            kwargs["sampler"]["f_low"] = attributes["flow"]
        elif "f_low" in list(attributes.keys()):
            kwargs["sampler"]["f_low"] = attributes["f_low"]
        if "fref" in list(attributes.keys()):
            kwargs["sampler"]["f_ref"] = attributes["fref"]
        elif "f_ref" in list(attributes.keys()):
            kwargs["sampler"]["f_ref"] = attributes["f_ref"]
        if "LAL_APPROXIMANT" in list(attributes.keys()):
            try:
                from lalsimulation import GetStringFromApproximant

                kwargs["meta_data"]["approximant"] = \
                    GetStringFromApproximant(int(attributes["LAL_APPROXIMANT"]))
            except Exception:
                kwargs["meta_data"]["approximant"] = \
                    int(attributes["LAL_APPROXIMANT"])
        if "number_of_live_points" in list(attributes.keys()):
            kwargs["meta_data"]["number_of_live_points"] = attributes["number_of_live_points"]
        if "segmentLength" in list(attributes.keys()):
            kwargs["meta_data"]["seglen"] = attributes["segmentLength"]
        if "sampleRate" in list(attributes.keys()):
            kwargs["meta_data"]["samplerate"] = attributes["sampleRate"]

        attributes = dict(f[path_to_sampler].attrs.items())
        if "log_bayes_factor" in list(attributes.keys()):
            kwargs["sampler"]["log_bayes_factor"] = np.round(
                attributes["log_bayes_factor"], 2)
        elif "bayes_factor" in list(attributes.keys()):
            kwargs["sampler"]["log_bayes_factor"] = np.round(
                np.log(attributes["log_bayes_factor"]), 2)
        if "log_evidence" in list(attributes.keys()):
            kwargs["sampler"]["log_evidence"] = np.round(
                attributes["log_evidence"], 2)
        elif "evidence" in list(attributes.keys()):
            kwargs["sampler"]["log_evidence"] = np.round(
                np.log(attributes["evidence"]), 2)
        if "log_prior_volume" in list(attributes.keys()):
            kwargs["sampler"]["log_prior_volume"] = np.round(
                attributes["log_prior_volume"], 2)
        elif "prior_volume" in list(attributes.keys()):
            kwargs["sampler"]["log_prior_volume"] = np.round(
                np.log(attributes["prior_volume"]), 2)
        if "number_of_live_points" in list(attributes.keys()):
            kwargs["meta_data"]["number_of_live_points"] = attributes["number_of_live_points"]
        if "segmentLength" in list(attributes.keys()):
            kwargs["meta_data"]["seglen"] = attributes["segmentLength"]
        if "sampleRate" in list(attributes.keys()):
            kwargs["meta_data"]["samplerate"] = attributes["sampleRate"]
        f.close()
        return kwargs

    @staticmethod
    def _grab_calibration_data_from_lalinference_file(path):
        """
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        attributes = f[path_to_samples].attrs.items()
        lalinference_names = list(f[path_to_samples].dtype.names)
        samples = [list(i) for i in f[path_to_samples]]
        keys_amp = np.sort([
            param for param in lalinference_names if "_spcal_amp" in param])
        keys_phase = np.sort([
            param for param in lalinference_names if "_spcal_phase" in
            param])
        log_frequencies = {
            key.split("_")[0]: [] for key, value in attributes if
            "_spcal_logfreq" in key or "_spcal_freq" in key}
        for key, value in attributes:
            if "_spcal_logfreq" in key:
                log_frequencies[key.split("_")[0]].append(float(value))
            if "_spcal_freq" in key:
                log_frequencies[key.split("_")[0]].append(np.log(float(value)))

        amp_params = {ifo: [] for ifo in log_frequencies.keys()}
        phase_params = {ifo: [] for ifo in log_frequencies.keys()}
        for key in keys_amp:
            ifo = key.split("_")[0]
            ind = lalinference_names.index(key)
            amp_params[ifo].append([float(i[ind]) for i in samples])
        for key in keys_phase:
            ifo = key.split("_")[0]
            ind = lalinference_names.index(key)
            phase_params[ifo].append([float(i[ind]) for i in samples])
        f.close()
        return log_frequencies, amp_params, phase_params

    @staticmethod
    def _grab_data_from_lalinference_file(path):
        """
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        lalinference_names = list(f[path_to_samples].dtype.names)
        samples = [list(i) for i in f[path_to_samples]]

        if "logdistance" in lalinference_names:
            lalinference_names.append("luminosity_distance")
            for num, i in enumerate(samples):
                samples[num].append(
                    np.exp(i[lalinference_names.index("logdistance")]))
        if "costheta_jn" in lalinference_names:
            lalinference_names.append("theta_jn")
            for num, i in enumerate(samples):
                samples[num].append(
                    np.arccos(i[lalinference_names.index("costheta_jn")]))
        extra_kwargs = LALInference.grab_extra_kwargs(path)
        extra_kwargs["sampler"]["nsamples"] = len(samples)
        try:
            version = f[path_to_samples].attrs["VERSION"].decode("utf-8")
        except Exception as e:
            version = None
        return {
            "parameters": lalinference_names,
            "samples": samples,
            "injection": None,
            "version": version,
            "kwargs": extra_kwargs
        }

    def add_injection_parameters_from_file(self, injection_file):
        """
        """
        self.injection_parameters = self._add_injection_parameters_from_file(
            injection_file, self._grab_injection_parameters_from_file)

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

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        self._add_marginalized_parameters_from_config_file(
            config_file, self._add_marginalized_parameters)

    @staticmethod
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
        import configparser
        from pesummary.gw.file.standard_names import standard_names

        config = configparser.ConfigParser()
        try:
            config.read(config_file)
            fixed_data = None
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
        except Exception:
            return parameters, samples

    @staticmethod
    def _add_marginalized_parameters(parameters, samples, config_file):
        """Open a LALInference configuration file and add the marginalized
        parameters to the list of parameters and samples

        Parameters
        ----------
        parameters: list
            list of existing parameters
        samples: list
            list of existing samples
        config_file: str
            path to the configuration file
        """
        import configparser
        from pesummary.gw.file.standard_names import standard_names

        config = configparser.ConfigParser()
        try:
            config.read(config_file)
            fixed_data = None
            if "engine" in config.sections():
                marg_par = {
                    key.split("marg")[1]: item for key, item in
                    config.items("engine") if "marg" in key}
            for i in marg_par.keys():
                if "time" in i and "geocent_time" not in parameters:
                    if "marginalized_geocent_time" in parameters:
                        ind = parameters.index("marginalized_geocent_time")
                        parameters.remove(parameters[ind])
                        parameters.append("geocent_time")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warn("You have marginalized over time and "
                                    "there are no time samples. Manually "
                                    "setting time to 100000s")
                        parameters.append("geocent_time")
                        for num, j in enumerate(samples):
                            samples[num].append(float(100000))
                if "phi" in i and "phase" not in parameters:
                    if "marginalized_phase" in parameters:
                        ind = parameters.index("marginalized_phase")
                        parameters.remove(parameters[ind])
                        parameters.append("phase")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warn("You have marginalized over phase and "
                                    "there are no phase samples. Manually "
                                    "setting the phase to be 0")
                        parameters.append("phase")
                        for num, j in enumerate(samples):
                            samples[num].append(float(0))
                if "dist" in i and "luminosity_distance" not in parameters:
                    if "marginalized_distance" in parameters:
                        ind = parameters.index("marginalized_distance")
                        parameters.remove(parameters[ind])
                        parameters.append("luminosity_distance")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warn("You have marginalized over distance and "
                                    "there are no distance samples. Manually "
                                    "setting distance to 100Mpc")
                        parameters.append("luminosity_distance")
                        for num, j in enumerate(samples):
                            samples[num].append(float(100.0))
            return parameters, samples
        except Exception:
            return parameters, samples
