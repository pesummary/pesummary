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
    def _parameters_in_lalinference_file(path):
        """Return the parameter names stored in the LALInference results file

        Parameters
        ----------
        """
        f = h5py.File(path)
        path_to_samples = GWRead.guess_path_to_samples(path)
        parameters = list(f[path_to_samples].dtype.names)
        f.close()
        return parameters

    @staticmethod
    def _samples_in_lalinference_file(path):
        """
        """
        f = h5py.File(path)
        path_to_samples = GWRead.guess_path_to_samples(path)
        samples = [list(i) for i in f[path_to_samples]]
        return samples

    @staticmethod
    def _check_for_calibration_data_in_lalinference_file(path):
        """
        """
        f = h5py.File(path)
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
    def _grab_calibration_data_from_lalinference_file(path):
        """
        """
        f = h5py.File(path)
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
            "_spcal_logfreq" in key}
        for key, value in attributes:
            if "_spcal_logfreq" in key:
                log_frequencies[key.split("_")[0]].append(float(value))

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
        f = h5py.File(path)
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
        spin_magnitudes = ["a_1", "a_2"]
        spin_angles = ["phi_jl", "tilt_1", "tilt_2", "phi_12"]
        if all(i in lalinference_names for i in spin_magnitudes):
            if all(i not in lalinference_names for i in spin_angles):
                lalinference_names.append("tilt_1")
                lalinference_names.append("tilt_2")
                for num, i in enumerate(samples):
                    samples[num].append(
                        np.arccos(np.sign(i[lalinference_names.index("a1")])))
                    samples[num].append(
                        np.arccos(np.sign(i[lalinference_names.index("a2")])))
                ind_a1 = lalinference_names.index("a_1")
                ind_a2 = lalinference_names.index("a_2")
                for num, i in enumerate(samples):
                    samples[num][ind_a1] = abs(samples[num][ind_a1])
                    samples[num][ind_a2] = abs(samples[num][ind_a2])
        return lalinference_names, samples, None

    def add_injection_parameters_from_file(self, injection_file):
        """
        """
        self.injection_parameters = GWRead._add_injection_parameters_from_file(
            self._grab_injection_parameters_from_file)

    def _grab_injection_parameters_from_file(self, injection_file):
        extension = injection_file.split(".")[-1]
        func_map = {"xml": self._grab_injection_data_from_xml_file,
                    "hdf5": self._grab_injection_data_from_hdf5_file,
                    "h5": self._grab_injection_data_from_hdf5_file}
        return func_map[extension](injection_file)

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
        """Return the full list of injection parameters

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
            "mass_ratio": lambda inj: con.q_from_m1_m2(
                inj.mass1, inj.mass2),
            "symmetric_mass_ratio": lambda inj: con.eta_from_m1_m2(
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