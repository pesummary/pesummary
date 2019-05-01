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

import configparser
import shutil

import json
import h5py
import deepdish

import numpy as np

from pesummary.command_line import command_line
import pesummary.file.conversions as con
from pesummary.utils.utils import logger
from pesummary.file.lalinference import LALInferenceResultsFile
from pesummary.file.standard_names import standard_names

try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False


def paths_to_key(key, dictionary, current_path=None):
    """Return the path to a key stored in a nested dictionary

    Parameters
    ----------
    key: str
        the key that you would like to find
    dictionary: dict
        the nested dictionary that has the key stored somewhere within it
    current_path: str, optional
        the current level in the dictionary
    """
    if current_path is None:
        current_path = []

    for k, v in dictionary.items():
        if k == key:
            yield current_path + [key]
        else:
            if isinstance(v, dict):
                path = current_path + [k]
                for z in paths_to_key(key, v, path):
                    yield z


def load_recusively(key, dictionary):
    """Return the entry for a key of format 'a/b/c/d'

    Parameters
    ----------
    key: str
        key of format 'a/b/c/d'
    dictionary: dict
        the dictionary that has the key stored
    """
    if "/" in key:
        key = key.split("/")
    if isinstance(key, str):
        key = [key]
    if key[-1] in dictionary.keys():
        yield dictionary[key[-1]]
    else:
        old, new = key[0], key[1:]
        for z in load_recusively(new, dictionary[old]):
            yield z


class OneFormat(object):
    """Class to convert a given results file into a standard format with all
    derived posterior distributions included

    Parameters
    ----------
    fil: str
       path to the results file
    inj: str, optional
       path to the file containing injection information
    config: str, optional
       path to the configuration file

    Attributes
    ----------
    extension: str
        the extension of the input file
    lalinference_hdf5_format: Bool
        Boolean determining if the hdf5 file is of LALInference format
    bilby_hdf5_format: Bool
        Boolean determining if the hdf5 file is of Bilby format
    data: list
        list containing the extracted data from the input file
    parameters: list
        list of parameters stored in the input file
    samples: list
        list of samples stored in the input file
    approximant: str
        the approximant stored in the input file
    """
    def __init__(self, fil, inj=None, config=None):
        logger.info("Extracting the information from %s" % (fil))
        self.fil = fil
        self.inj = inj
        self.config = config
        self.data = None
        self.injection_data = None

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

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = None
        self.fixed_data = None
        self.marg_par = None
        if config:
            data = self._extract_data_from_config_file(config)
            self.fixed_data = data["fixed_data"]
            self.marg_par = data["marginalized_parameters"]
            self._config = config

    @property
    def extension(self):
        return self.fil.split(".")[-1]

    @property
    def lalinference_hdf5_format(self):
        f = h5py.File(self.fil)
        keys = list(f.keys())
        f.close()
        if "lalinference" in keys:
            return True
        return False

    @property
    def bilby_hdf5_format(self):
        f = h5py.File(self.fil)
        keys = list(f.keys())
        f.close()
        if "data" in keys or "posterior" in keys:
            return True
        return False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        func_map = {"json": self._grab_data_from_json_file,
                    "hdf5": self._grab_data_from_hdf5_file,
                    "h5": self._grab_data_from_hdf5_file,
                    "dat": self._grab_data_from_dat_file,
                    "txt": self._grab_data_from_dat_file}
        data = func_map[self.extension]()
        parameters = data[0]
        samples = data[1]

        if self.fixed_data:

            for i in self.fixed_data.keys():
                fixed_parameter = i
                fixed_value = self.fixed_data[i]

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

        if self.marg_par:
            for i in self.marg_par.keys():
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

        if len(data) > 2:
            self._data = [parameters, samples, data[2]]
        else:
            self._data = [parameters, samples]

    @property
    def parameters(self):
        return self.data[0]

    @property
    def samples(self):
        return self.data[1]

    @property
    def approximant(self):
        if len(self.data) > 2:
            return self.data[2]
        return "none"

    @property
    def injection_data(self):
        return self._injection_data

    @injection_data.setter
    def injection_data(self, injection_data):
        if self.inj:
            extension = self.inj.split(".")[-1]
            func_map = {"xml": self._grab_injection_data_from_xml_file,
                        "hdf5": self._grab_injection_data_from_hdf5_file,
                        "h5": self._grab_injection_data_from_hdf5_file}
            self._injection_data = func_map[extension]()
        else:
            self._injection_data = [
                self.parameters, [float("nan")] * len(self.parameters)]

    @property
    def injection_parameters(self):
        return self.injection_data[0]

    @property
    def injection_values(self):
        return self.injection_data[1]

    def _grab_data_from_hdf5_file(self):
        """Grab the data stored in an hdf5 file
        """
        self._data_structure = []
        if self.lalinference_hdf5_format:
            return self._grab_data_with_h5py()
        elif self.bilby_hdf5_format:
            try:
                return self._grab_data_with_deepdish()
            except Exception as e:
                logger.warning("Failed to open %s with deepdish because %s. "
                               "Trying to grab the data with 'h5py'." % (
                                   self.fil, e))
                return self._grab_data_with_h5py()
        else:
            logger.warning("Unrecognised HDF5 format. Trying to open and find "
                           "the data")
            try:
                return self._grab_data_with_h5py()
            except Exception:
                raise Exception("Failed to extract the data from the results "
                                "file. Please reformat to either the bilby "
                                "or LALInference format")

    def _add_to_list(self, item):
        self._data_structure.append(item)

    def _grab_data_with_h5py(self):
        """Grab the data stored in an hdf5 file using h5py
        """
        samples = []
        f = h5py.File(self.fil)
        f.visit(self._add_to_list)
        for i in self._data_structure:
            condition1 = "posterior_samples" in i or "posterior"in i
            condition2 = "posterior_samples/" not in i and "posterior/" not in i
            if condition1 and condition2:
                path = i
        f.close()
        if self.lalinference_hdf5_format:
            g = LALInferenceResultsFile(self.fil)
            parameters, samples = g.grab_samples()
        elif self.bilby_hdf5_format:
            f = h5py.File(self.fil)
            parameters, data = [], []
            blocks = [i for i in f["%s" % (path)] if "block" in i]
            for i in blocks:
                block_name = i.split("_")[0]
                if "items" in i:
                    for par in f["%s/%s" % (path, i)]:
                        if par == b"waveform_approximant":
                            blocks.remove(block_name + "_items")
                            blocks.remove(block_name + "_values")
            for i in sorted(blocks):
                if "items" in i:
                    for par in f["%s/%s" % (path, i)]:
                        if par == b"logL":
                            parameters.append(b"log_likelihood")
                        else:
                            parameters.append(par)
                if "values" in i:
                    if len(data) == 0:
                        for dat in f["%s/%s" % (path, i)]:
                            data.append(list(np.real(dat)))
                    else:
                        for num, dat in enumerate(f["%s/%s" % (path, i)]):
                            data[num] += list(np.real(dat))
            parameters = [i.decode("utf-8") for i in parameters]
            f.close()
        return parameters, samples

    def _grab_data_with_deepdish(self):
        """Grab the data stored in an hdf5 file using deepdish
        """
        approx = "none"
        f = deepdish.io.load(self.fil)
        path, = paths_to_key("posterior", f)
        path = path[0]
        reduced_f, = load_recusively(path, f)
        parameters = [i for i in reduced_f.keys()]
        if "waveform_approximant" in parameters:
            approx = reduced_f["waveform_approximant"][0]
            parameters.remove("waveform_approximant")
        data = np.zeros([len(reduced_f[parameters[0]]), len(parameters)])
        for num, par in enumerate(parameters):
            for key, i in enumerate(reduced_f[par]):
                data[key][num] = float(np.real(i))
        data = data.tolist()
        for num, par in enumerate(parameters):
            if par == "logL":
                parameters[num] = "log_likelihood"
        return parameters, data, approx

    def _grab_data_from_json_file(self):
        """Grab the data stored in a .json file
        """
        with open(self.fil) as f:
            data = json.load(f)
        path, = paths_to_key("posterior", data)
        path = path[0]
        if "content" in data[path].keys():
            path += "/content"
        reduced_data, = load_recusively(path, data)
        parameters = list(reduced_data.keys())
        parameters = [standard_names[i] for i in list(reduced_data.keys()) if i
                      in standard_names.keys()]

        path_to_approximant = [
            i for i in paths_to_key("waveform_approximant", reduced_data)]
        try:
            approximant, = load_recusively("/".join(path_to_approximant[0]),
                                           data)
        except Exception:
            approximant = "none"

        samples = [[
            reduced_data[j][i] if not isinstance(reduced_data[j][i], dict)
            else reduced_data[j][i]["real"] for j in parameters] for i in
            range(len(reduced_data[parameters[0]]))]
        return parameters, samples, approximant

    def _grab_data_from_dat_file(self):
        """Grab the data stored in a .dat file
        """
        dat_file = np.genfromtxt(self.fil, names=True)
        stored_parameters = [i for i in dat_file.dtype.names]
        stored_parameters = self._check_definition_of_inclination(
            stored_parameters)
        parameters = [
            i for i in stored_parameters if i in standard_names.keys()]
        indices = [stored_parameters.index(i) for i in parameters]
        parameters = [standard_names[i] for i in parameters]
        samples = [[x[i] for i in indices] for x in dat_file]

        condition1 = "luminosity" not in parameters
        condition2 = "logdistance" in stored_parameters
        if condition1 and condition2:
            parameters.append("luminosity_distance")
            for num, i in enumerate(dat_file):
                samples[num].append(
                    np.exp(i[stored_parameters.index("logdistance")]))

        return parameters, samples

    def _extract_data_from_config_file(self, cp):
        """Grab the data from a config file
        """
        config = configparser.ConfigParser()
        try:
            config.read(cp)
            fixed_data = None
            marg_par = None
            if "engine" in config.sections():
                fixed_data = {
                    key.split("fix-")[1]: item for key, item in
                    config.items("engine") if "fix" in key}
                marg_par = {
                    key.split("marg")[1]: item for key, item in
                    config.items("engine") if "marg" in key}
            return {"fixed_data": fixed_data,
                    "marginalized_parameters": marg_par}
        except Exception:
            return {"fixed_data": None,
                    "marginalized_parameters": None}

    def _grab_injection_data_from_xml_file(self):
        """Grab the data from an xml injection file
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
            "mass_ratio": lambda inj: con.q_from_m1_m2(inj.mass1, inj.mass2),
            "symmetric_mass_ratio": lambda inj: con.eta_from_m1_m2(
                inj.mass1, inj.mass2),
            "total_mass": lambda inj: inj.mass1 + inj.mass2,
            "chi_p": lambda inj: con._chi_p(
                inj.mass1, inj.mass2, inj.spin1x, inj.spin1y, inj.spin2x,
                inj.spin2y),
            "chi_eff": lambda inj: con._chi_eff(inj.mass1, inj.mass2,
                                                inj.spin1z, inj.spin2z)}
        injection_parameters = self.parameters
        if GLUE:
            xmldoc = ligolw_utils.load_filename(
                self.inj, contenthandler=lsctables.use_in(
                    ligolw.LIGOLWContentHandler))
            table = lsctables.SimInspiralTable.get_table(xmldoc)[0]
            injection_values = [
                func_map[i](table) if i in func_map.keys() else float("nan")
                for i in self.parameters]
        else:
            injection_values = [float("nan")] * len(self.parameters)
        return injection_parameters, injection_values

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

    def _mchirp_from_mchirp_source_z(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["chirp_mass_source", "redshift"])
        chirp_mass = con.mchirp_from_mchirp_source_z(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _q_from_eta(self):
        self.parameters.append("mass_ratio")
        samples = self.specific_parameter_samples("symmetric_mass_ratio")
        mass_ratio = con.q_from_eta(samples)
        self.append_data(mass_ratio)

    def _q_from_m1_m2(self):
        self.parameters.append("mass_ratio")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        mass_ratio = con.q_from_m1_m2(samples[0], samples[1])
        self.append_data(mass_ratio)

    def _invert_q(self):
        ind = self.parameters.index("mass_ratio")
        for num, i in enumerate(self.samples):
            self.samples[num][ind] = 1. / self.samples[num][ind]

    def _mchirp_from_mtotal_q(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["total_mass", "mass_ratio"])
        chirp_mass = con.mchirp_from_mtotal_q(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _m1_from_mchirp_q(self):
        self.parameters.append("mass_1")
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_1 = con.m1_from_mchirp_q(samples[0], samples[1])
        self.append_data(mass_1)

    def _m2_from_mchirp_q(self):
        self.parameters.append("mass_2")
        samples = self.specific_parameter_samples(["chirp_mass", "mass_ratio"])
        mass_2 = con.m2_from_mchirp_q(samples[0], samples[1])
        self.append_data(mass_2)

    def _reference_frequency(self):
        self.parameters.append("reference_frequency")
        nsamples = len(self.samples)
        self.append_data([20.] * nsamples)

    def _mtotal_from_m1_m2(self):
        self.parameters.append("total_mass")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        m_total = con.m_total_from_m1_m2(samples[0], samples[1])
        self.append_data(m_total)

    def _mchirp_from_m1_m2(self):
        self.parameters.append("chirp_mass")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        chirp_mass = con.m_total_from_m1_m2(samples[0], samples[1])
        self.append_data(chirp_mass)

    def _eta_from_m1_m2(self):
        self.parameters.append("symmetric_mass_ratio")
        samples = self.specific_parameter_samples(["mass_1", "mass_2"])
        eta = con.eta_from_m1_m2(samples[0], samples[1])
        self.append_data(eta)

    def _phi_12_from_phi1_phi2(self):
        self.parameters.append("phi_12")
        samples = self.specific_parameter_samples(["phi_1", "phi_2"])
        phi_12 = con.phi_12_from_phi1_phi2(samples[0], samples[1])
        self.append_data(phi_12)

    def _spin_angles(self):
        spin_angles = ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12",
                       "a_1", "a_2"]
        spin_angles_to_calculate = [
            i for i in spin_angles if i not in self.parameters]
        for i in spin_angles_to_calculate:
            self.parameters.append(i)
        spin_components = [
            "mass_1", "mass_2", "iota", "spin_1x", "spin_1y", "spin_1z",
            "spin_2x", "spin_2y", "spin_2z", "reference_frequency", "phase"]
        samples = self.specific_parameter_samples(spin_components)
        spin_angles = con.spin_angles(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spin_angles_to_calculate:
            ind = spin_angles_to_calculate.index(i)
            data = np.array([i[ind] for i in spin_angles])
            self.append_data(data)

    def _component_spins(self):
        spins = ["iota", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y",
                 "spin_2z"]
        spins_to_calculate = [
            i for i in spins if i not in self.parameters]
        for i in spins_to_calculate:
            self.parameters.append(i)
        spin_angles = [
            "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
            "mass_1", "mass_2", "reference_frequency", "phase"]
        samples = self.specific_parameter_samples(spin_angles)
        spin_components = con.component_spins(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])

        for i in spins_to_calculate:
            ind = spins.index(i)
            data = np.array([i[ind] for i in spin_components])
            self.append_data(data)

    def _chi_p(self):
        self.parameters.append("chi_p")
        parameters = [
            "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]
        samples = self.specific_parameter_samples(parameters)
        chi_p_samples = con.chi_p(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5])
        self.append_data(chi_p_samples)

    def _chi_eff(self):
        self.parameters.append("chi_eff")
        parameters = ["mass_1", "mass_2", "spin_1z", "spin_2z"]
        samples = self.specific_parameter_samples(parameters)
        chi_eff_samples = con.chi_eff(
            samples[0], samples[1], samples[2], samples[3])
        self.append_data(chi_eff_samples)

    def _cos_tilt_1_from_tilt_1(self):
        self.parameters.append("cos_tilt_1")
        samples = self.specific_parameter_samples("tilt_1")
        cos_tilt_1 = np.cos(samples)
        self.append_data(cos_tilt_1)

    def _cos_tilt_2_from_tilt_2(self):
        self.parameters.append("cos_tilt_2")
        samples = self.specific_parameter_samples("tilt_2")
        cos_tilt_2 = np.cos(samples)
        self.append_data(cos_tilt_2)

    def _dL_from_z(self):
        self.parameters.append("luminosity_distance")
        samples = self.specific_parameter_samples("redshift")
        distance = con.dL_from_z(samples)
        self.append_data(distance)

    def _z_from_dL(self):
        self.parameters.append("redshift")
        samples = self.specific_parameter_samples("luminosity_distance")
        redshift = con.z_from_dL(samples)
        self.append_data(redshift)

    def _comoving_distance_from_z(self):
        self.parameters.append("comoving_distance")
        samples = self.specific_parameter_samples("redshift")
        distance = con.comoving_distance_from_z(samples)
        self.append_data(distance)

    def _m1_source_from_m1_z(self):
        self.parameters.append("mass_1_source")
        samples = self.specific_parameter_samples(["mass_1", "redshift"])
        mass_1_source = con.m1_source_from_m1_z(samples[0], samples[1])
        self.append_data(mass_1_source)

    def _m2_source_from_m2_z(self):
        self.parameters.append("mass_2_source")
        samples = self.specific_parameter_samples(["mass_2", "redshift"])
        mass_2_source = con.m2_source_from_m2_z(samples[0], samples[1])
        self.append_data(mass_2_source)

    def _mtotal_source_from_mtotal_z(self):
        self.parameters.append("total_mass_source")
        samples = self.specific_parameter_samples(["total_mass", "redshift"])
        total_mass_source = con.m_total_source_from_mtotal_z(samples[0], samples[1])
        self.append_data(total_mass_source)

    def _mchirp_source_from_mchirp_z(self):
        self.parameters.append("chirp_mass_source")
        samples = self.specific_parameter_samples(["chirp_mass", "redshift"])
        chirp_mass_source = con.mchirp_source_from_mchirp_z(samples[0], samples[1])
        self.append_data(chirp_mass_source)

    def generate_all_posterior_samples(self):
        logger.debug("Starting to generate all derived posteriors")
        if "chirp_mass" not in self.parameters and "chirp_mass_source" in \
                self.parameters and "redshift" in self.parameters:
            self._mchirp_from_mchirp_source_z()
        if "mass_ratio" not in self.parameters and "symmetric_mass_ratio" in \
                self.parameters:
            self._q_from_eta()
        if "mass_ratio" not in self.parameters and "mass_1" in self.parameters \
                and "mass_2" in self.parameters:
            self._q_from_m1_m2()
        if "mass_ratio" in self.parameters:
            ind = self.parameters.index("mass_ratio")
            median = np.median([i[ind] for i in self.samples])
            if median < 1.:
                self._invert_q()
        if "chirp_mass" not in self.parameters and "total_mass" in self.parameters:
            self._mchirp_from_mtotal_q()
        if "mass_1" not in self.parameters and "chirp_mass" in self.parameters:
            self._m1_from_mchirp_q()
        if "mass_2" not in self.parameters and "chirp_mass" in self.parameters:
            self._m2_from_mchirp_q()
        if "reference_frequency" not in self.parameters:
            self._reference_frequency()

        condition1 = "phi_12" not in self.parameters
        condition2 = "phi_1" in self.parameters and "phi_2" in self.parameters
        if condition1 and condition2:
            self._phi_12_from_phi1_phi2()
        if "mass_1" in self.parameters and "mass_2" in self.parameters:
            if "total_mass" not in self.parameters:
                self._mtotal_from_m1_m2()
            if "chirp_mass" not in self.parameters:
                self._mchirp_from_m1_m2()
            if "symmetric_mass_ratio" not in self.parameters:
                self._eta_from_m1_m2()
            spin_components = [
                "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
            spin_angles = [
                "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1",
                "a_2", "mass_1", "mass_2", "reference_frequency", "phase"]
            if all(i in self.parameters for i in spin_components):
                self._spin_angles()
            if all(i in self.parameters for i in spin_angles):
                self._component_spins()
            if "chi_p" not in self.parameters and "chi_eff" not in self.parameters:
                if all(i in self.parameters for i in spin_angles):
                    self._chi_p()
                    self._chi_eff()
        if "cos_tilt_1" not in self.parameters and "tilt_1" in self.parameters:
            self._cos_tilt_1_from_tilt_1()
        if "cos_tilt_2" not in self.parameters and "tilt_2" in self.parameters:
            self._cos_tilt_2_from_tilt_2()
        if "luminosity_distance" not in self.parameters and "redshift" in self.parameters:
            self._dL_from_z()
        if "redshift" not in self.parameters and "luminosity_distance" in self.parameters:
            self._z_from_dL()
        if "comoving_distance" not in self.parameters and "redshift" in self.parameters:
            self._comoving_distance_from_z()
        if "redshift" in self.parameters:
            if "mass_1_source" not in self.parameters and "mass_1" in self.parameters:
                self._m1_source_from_m1_z()
            if "mass_2_source" not in self.parameters and "mass_2" in self.parameters:
                self._m2_source_from_m2_z()
            if "total_mass_source" not in self.parameters and "total_mass" in self.parameters:
                self._mtotal_source_from_mtotal_z()
            if "chirp_mass_source" not in self.parameters and "chirp_mass" in self.parameters:
                self._mchirp_source_from_mchirp_z()
        if "reference_frequency" in self.parameters:
            ind = self.parameters.index("reference_frequency")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        if "minimum_frequency" in self.parameters:
            ind = self.parameters.index("minimum_frequency")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        if "logPrior" in self.parameters:
            ind = self.parameters.index("logPrior")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        if "log_prior" in self.parameters:
            ind = self.parameters.index("log_prior")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                del i[ind]
        self._update_injection_data()

    def _update_injection_data(self):
        if self.inj:
            extension = self.inj.split(".")[-1]
            func_map = {"xml": self._grab_injection_data_from_xml_file,
                        "hdf5": self._grab_injection_data_from_hdf5_file,
                        "h5": self._grab_injection_data_from_hdf5_file}
            self._injection_data = func_map[extension]()
        else:
            self._injection_data = [
                self.parameters, [float("nan")] * len(self.parameters)]

    def save(self):
        parameters = np.array(self.parameters, dtype="S")
        injection_parameters = np.array(self.injection_parameters, dtype="S")
        injection_data = np.array(self.injection_values)
        f = h5py.File("%s_temp" % (self.fil), "w")
        posterior_samples_group = f.create_group("posterior_samples")
        label_group = posterior_samples_group.create_group("label")
        group = label_group.create_group(self.approximant)
        group.create_dataset("parameter_names", data=parameters)
        group.create_dataset("samples", data=self.samples)
        group.create_dataset("injection_parameters", data=injection_parameters)
        group.create_dataset("injection_data", data=injection_data)
        f.close()
        return "%s_temp" % (self.fil)


def add_specific_arguments(parser):
    """Add command line arguments that are specific to pesummary_convert

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments
    """
    parser.add_argument("-o", "--outpath", dest="out",
                        help="location of output file", default=None)
    return parser


def main():
    """Top-level interface for pesummary_convert.py
    """
    parser = command_line()
    parser = add_specific_arguments(parser)
    opts = parser.parse_args()
    if opts.inj_file and len(opts.samples) != len(opts.inj_file):
        raise Exception("Please ensure that the number of results files "
                        "matches the number of injection files")
    if opts.config and len(opts.samples) != len(opts.config):
        raise Exception("Please ensure that the number of results files "
                        "matches the number of configuration files")
    if not opts.inj_file:
        opts.inj_file = []
        for i in range(len(opts.samples)):
            opts.inj_file.append(None)
    if not opts.config:
        opts.config = []
        for i in range(len(opts.samples)):
            opts.config.append(None)
    for num, i in enumerate(opts.samples):
        f = OneFormat(i, opts.inj_file[num], config=opts.config[num])
        f.generate_all_posterior_samples()
        g = f.save()
        if opts.out:
            shutil.move(g, opts.out)
