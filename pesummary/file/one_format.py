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

import h5py
import deepdish

import numpy as np

from pesummary.command_line import command_line
import pesummary.file.conversions as con
from pesummary.utils.utils import logger

try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False

standard_names = {"logL": "log_likelihood",
                  "logl": "log_likelihood",
                  "tilt1": "tilt_1",
                  "tilt_spin1": "tilt_1",
                  "tilt2": "tilt_2",
                  "tilt_spin2": "tilt_2",
                  "costilt1": "cos_tilt_1",
                  "costilt2": "cos_tilt_2",
                  "redshift": "redshift",
                  "l1_optimal_snr": "L1_optimal_snr",
                  "h1_optimal_snr": "H1_optimal_snr",
                  "v1_optimal_snr": "V1_optimal_snr",
                  "L1_optimal_snr": "L1_optimal_snr",
                  "H1_optimal_snr": "H1_optimal_snr",
                  "V1_optimal_snr": "V1_optimal_snr",
                  "E1_optimal_snr": "E1_optimal_snr",
                  "mc_source": "chirp_mass_source",
                  "chirpmass_source": "chirp_mass_source",
                  "eta": "symmetric_mass_ratio",
                  "m1": "mass_1",
                  "m2": "mass_2",
                  "ra": "ra",
                  "rightascension": "ra",
                  "dec": "dec",
                  "declination": "dec",
                  "iota": "iota",
                  "m2_source": "mass_2_source",
                  "m1_source": "mass_1_source",
                  "phi1": "phi_1",
                  "phi2": "phi_2",
                  "psi": "psi",
                  "polarisation": "psi",
                  "phi12": "phi_12",
                  "phi_12": "phi_12",
                  "phi_jl": "phi_jl",
                  "phijl": "phijl",
                  "a1": "a_1",
                  "a_spin1": "a_1",
                  "a2": "a_2",
                  "a_spin2": "a_2",
                  "chi_p": "chi_p",
                  "phase": "phase",
                  "distance": "luminosity_distance",
                  "dist": "luminosity_distance",
                  "mc": "chirp_mass",
                  "chirpmass": "chirp_mass",
                  "chi_eff": "chi_eff",
                  "mtotal_source": "total_mass_source",
                  "mtotal": "total_mass",
                  "q": "mass_ratio",
                  "time": "geocent_time",
                  "theta_jn": "iota"}


class OneFormat(object):
    """Class to convert a given results file into a standard format with all
    derived posterior distributions included

    Parameters
    ----------
    fil: str
       path to the results file
    inj: str
       path to the file containing injection information
    config: str, optional
       path to the configuration file

    Attributes
    ----------
    lalinference: Bool
        Boolean to determine if LALInference was used to generate the results
        file
    bilby: Bool
        Boolean to determine if bilby was used to generate the results file
    approximant: str
        The approximant that was used to generate the results file. If the
        approximant cannot be extracted, this is "none".
    parameters: list
        List of parameters that have corresponding posterior distributions
    samples: list
        List of posterior samples for each parameter
    """
    def __init__(self, fil, inj, config=None):
        self.fil = fil
        self.inj = inj
        self.config = config
        self.extension = False
        if self.extension == "dat":
            self.fil = convert_dat_to_h5(self.fil)
        self.lalinference = False
        self.bilby = False
        self.approximant = None
        self.parameters = None
        self.samples = None
        fixed_parameters = None
        if self.config and self.lalinference:
            fixed_parameters = self.fixed_parameters()
        if fixed_parameters:
            self._append_fixed_parameters(fixed_parameters)

    def fixed_parameters(self):
        """Extract the fixed parameters from the configuration file. This is
        useful for LALInference data files.

        Parameters
        ----------
        cp: str
            path to the location of the configuration file
        """
        config = configparser.ConfigParser()
        config.read(self.config)
        fixed_params = None
        if "engine" in config.sections():
            fixed_params = [
                list(i) for i in config.items("engine") if "fix" in i[0]]
        return fixed_params

    @staticmethod
    def keys(fil):
        f = h5py.File(fil)
        keys = [i for i in f.keys()]
        f.close()
        return keys

    def grab_approximant(self):
        approx = "none"
        if self.bilby:
            try:
                f = deepdish.io.load(self.fil)
                parameters = [i for i in f["posterior"].keys()]
                if "waveform_approximant" in parameters:
                    approx = f["posterior"]["waveform_approximant"][0]
            except Exception:
                pass
        return approx

    @property
    def lalinference(self):
        return self._lalinference

    @lalinference.setter
    def lalinference(self, lalinference):
        f = h5py.File(self.fil)
        keys = self.keys(self.fil)
        if "lalinference" in keys:
            logger.info("LALInference was used to generate %s" % (self.fil))
            self._lalinference = True
            sampler = [i for i in f["lalinference"].keys()]
            self._data_path = "lalinference/%s/posterior_samples" % (sampler[0])
        else:
            self._lalinference = False

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, extension):
        ext = self.fil.split(".")[-1]
        if ext == "h5" or ext == "hdf5":
            self._extension = "h5"
        elif ext == "dat":
            self._extension = "dat"

    @property
    def bilby(self):
        return self._bilby

    @bilby.setter
    def bilby(self, bilby):
        self._bilby = False
        keys = self.keys(self.fil)
        if "data" in keys:
            logger.info("BILBY >= v0.3.3 was used to generate %s" % (self.fil))
            self._bilby = True
            self._data_path = "data/posterior"
        elif "posterior" in keys:
            logger.info("BILBY >= v0.3.1 was used to generate %s" % (self.fil))
            self._bilby = True
            self._data_path = "posterior"

    @property
    def approximant(self):
        return self._approximant

    @approximant.setter
    def approximant(self, approximant):
        self._approximant = self.grab_approximant()

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = self.grab_data()[0]

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = self.grab_data()[1]

    def _append_fixed_parameters(self, fixed_parameters):
        """Generate samples for the fixed parameters and append them to them
        to the samples array.

        Parameters
        ----------
        fixed_parameters: ndlist
            list of fixed parameters and their fixed values
        """
        if not fixed_parameters:
            pass
        for i in fixed_parameters:
            try:
                param = standard_names[i[0].split("fix-")[1]]
                if param in self.parameters:
                    pass
                self.parameters.append(param)
                self.append_data([float(i[1])] * len(self.samples))
            except Exception:
                param = i[0].split("fix-")[1]
                if param == "logdistance":
                    self.parameters.append(standard_names["distance"])
                    self.append_data([np.exp(float(i[1]))] * len(self.samples))
                if param == "costheta_jn":
                    self.parameters.append(standard_names["theta_jn"])
                    self.append_data([np.arccos(float(i[1]))] * len(self.samples))

    def injection_parameters(self):
        return get_injection_parameters(
            self.parameters, self.inj, LALINFERENCE=self.lalinference,
            BILBY=self.bilby)

    def save(self):
        parameters = np.array(self.parameters, dtype="S")
        injection_properties = self.injection_parameters()
        injection_parameters = np.array(injection_properties[0], dtype="S")
        injection_data = np.array(injection_properties[1])
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

    def grab_data(self):
        f = h5py.File(self.fil)
        if self.lalinference:
            lalinference_names = f[self._data_path].dtype.names
            data = []
            parameters = [i for i in lalinference_names if i in standard_names.keys()]
            for i in f[self._data_path]:
                data.append([i[lalinference_names.index(j)] for j in parameters])
            parameters = [standard_names[i] for i in parameters]
            if "luminosity_distance" not in parameters and "logdistance" in lalinference_names:
                parameters.append("luminosity_distance")
                for num, i in enumerate(f[self._data_path]):
                    data[num].append(np.exp(i[lalinference_names.index("logdistance")]))
            if "iota" not in parameters and "costheta_jn" in lalinference_names:
                parameters.append("iota")
                for num, i in enumerate(f[self._data_path]):
                    data[num].append(np.arccos(i[lalinference_names.index("costheta_jn")]))
        if self.bilby:
            approx = "none"
            try:
                logger.debug("Trying to load with file with deepdish")
                f = deepdish.io.load(self.fil)
                parameters, data, approx = load_with_deepdish(f)
            except Exception as e:
                logger.debug("Failed to load file with deepdish because %s. "
                             "Using h5py instead" % (e))
                f = h5py.File(self.fil)
                parameters, data = load_with_h5py(f, self._data_path)
        return parameters, data

    def _specific_parameter_samples(self, param):
        ind = self.parameters.index(param)
        samples = np.array([i[ind] for i in self.samples])
        return samples

    def specific_parameter_samples(self, param):
        if type(param) == list:
            samples = [self._specific_parameter_samples(i) for i in param]
        else:
            samples = self._specific_parameter_samples(param)
        return samples

    def append_data(self, samples):
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

    def _component_spins(self):
        spins = ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
        for i in spins:
            self.parameters.append(i)
        spin_angles = [
            "iota", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
            "mass_1", "mass_2", "reference_frequency", "phase"]
        samples = self.specific_parameter_samples(spin_angles)
        spin_components = con.component_spins(
            samples[0], samples[1], samples[2], samples[3], samples[4],
            samples[5], samples[6], samples[7], samples[8], samples[9],
            samples[10])
        spin1x = np.array([i[1] for i in spin_components])
        spin1y = np.array([i[2] for i in spin_components])
        spin1z = np.array([i[3] for i in spin_components])
        spin2x = np.array([i[4] for i in spin_components])
        spin2y = np.array([i[5] for i in spin_components])
        spin2z = np.array([i[6] for i in spin_components])
        self.append_data(spin1x)
        self.append_data(spin1y)
        self.append_data(spin1z)
        self.append_data(spin2x)
        self.append_data(spin2y)
        self.append_data(spin2z)

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
                "iota", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1",
                "a_2", "mass_1", "mass_2", "reference_frequency", "phase"]
            if all(i not in self.parameters for i in spin_components):
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
                i.remove(i[ind])
        if "minimum_frequency" in self.parameters:
            ind = self.parameters.index("minimum_frequency")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                i.remove(i[ind])
        if "logPrior" in self.parameters:
            ind = self.parameters.index("logPrior")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                i.remove(i[ind])
        if "log_prior" in self.parameters:
            ind = self.parameters.index("log_prior")
            self.parameters.remove(self.parameters[ind])
            for i in self.samples:
                i.remove(i[ind])


def load_with_deepdish(f):
    """Return the data and parameters that appear in a given h5 file assuming
    that the file has been loaded with deepdish

    Parameters
    ----------
    f: dict
        results file loaded with deepdish
    """
    approx = "none"
    parameters = [i for i in f["posterior"].keys()]
    if "waveform_approximant" in parameters:
        approx = f["posterior"]["waveform_approximant"][0]
        parameters.remove("waveform_approximant")
    data = np.zeros([len(f["posterior"]), len(parameters)])
    for num, par in enumerate(parameters):
        for key, i in enumerate(f["posterior"][par]):
            data[key][num] = float(np.real(i))
    data = data.tolist()
    for num, par in enumerate(parameters):
        if par == "logL":
            parameters[num] = "log_likelihood"
    return parameters, data, approx


def load_with_h5py(f, path):
    """Return the data and parameters that appear in a given h5 file assuming
    that the file has been loaded with h5py

    Parameters
    ----------
    f: h5py._hl.files.File
        results file loaded with h5py
    """
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
    return parameters, data


def get_injection_parameters(parameters, inj_file, LALINFERENCE=False,
                             BILBY=False):
    """Grab the injection parameters from an xml injection file

    Parameters
    ----------
    parameters: list
        list of parameters that you have samples for
    inj_file: str
        path to the location of the injection file
    """
    _q_func = con.q_from_m1_m2
    _eta_func = con.eta_from_m1_m2
    func_map = {"chirp_mass": lambda inj: inj.mchirp,
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
                "mass_ratio": lambda inj: _q_func(inj.mass1, inj.mass2),
                "symmetric_mass_ratio": lambda inj: _eta_func(inj.mass1, inj.mass2),
                "total_mass": lambda inj: inj.mass1 + inj.mass2,
                "chi_p": lambda inj: con._chi_p(inj.mass1, inj.mass2, inj.spin1x,
                                                inj.spin1y, inj.spin2x, inj.spin2y),
                "chi_eff": lambda inj: con._chi_eff(inj.mass1, inj.mass2, inj.spin1z,
                                                    inj.spin2z)}
    inj_par = parameters
    if LALINFERENCE:
        if inj_file is None:
            inj_data = [float("nan")] * len(parameters)
        else:
            if GLUE:
                xmldoc = ligolw_utils.load_filename(
                    inj_file, contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
                table = lsctables.SimInspiralTable.get_table(xmldoc)[0]
                inj_data = [func_map[i](table) if i in func_map.keys() else
                            float("nan") for i in parameters]
            else:
                inj_data = [float("nan")] * len(parameters)
    if BILBY:
        try:
            f = deepdish.io.load(inj_file)
            inj_keys = f["injection_parameters"].keys()
            inj_data = [f["injection_parameters"][key] if key in inj_keys
                        else float("nan") for key in parameters]
        except Exception:
            inj_data = [float("nan")] * len(parameters)
    return [inj_par, inj_data]


def convert_dat_to_h5(f):
    """Convert a dat file to the lalinference framework

    Parameters
    ----------
    f: str
        path to the dat file that you would like to convert to h5
    """
    dat_file = np.genfromtxt(f, names=True)
    file_name = f.split(".dat")[0]
    h5_file = h5py.File("%s.hdf5" % (file_name), 'w')
    lalinference_group = h5_file.create_group('lalinference')
    sampler_group = lalinference_group.create_group('lalinference_mcmc')
    sampler_group.create_dataset('posterior_samples', data=dat_file)
    h5_file.close()
    return "%s.hdf5" % (file_name)


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
