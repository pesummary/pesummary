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

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import h5py
import deepdish

import numpy as np

try:
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from lal import MSUN_SI
    LALINFERENCE_INSTALL=True
except:
    LALINFERENCE_INSTALL=False

def _make_hdf5_file(name, data, parameters, approximant=None):
    """
    """
    f = h5py.File("%s_temp" %(name), "w")
    f.create_dataset("parameter_names", data=parameters)
    f.create_dataset("samples", data=data)
    f.create_dataset("approximant", data=approximant)
    f.close()

def _mchirp_from_m1_m2(mass1, mass2):
    """Return the chirp mass given the samples for mass1 and mass2

    Parameters
    ----------
    """
    return (mass1 * mass2)**0.6 / (mass1 + mass2)**0.2

def _m_total_from_m1_m2(mass1, mass2):
    """Return the total mass given the samples for mass1 and mass2
    """
    return mass1 + mass2

def _m1_from_mchirp_q(mchirp, q):
    """Return the mass of the larger black hole given the chirp mass and
    mass ratio
    """
    return (q**(2./5.))*((1.0 + q)**(1./5.))*mchirp 

def _m2_from_mchirp_q(mchirp, q):
    """Return the mass of the smaller black hole given the chirp mass and
    mass ratio
    """
    return (q**(-3./5.))*((1.0 + q)**(1./5.))*mchirp

def _eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    return (mass1 * mass2) / (mass1 + mass2)**2

def _q_from_m1_m2(mass1, mass2):
    """Return the mass ratio given the samples for mass1 and mass2
    """
    return mass1 / mass2

def _q_from_eta(chirp_mass, symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5

def _mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    mass1 = mass_ratio*total_mass / (1.+mass_ratio)
    mass2 = total_mass / (1.+mass_ratio)
    return eta_from_m1_m2(mass1, mass2)**(3./5) * (mass1+mass2)

def _chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return chi_p given samples for mass1, mass2, spin1x, spin1y, spin2x,
    spin2y
    """
    mass_ratio = mass1/mass2
    B1 = 2.0 + 1.5*mass_ratio
    B2 = 2.0 + 3.0 / (2*mass_ratio)
    S1_perp = (spin1x**2 + spin1y**2)**0.5
    S2_perp = (spin2x**2 + spin2y**2)**0.5
    chi_p = 1.0/B1 * np.maximum(B1*S1_perp, B2*S2_perp)
    return chi_p

def _chi_eff(mass1, mass2, spin1z, spin2z):
    """Return chi_eff given samples for mass1, mass2, spin1z, spin2z
    """
    return (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)

def _component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                     mass_2, f_ref, phase):
    """Return the component spins given sames for theta_jn, phi_jl, tilt_1,
    tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase
    """
    if LALINFERENCE_INSTALL:
        data = []
        for i in range(len(theta_jn)):
            iota, S1x, S1y, S1z, S2x, S2y, S2z = \
                SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn[i], phi_jl[i], tilt_1[i], tilt_2[i], phi_12[i],
                    a_1[i], a_2[i], mass_1[i]*MSUN_SI, mass_2[i]*MSUN_SI,
                    f_ref[i], phase[i])
            data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
        return data
    else:
        raise Exception("Please install LALSuite for full conversions")

def all_parameters(data, parameters):
    """Return an array of samples for all CBC parameters

    Parameters
    ----------
    data: list
        list of samples returned from your sampler of choice
    parameters: list
        list of parameters that have been sampled over
    """
    if b"mass_ratio" not in parameters and b"symmetric_mass_ratio" in parameters:
        parameters.append("mass_ratio")
        symmetric_mass_ratio_ind = parameters.index(b"symmetric_mass_ratio")
        symmetric_mass_ratio = np.array([i[symmetric_mass_ratio_ind] for i in data])
        mass_ratio = _q_from_eta(symmetric_mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_ratio[num])
    if b"mass_ratio" not in parameters and b"mass_1" in parameters and b"mass_2" in parameters:
        parameters.append(b"mass_ratio")
        mass1_ind = parameters.index(b"mass_1")
        mass1 = np.array([i[mass1_ind] for i in data])
        mass2_ind = parameters.index(b"mass_2")
        mass2 = np.array([i[mass2_ind] for i in data])
        q = _q_from_m1_m2(mass1, mass2)
        for num, i in enumerate(data):
            data[num].append(q[num])
    if b"mass_ratio" in parameters:
        mass_ratio_ind = parameters.index(b"mass_ratio")
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        median = np.median(mass_ratio)
        if median < 1:
            # define mass_ratio so that q>1
             for i in data:
                 i[mass_ratio_ind] = 1/i[mass_ratio_ind]
    if b"chirp_mass" not in parameters and b"total_mass" in parameters:
        parameters.append(b"chirp_mass")
        total_mass_ind = parameters.index(b"total_mass")
        mass_ratio_ind = parameters.index(b"mass_ratio")
        total_mass = np.array([i[total_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        chirp_mass = _mchirp_from_mtotal_q(total_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(chirp_mass[num])
    if b"mass_1" not in parameters and b"chirp_mass" in parameters:
        parameters.append(b"mass_1")
        chirp_mass_ind = parameters.index(b"chirp_mass")
        mass_ratio_ind = parameters.index(b"mass_ratio")
        chirp_mass = np.array([i[chirp_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        mass_1 = _m1_from_mchirp_q(chirp_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_1[num])
    if b"mass_2" not in parameters and b"chirp_mass" in parameters:
        parameters.append(b"mass_2")
        chirp_mass_ind = parameters.index(b"chirp_mass")
        mass_ratio_ind = parameters.index(b"mass_ratio")
        chirp_mass = np.array([i[chirp_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        mass_2 = _m2_from_mchirp_q(chirp_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_2[num])

    ##################################################
    # True reference frequency needs to be extracted #
    #              NEEDS TO BE FIXED                 #
    ##################################################

    if b"reference_frequency" not in parameters:
        parameters.append(b"reference_frequency")
        for num, i in enumerate(data):
            data[num].append(20.)

    if b"mass_1" in parameters and b"mass_2" in parameters:
        mass1_ind = parameters.index(b"mass_1")
        mass1 = np.array([i[mass1_ind] for i in data])
        mass2_ind = parameters.index(b"mass_2")
        mass2 = np.array([i[mass2_ind] for i in data])
        if b"total_mass" not in parameters:
            parameters.append(b"total_mass")
            m_total = _m_total_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(m_total[num])
        if b"chirp_mass" not in parameters:
            parameters.append(b"chirp_mass")
            mchirp = _mchirp_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(mchirp[num])
        if b"symmetric_mass_ratio" not in parameters:
            parameters.append(b"symmetric_mass_ratio")
            eta = _eta_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(eta[num])

        spin_components = [b"spin_1x", b"spin_1y", b"spin_1z", b"spin_2x",
                           b"spin_2y", b"spin_2z"]
        spin_angles = [b"iota", b"phi_jl", b"tilt_1", b"tilt_2", b"phi_12", b"a_1", b"a_2",
                       b"mass_1", b"mass_2", b"reference_frequency", b"phase"]
        if all(i not in parameters for i in spin_components):
            if all(i in parameters for i in spin_angles):
                parameters.append(b"spin_1x")
                parameters.append(b"spin_1y")
                parameters.append(b"spin_1z")
                parameters.append(b"spin_2x")
                parameters.append(b"spin_2y")
                parameters.append(b"spin_2z")
                indices = [parameters.index(b"%s" %(i)) for i in spin_angles]
                iota = np.array([i[indices[0]] for i in data])
                phijl = np.array([i[indices[1]] for i in data])
                tilt1 = np.array([i[indices[2]] for i in data])
                tilt2 = np.array([i[indices[3]] for i in data])
                phi12 = np.array([i[indices[4]] for i in data])
                a1 = np.array([i[indices[5]] for i in data])
                a2 = np.array([i[indices[6]] for i in data])
                mass1 = np.array([i[indices[7]] for i in data])
                mass2 = np.array([i[indices[8]] for i in data])
                f_ref = np.array([i[indices[9]] for i in data])
                phase = np.array([i[indices[10]] for i in data])
                spin_parameters = _component_spins(iota, phijl, tilt1, tilt2, phi12,
                                                   a1, a2, mass1, mass2, f_ref, phase)
                for num, i in enumerate(data):
                    data[num].append(spin_parameters[num][1])
                    data[num].append(spin_parameters[num][2])
                    data[num].append(spin_parameters[num][3])
                    data[num].append(spin_parameters[num][4])
                    data[num].append(spin_parameters[num][5])
                    data[num].append(spin_parameters[num][6])
        if b"chi_p" not in parameters and b"chi_eff" not in parameters:
            if all(i in parameters for i in spin_angles):
                parameters.append(b"chi_p")
                parameters.append(b"chi_eff")
                indices = [parameters.index(b"%s" %(i)) for i in spin_components]
                spin1x = np.array([i[indices[0]] for i in data])
                spin1y = np.array([i[indices[1]] for i in data])
                spin1z = np.array([i[indices[2]] for i in data])
                spin2x = np.array([i[indices[3]] for i in data])
                spin2y = np.array([i[indices[4]] for i in data])
                spin2z = np.array([i[indices[5]] for i in data])
                chi_p = _chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y)
                chi_eff = _chi_eff(mass1, mass2, spin1z, spin2z)
                for num, i in enumerate(data):
                    data[num].append(chi_p[num])
                    data[num].append(chi_eff[num])
    if b"cos_tilt_1" not in parameters and b"tilt_1" in parameters:
        parameters.append(b"cos_tilt_1")
        tilt_1_ind = parameters.index(b"tilt_1")
        tilt_1 = np.array([i[tilt_1_ind] for i in data])
        cos_tilt_1 = np.cos(tilt_1)
        for num, i in enumerate(data):
            data[num].append(cos_tilt_1[num])
    if b"cos_tilt_2" not in parameters and b"tilt_2" in parameters:
        parameters.append(b"cos_tilt_2")
        tilt_2_ind = parameters.index(b"tilt_2")
        tilt_2 = np.array([i[tilt_2_ind] for i in data])
        cos_tilt_2 = np.cos(tilt_2)
        for num, i in enumerate(data):
            data[num].append(cos_tilt_2[num])
    return data, parameters

def one_format(fil):
    """Looks at the input file and puts it into a standard form such that all
    parameter estimation codes can use the summary pages.

    Parameters
    ----------
    fil: str
        path to the results file location
    """
    LALINFERENCE = False
    BILBY = False
    file_location = "/".join(fil.split("/")[:-1])
    f = h5py.File(fil)
    keys = [i for i in f.keys()]
    if "lalinference" in keys:
        LALINFERENCE = True
        logging.info("LALInference was used to generate %s" %(fil))
    elif "data" in keys:
        BILBY = True
        logging.info("BILBY >= v0.3.3 was used to generate %s" %(fil))
        path = "data/posterior"
    elif "posterior" in keys:
        BILBY = True
        logging.info("BILBY >= v0.3.1 <= v0.3.3 was used to generate %s" %(fil))
        path = "posterior"
    else:
        raise Exception("Data format not understood")
    if LALINFERENCE:
        standard_names = {"logL": "log_likelihood",
                          "tilt1": "tilt_1",
                          "tilt2": "tilt_2",
                          "costilt1": "cos_tilt_1",
                          "costilt2": "cos_tilt_2",
                          "redshift": "redshift",
                          "phi_jl": "phi_jl",
                          "l1_optimal_snr": "L1_optimal_snr",
                          "h1_optimal_snr": "H1_optimal_snr",
                          "mc_source": "chirp_mass_source",
                          "eta": "symmetric_mass_ratio",
                          "m1": "mass_1",
                          "m2": "mass_2",
                          "ra": "ra",
                          "dec": "dec",
                          "iota": "iota",
                          "m2_source": "mass_2_source",
                          "m1_source": "mass_1_source",
                          "phi1": "phi_1",
                          "phi2": "phi_2",
                          "psi": "psi",
                          "phi12": "phi_12",
                          "a1": "a_1",
                          "a2": "a_2",
                          "chi_p": "chi_p",
                          "phase": "phase",
                          "distance": "luminosity_distance",
                          "mc": "chirp_mass",
                          "chi_eff": "chi_eff",
                          "mtotal_source": "total_mass_source",
                          "mtotal": "total_mass",
                          "q": "mass_ratio",
                          "time": "geocent_time"}
        sampler = [i for i in f["lalinference"].keys()]
        data_path = "lalinference/%s/posterior_samples" %(sampler[0])
        lalinference_names = f[data_path].dtype.names
        parameters, data = [], []
        for i in lalinference_names:
            if i in standard_names.keys():
                parameters.append(i)
        for i in f[data_path]:
            data.append([i[lalinference_names.index(j)] for j in parameters])
        parameters = [standard_names[i] for i in parameters]
        index = lalinference_names.index
        if b"luminosity_distance" not in parameters and b"logdistance" in lalinference_names:
            parameters.append(b"luminosity_distance")
            for num, i in enumerate(f[data_path]):
                data[num].append(np.exp(i[index(b"logdistance")]))
        if b"iota" not in parameters and b"costheta_jn" in lalinference_names:
            parameters.append(b"iota")
            for num, i in enumerate(f[data_path]):
                data[num].append(np.arccos(i[index(b"costheta_jn")]))
    if BILBY:
        approx = "None"
        parameters, data = [], []
        try:
            logging.info("Trying to load with file with deepdish")
            f = deepdish.io.load(fil)
            parameters, data, approx = load_with_deepdish(f)
        except:
            logging.info("Failed to load file with deepdish. Using h5py to "
                         "load in data")
            f = h5py.File(fil)
            parameters, data = load_with_h5py(f, path)
    data, parameters = all_parameters(data, parameters)
    if b"reference_frequency" in parameters:
        index = parameters.index(b"reference_frequency")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    if b"minimum_frequency" in parameters:
        index = parameters.index(b"minimum_frequency")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    if b"logPrior" in parameters:
        index = parameters.index(b"logPrior")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    _make_hdf5_file(fil, np.array(data), np.array(parameters, dtype="S"),
                    approximant=np.array([approx], dtype="S"))
    return "%s_temp" %(fil)

def load_with_deepdish(f):
    """Return the data and parameters that appear in a given h5 file assuming
    that the file has been loaded with deepdish

    Parameters
    ----------
    f: dict
        results file loaded with deepdish 
    """
    approx = None
    parameters = [i for i in f["posterior"].keys()]
    if "waveform_approximant" in parameters:
        approx = f["posterior"]["waveform_approximant"][0]
        parameters.remove("waveform_approximant")
    data = [[float(np.real(i)) for i in f["posterior"][par]] for par in parameters]
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
    blocks = [i for i in f["%s" %(path)] if "block" in i]
    for i in blocks:
        block_name = i.split("_")[0]
        if "items" in i:
            for par in f["%s/%s" %(path,i)]:
                if par == b"waveform_approximant":
                    blocks.remove(block_name+"_items")
                    blocks.remove(block_name+"_values")
    for i in sorted(blocks):
        if "items" in i:
            for par in f["%s/%s" %(path,i)]:
                if par == b"logL":
                    parameters.append(b"log_likelihood")
                else:
                    parameters.append(par)
        if "values" in i:
            if len(data) == 0:
                for dat in f["%s/%s" %(path, i)]:
                    data.append(list(np.real(dat)))
            else:
                for num, dat in enumerate(f["%s/%s" %(path, i)]):
                    data[num] += list(np.real(dat))
    return parameters, data
