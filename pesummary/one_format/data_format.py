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
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE=True
except:
    GLUE=False

try:
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from lal import MSUN_SI
    LALINFERENCE_INSTALL=True
except:
    LALINFERENCE_INSTALL=False

try:
    from astropy.cosmology import z_at_value, Planck15
    import astropy.units as u
    ASTROPY=True
except ImportError:
    ASTROPY=False
    logger.warning("You do not have astropy installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


def _make_hdf5_file(name, data, parameters, approximant, inj_par=None,
                    inj_data=None):
    """
    """
    f = h5py.File("%s_temp" %(name), "w")
    f.create_dataset("parameter_names", data=parameters)
    f.create_dataset("samples", data=data)
    f.create_dataset("approximant", data=approximant)
    f.create_dataset("injection_parameters", data=inj_par)
    f.create_dataset("injection_data", data=inj_data)
    f.close()

@np.vectorize
def _z_from_dL(luminosity_distance):
    """Return the redshift given samples for the luminosity distance
    """
    return z_at_value(Planck15.luminosity_distance, luminosity_distance*u.Mpc)

def _dL_from_z(redshift):
    """Return the luminosity distance given samples for the redshift
    """
    return Planck15.luminosity_distance(redshift).value

def _comoving_distance_from_z(redshift):
    """Return the comoving distance given samples for the redshift
    """
    return Planck15.comoving_distance(redshift).value 

def _m1_source_from_m1_z(mass1, z):
    """Return the source mass of the bigger black hole given samples for the
    detector mass of the bigger black hole and the redshift
    """
    return mass1 / (1. + z)

def _m2_source_from_m2_z(mass2, z):
    """Return the source mass of the smaller black hole given samples for the
    detector mass of the smaller black hole and the redshift
    """
    return mass2 / (1. + z)

def _m_total_source_from_mtotal_z(total_mass, z):
    """Return the source total mass of the binary given samples for detector
    total mass and redshift
    """
    return total_mass / (1. + z)

def _mchirp_source_from_mchirp_z(mchirp, z):
    """Return the source chirp mass of the binary given samples for detector
    chirp mass and redshift
    """
    return mchirp / (1. + z)

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

def _q_from_eta(symmetric_mass_ratio):
    """Return the mass ratio given samples for symmetric mass ratio
    """
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5

def _mchirp_from_mtotal_q(total_mass, mass_ratio):
    """Return the chirp mass given samples for total mass and mass ratio
    """
    mass1 = mass_ratio*total_mass / (1.+mass_ratio)
    mass2 = total_mass / (1.+mass_ratio)
    return _eta_from_m1_m2(mass1, mass2)**(3./5) * (mass1+mass2)

def _chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Return chi_p given samples for mass1, mass2, spin1x, spin1y, spin2x,
    spin2y
    """
    mass_ratio = mass1/mass2
    B1 = 2.0 + 1.5*mass_ratio
    B2 = 2.0 + 3.0 / (2*mass_ratio)
    S1_perp = ((spin1x)**2 + (spin1y)**2)**0.5
    S2_perp = ((spin2x)**2 + (spin2y)**2)**0.5
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
    if "mass_ratio" not in parameters and "symmetric_mass_ratio" in parameters:
        parameters.append("mass_ratio")
        symmetric_mass_ratio_ind = parameters.index("symmetric_mass_ratio")
        symmetric_mass_ratio = np.array([i[symmetric_mass_ratio_ind] for i in data])
        mass_ratio = _q_from_eta(symmetric_mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_ratio[num])
    if "mass_ratio" not in parameters and "mass_1" in parameters and "mass_2" in parameters:
        parameters.append("mass_ratio")
        mass1_ind = parameters.index("mass_1")
        mass1 = np.array([i[mass1_ind] for i in data])
        mass2_ind = parameters.index("mass_2")
        mass2 = np.array([i[mass2_ind] for i in data])
        q = _q_from_m1_m2(mass1, mass2)
        for num, i in enumerate(data):
            data[num].append(q[num])
    if "mass_ratio" in parameters:
        mass_ratio_ind = parameters.index("mass_ratio")
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        median = np.median(mass_ratio)
        if median < 1:
            # define mass_ratio so that q>1
             for i in data:
                 i[mass_ratio_ind] = 1/i[mass_ratio_ind]
    if "chirp_mass" not in parameters and "total_mass" in parameters:
        parameters.append("chirp_mass")
        total_mass_ind = parameters.index("total_mass")
        mass_ratio_ind = parameters.index("mass_ratio")
        total_mass = np.array([i[total_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        chirp_mass = _mchirp_from_mtotal_q(total_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(chirp_mass[num])
    if "mass_1" not in parameters and "chirp_mass" in parameters:
        parameters.append("mass_1")
        chirp_mass_ind = parameters.index("chirp_mass")
        mass_ratio_ind = parameters.index("mass_ratio")
        chirp_mass = np.array([i[chirp_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        mass_1 = _m1_from_mchirp_q(chirp_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_1[num])
    if "mass_2" not in parameters and "chirp_mass" in parameters:
        parameters.append("mass_2")
        chirp_mass_ind = parameters.index("chirp_mass")
        mass_ratio_ind = parameters.index("mass_ratio")
        chirp_mass = np.array([i[chirp_mass_ind] for i in data])
        mass_ratio = np.array([i[mass_ratio_ind] for i in data])
        mass_2 = _m2_from_mchirp_q(chirp_mass, mass_ratio)
        for num, i in enumerate(data):
            data[num].append(mass_2[num])

    ##################################################
    # True reference frequency needs to be extracted #
    #              NEEDS TO BE FIXED                 #
    ##################################################

    if "reference_frequency" not in parameters:
        parameters.append("reference_frequency")
        for num, i in enumerate(data):
            data[num].append(20.)

    if "mass_1" in parameters and "mass_2" in parameters:
        mass1_ind = parameters.index("mass_1")
        mass1 = np.array([i[mass1_ind] for i in data])
        mass2_ind = parameters.index("mass_2")
        mass2 = np.array([i[mass2_ind] for i in data])
        if "total_mass" not in parameters:
            parameters.append("total_mass")
            m_total = _m_total_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(m_total[num])
        if "chirp_mass" not in parameters:
            parameters.append("chirp_mass")
            mchirp = _mchirp_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(mchirp[num])
        if "symmetric_mass_ratio" not in parameters:
            parameters.append("symmetric_mass_ratio")
            eta = _eta_from_m1_m2(mass1, mass2)
            for num, i in enumerate(data):
                data[num].append(eta[num])

        spin_components = ["spin_1x", "spin_1y", "spin_1z", "spin_2x",
                           "spin_2y", "spin_2z"]
        spin_angles = ["iota", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2",
                       "mass_1", "mass_2", "reference_frequency", "phase"]
        if all(i not in parameters for i in spin_components):
            if all(i in parameters for i in spin_angles):
                parameters.append("spin_1x")
                parameters.append("spin_1y")
                parameters.append("spin_1z")
                parameters.append("spin_2x")
                parameters.append("spin_2y")
                parameters.append("spin_2z")
                indices = [parameters.index("%s" %(i)) for i in spin_angles]
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
        if "chi_p" not in parameters and "chi_eff" not in parameters:
            if all(i in parameters for i in spin_angles):
                parameters.append("chi_p")
                parameters.append("chi_eff")
                indices = [parameters.index("%s" %(i)) for i in spin_components]
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
    if "cos_tilt_1" not in parameters and "tilt_1" in parameters:
        parameters.append("cos_tilt_1")
        tilt_1_ind = parameters.index("tilt_1")
        tilt_1 = np.array([i[tilt_1_ind] for i in data])
        cos_tilt_1 = np.cos(tilt_1)
        for num, i in enumerate(data):
            data[num].append(cos_tilt_1[num])
    if "cos_tilt_2" not in parameters and "tilt_2" in parameters:
        parameters.append("cos_tilt_2")
        tilt_2_ind = parameters.index("tilt_2")
        tilt_2 = np.array([i[tilt_2_ind] for i in data])
        cos_tilt_2 = np.cos(tilt_2)
        for num, i in enumerate(data):
            data[num].append(cos_tilt_2[num])
    if "luminosity_distance" not in parameters and "redshift" in parameters:
        parameters.append("luminosity_distance")
        redshift_ind = parameters.index("redshift")
        redshift = np.array([i[redshift_ind] for i in data])
        luminosity_distance = _dL_from_z(redshift)
        for num, i in enumerate(data):
            data[num].append(luminosity_distance[num])
    if "redshift" not in parameters and "luminosity_distance" in parameters:
        parameters.append("redshift")
        luminosity_distance_ind = parameters.index("luminosity_distance")
        luminosity_distance = np.array([i[luminosity_distance_ind] for i in data])
        redshift = _z_from_dL(luminosity_distance)
        for num, i in enumerate(data):
            data[num].append(redshift[num])
    if "comoving_distance" not in parameters and "redshift" in parameters:
        parameters.append("comoving_distance")
        redshift_ind = parameters.index("redshift")
        redshift = np.array([i[redshift_ind] for i in data])
        comoving_distance = _comoving_distance_from_z(redshift)
        for num, i in enumerate(data):
            data[num].append(comoving_distance[num])
    if "redshift" in parameters:
        redshift_ind = parameters.index("redshift")
        redshift = np.array([i[redshift_ind] for i in data])
        if "mass_1_source" not in parameters and "mass_1" in parameters:
            parameters.append("mass_1_source")
            mass_1_ind = parameters.index("mass_1")
            mass_1 = np.array([i[mass_1_ind] for i in data])
            mass_1_source = _m1_source_from_m1_z(mass_1, redshift)
            for num, i in enumerate(data):
                data[num].append(mass_1_source[num])
        if "mass_2_source" not in parameters and "mass_2" in parameters:
            parameters.append("mass_2_source")
            mass_2_ind = parameters.index("mass_2")
            mass_2 = np.array([i[mass_2_ind] for i in data])
            mass_2_source = _m2_source_from_m2_z(mass_2, redshift)
            for num, i in enumerate(data):
                data[num].append(mass_2_source[num])
        if "total_mass_source" not in parameters and "total_mass" in parameters:
            parameters.append("total_mass_source")
            total_mass_ind = parameters.index("total_mass")
            total_mass = np.array([i[total_mass_ind] for i in data])
            total_mass_source = _m_total_source_from_mtotal_z(total_mass, redshift)
            for num, i in enumerate(data):
                data[num].append(total_mass_source[num])
        if "chirp_mass_source" not in parameters and "chirp_mass" in parameters:
            parameters.append("chirp_mass_source")
            chirp_mass_ind = parameters.index("chirp_mass")
            chirp_mass = np.array([i[chirp_mass_ind] for i in data])
            chirp_mass_source = _mchirp_source_from_mchirp_z(chirp_mass, redshift)
            for num, i in enumerate(data):
                data[num].append(chirp_mass_source[num])
    return data, parameters

def one_format(fil, inj):
    """Looks at the input file and puts it into a standard form such that all
    parameter estimation codes can use the summary pages.

    Parameters
    ----------
    fil: str
        path to the results file location
    inj: str
        path to the location of the injection file
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
                          "logl": "log_likelihood",
                          "tilt1": "tilt_1",
                          "tilt_spin1": "tilt_1",
                          "tilt2": "tilt_2",
                          "tilt_spin2": "tilt_2",
                          "costilt1": "cos_tilt_1",
                          "costilt2": "cos_tilt_2",
                          "redshift": "redshift",
                          "phi_jl": "phi_jl",
                          "l1_optimal_snr": "L1_optimal_snr",
                          "h1_optimal_snr": "H1_optimal_snr",
                          "v1_optimal_snr": "V1_optimal_snr",
                          "L1_optimal_snr": "L1_optimal_snr",
                          "H1_optimal_snr": "H1_optimal_snr",
                          "V1_optimal_snr": "V1_optimal_snr",
                          "E1_optimal_snr": "E1_optimal_snr",
                          "mc_source": "chirp_mass_source",
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
        approx = "none"
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
        if "luminosity_distance" not in parameters and "logdistance" in lalinference_names:
            parameters.append("luminosity_distance")
            for num, i in enumerate(f[data_path]):
                data[num].append(np.exp(i[index("logdistance")]))
        if "iota" not in parameters and "costheta_jn" in lalinference_names:
            parameters.append("iota")
            for num, i in enumerate(f[data_path]):
                data[num].append(np.arccos(i[index("costheta_jn")]))
    if BILBY:
        approx = "none"
        inj_par, inj_data = [], []
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
    if "reference_frequency" in parameters:
        index = parameters.index("reference_frequency")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    if "minimum_frequency" in parameters:
        index = parameters.index("minimum_frequency")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    if "logPrior" in parameters:
        index = parameters.index("logPrior")
        parameters.remove(parameters[index])
        for i in data:
            i.remove(i[index])
    if LALINFERENCE:
        inj_par, inj_data = get_injection_parameters(parameters, inj, 
                                                     LALINFERENCE=True)
    if BILBY:
        inj_par, inj_data = get_injection_parameters(parameters, fil, BILBY=True)
    _make_hdf5_file(fil, np.array(data), np.array(parameters, dtype="S"),
                    np.array([approx], dtype="S"),
                    inj_par = np.array(inj_par, dtype="S"),
                    inj_data = np.array(inj_data))
    return "%s_temp" %(fil)

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
    _q_func = _q_from_m1_m2
    _eta_func = _eta_from_m1_m2
    _M_func = _m_total_from_m1_m2
    func_map = {"chirp_mass": lambda inj:inj.mchirp,
                "symmetric_mass_ratio": lambda inj: inj.eta,
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
                "total_mass": lambda inj: inj.mass1+inj.mass2,
                "chi_p": lambda inj: _chi_p(inj.mass1, inj.mass2, inj.spin1x,
                                            inj.spin1y, inj.spin2x, inj.spin2y),
                "chi_eff": lambda inj: _chi_eff(inj.mass1, inj.mass2, inj.spin1z,
                                                inj.spin2z)}
    inj_par = parameters
    if LALINFERENCE:
        if inj_file == None:
            inj_data = [float("nan")]*len(parameters)
        else:
            if GLUE:
                xmldoc = ligolw_utils.load_filename(inj_file, contenthandler= \
                                     lsctables.use_in(ligolw.LIGOLWContentHandler))
                table=lsctables.SimInspiralTable.get_table(xmldoc)[0]
                inj_data = [func_map[i](table) if i in func_map.keys() else \
                            float("nan") for i in parameters]
            else:
                inj_data = [float("nan")]*len(parameters)
    if BILBY:
        try:
            f = deepdish.io.load(inj_file)
            inj_keys = f["injection_parameters"].keys()
            inj_data = [f["injection_parameters"][key] if key in inj_keys \
                          else float("nan") for key in parameters]
        except:
            inj_data = [float("nan")]*len(parameters)
    return inj_par, inj_data
