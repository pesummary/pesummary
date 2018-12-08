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

import numpy as np

def _make_hdf5_file(name, data, parameters):
    """
    """
    f = h5py.File("%s_temp" %(name), "w")
    f.create_dataset("parameter_names", data=parameters)
    f.create_dataset("samples", data=data)
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

def _eta_from_m1_m2(mass1, mass2):
    """Return the symmetric mass ratio given the samples for mass1 and mass2
    """
    return (mass1 * mass2) / (mass1 + mass2)**2

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
        sampler = [i for i in f["lalinference"].keys()]
        data_path = "lalinference/%s/posterior_samples" %(sampler[0])
        parameters = np.array([i for i in f[data_path].dtype.names])
        data = np.array([i for i in f[data_path]])
    if BILBY:
        parameters, data = [], []
        for i in sorted(f["%s" %(path)].keys()):
            if "block2" in i:
                pass
            else:
                if "items" in i:
                    for par in f["%s/%s" %(path,i)]:
                        parameters.append(par)
                if "values" in i:
                    if len(data) == 0:
                        for dat in f["%s/%s" %(path, i)]:
                            data.append(list(np.real(dat)))
                    else:
                        for num, dat in enumerate(f["%s/%s" %(path, i)]):
                            data[num] += list(np.real(dat))
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
        mass1_ind = parameters.index("mass_1")
        mass1 = np.array([i[mass1_ind] for i in data])
        mass2_ind = parameters.index("mass_2")
        mass2 = np.array([i[mass2_ind] for i in data])
        if "total_mass" not in parameters:
            parameters = ["total_mass"] + parameters
            m_total = _m_total_from_m1_m2(mass1, mass2)
            if "chirp_mass" not in parameters:
                parameters = ["chirp_mass"] + parameters
                m_chirp = _mchirp_from_m1_m2(mass1, mass2)
                for i in np.arange(len(data)):
                    data[i] = [m_chirp[i], m_total[i]] + data[i]
            else:
                for i in np.arange(len(data)):
                    data[i] = [m_total[i]] + data[i]
    _make_hdf5_file(fil, np.array(data), np.array(parameters))
    return "%s_temp" %(fil)
