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
import h5py
from scipy import interpolate

from pesummary.gw.file.standard_names import standard_names


class LALInferenceResultsFile(object):
    """Class to handle LALInference results format

    Attributes
    ----------
    path: str
        the path to the posterior samples stored in the hdf5 file
    """
    def __init__(self, f):
        self.fil = f
        self.path = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._data_structure = []
        f = h5py.File(self.fil)
        f.visit(self._add_to_list)
        for i in self._data_structure:
            condition1 = "posterior_samples" in i or "posterior" in i
            condition2 = "posterior_samples/" not in i and "posterior/" not in i
            if condition1 and condition2:
                self._path = i
        f.close()

    def _add_to_list(self, item):
        self._data_structure.append(item)

    def _lalinference_parameters_and_samples(self):
        """Grab the parameters and samples stored in file
        """
        samples = []
        f = h5py.File(self.fil, "r")
        lalinference_names = f[self.path].dtype.names
        parameters = [i for i in lalinference_names]
        for i in f[self.path]:
            samples.append(
                [i[lalinference_names.index(j)] for j in lalinference_names])
        return parameters, samples

    def grab_samples(self):
        """Return the posterior samples stored in the results file
        """
        samples = []
        f = h5py.File(self.fil, "r")
        lalinference_names = f[self.path].dtype.names
        parameters = [
            i for i in lalinference_names if i in standard_names.keys()]
        for i in f[self.path]:
            samples.append(
                [i[lalinference_names.index(j)] for j in parameters])
        parameters = [standard_names[i] for i in parameters]

        condition1 = "luminosity" not in parameters
        condition2 = "logdistance" in lalinference_names
        if condition1 and condition2:
            parameters.append("luminosity_distance")
            for num, i in enumerate(f[self.path]):
                samples[num].append(
                    np.exp(i[lalinference_names.index("logdistance")]))
        if "theta_jn" not in parameters and "costheta_jn" in lalinference_names:
            parameters.append("theta_jn")
            for num, i in enumerate(f[self.path]):
                samples[num].append(
                    np.arccos(i[lalinference_names.index("costheta_jn")]))
        f.close()
        return parameters, samples

    def grab_calibration_data(self):
        """Return the calibration data stored in the results file
        """
        parameters, samples = self._lalinference_parameters_and_samples()
        f = h5py.File(self.fil)
        attributes = f[self.path].attrs.items()

        log_frequencies = {
            key.split("_")[0]: [] for key, value in attributes if
            "_spcal_logfreq" in key}
        for key, value in attributes:
            if "_spcal_logfreq" in key:
                log_frequencies[key.split("_")[0]].append(float(value))

        keys_amp = np.sort([
            param for param in parameters if "_spcal_amp" in param])
        keys_phase = np.sort([
            param for param in parameters if "_spcal_phase" in
            param])

        amp_params = {ifo: [] for ifo in log_frequencies.keys()}
        phase_params = {ifo: [] for ifo in log_frequencies.keys()}
        for key in keys_amp:
            ifo = key.split("_")[0]
            ind = parameters.index(key)
            amp_params[ifo].append([float(i[ind]) for i in samples])
        for key in keys_phase:
            ifo = key.split("_")[0]
            ind = parameters.index(key)
            phase_params[ifo].append([float(i[ind]) for i in samples])
        f.close()

        total = []
        for key in log_frequencies.keys():
            f = np.exp(log_frequencies[key])
            fs = np.linspace(np.min(f), np.max(f), 100)
            data = [interpolate.spline(log_frequencies[key], samp, np.log(fs)) for samp
                    in np.column_stack(amp_params[key])]
            amplitude_upper = 1. - np.percentile(data, 90, axis=0)
            amplitude_lower = 1. - np.percentile(data, 10, axis=0)
            amplitude_median = 1. - np.median(data, axis=0)

            data = [interpolate.spline(log_frequencies[key], samp, np.log(fs)) for samp
                    in np.column_stack(phase_params[key])]

            phase_upper = np.percentile(data, 90, axis=0)
            phase_lower = np.percentile(data, 10, axis=0)
            phase_median = np.median(data, axis=0)
            total.append(np.column_stack(
                [fs, amplitude_median, phase_median, amplitude_lower,
                 phase_lower, amplitude_upper, phase_upper]))
        return total, list(log_frequencies.keys())
