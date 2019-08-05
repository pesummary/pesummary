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

import pesummary
from pesummary.gw.file.formats.base_read import GWRead
from pesummary.core.file.formats.pesummary import PESummary as CorePESummary

import os
from glob import glob


class PESummary(CorePESummary):
    """This class handles the existing posterior_samples.h5 file

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    existing_file: str
        the path to the existing posterior_samples.h5 file
    existing_approximants: list
        list of approximants that have been used in the previous analysis
    existing_labels: list
        list of labels that have been used in the previous analysis
    existing_samples: nd list
        nd list of samples stored for each approximant used in the previous
        analysis
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(PESummary, self).__init__(path_to_results_file)
        load_kwargs = {
            "grab_data_from_dictionary": PESummary._grab_data_from_dictionary}
        self.load(self._grab_data_from_pesummary_file, **load_kwargs)

    @classmethod
    def load_file(cls, path):
        if os.path.isdir(path):
            files = glob(path + "/*")
            if "home.html" in files:
                path = glob(path + "/samples/posterior_samples*")[0]
            else:
                raise Exception(
                    "Unable to find a file called 'posterior_samples' in "
                    "the directory %s" % (path + "/samples"))
        if not os.path.isfile(path):
            raise Exception("%s does not exist" % (path))
        return cls(path)

    @staticmethod
    def _grab_data_from_dictionary(dictionary):
        """
        """
        labels = list(dictionary["posterior_samples"].keys())
        existing_structure = {
            i: j for i in labels for j in
            dictionary["posterior_samples"]["%s" % (i)].keys()}
        labels = list(existing_structure.keys())

        parameter_list, sample_list, approx_list, inj_list = [], [], [], []
        for num, i in enumerate(labels):
            p = [j for j in dictionary["posterior_samples"]["%s" % (i)]["parameter_names"]]
            s = [j for j in dictionary["posterior_samples"]["%s" % (i)]["samples"]]
            if "injection_data" in dictionary.keys():
                inj = [j for j in dictionary["injection_data"]["%s" % (i)]["injection_values"]]
                inj_list.append({i: j for i, j in zip(p, inj)})
            if isinstance(p[0], bytes):
                parameter_list.append([j.decode("utf-8") for j in p])
            else:
                parameter_list.append([j for j in p])
            sample_list.append(s)
            psd, cal, config = None, None, None
            if "config_file" in dictionary.keys():
                config, = GWRead.load_recusively("config_file", dictionary)
            if "psds" in dictionary.keys():
                psd, = GWRead.load_recusively("psds", dictionary)
            if "calibration_envelope" in dictionary.keys():
                cal, = GWRead.load_recusively("calibration_envelope", dictionary)
            if "approximant" in dictionary.keys():
                approx_list.append(dictionary["approximant"]["%s" % (i)])
            else:
                approx_list.append(None)
        setattr(PESummary, "labels", labels)
        setattr(PESummary, "config", config)
        setattr(PESummary, "psd", psd)
        setattr(PESummary, "calibration", cal)
        setattr(PESummary, "approximant", approx_list)
        return parameter_list, sample_list, inj_list

    @property
    def calibration_data_in_results_file(self):
        if self.calibration:
            keys = [list(self.calibration[i].keys()) for i in self.labels]
            total = [[self.calibration[key][ifo] for ifo in keys[num]] for
                     num, key in enumerate(self.labels)]
            return total, keys
        return None

    @property
    def detectors(self):
        det_list = []
        for i in self.parameters:
            detectors = []
            for param in i:
                if "_optimal_snr" in param and param != "network_optimal_snr":
                    detectors.append(param.split("_optimal_snr")[0])
            if detectors == []:
                detectors.append(None)
            det_list.append(detectors)
        return det_list

    def to_bilby(self):
        """Convert a PESummary metafile to a bilby results object
        """
        from bilby.gw.result import CompactBinaryCoalescenceResult
        from pandas import DataFrame

        objects = {}
        for num, i in enumerate(self.labels):
            posterior_data_frame = DataFrame(
                self.samples[num], columns=self.parameters[num])
            meta_data = {
                "likelihood": {
                    "waveform_arguments": {
                        "waveform_approximant": self.approximant[num]},
                    "interferometers": self.detectors[num]}}
            bilby_object = CompactBinaryCoalescenceResult(
                search_parameter_keys=self.existing_parameters[num],
                posterior=posterior_data_frame, label="pesummary_%s" % (i),
                samples=self.existing_samples[num],
                meta_data=meta_data)
            objects[i] = bilby_object
        return objects

    def to_lalinference(self, outdir="./"):
        """Save a PESummary metafile as a lalinference hdf5 file
        """
        import numpy as np
        import h5py

        for num, i in enumerate(self.existing_labels):
            lalinference_samples = np.array(
                [tuple(i) for i in self.existing_samples[num]],
                dtype=[(i, '<f4') for i in self.existing_parameters[num]])

            try:
                f = h5py.File("%s/lalinference_file_%s.hdf5" % (outdir, i), "w")
            except Exception:
                raise Exception("Please make sure you have write permission in "
                                "%s" % (outdir))
            lalinference = f.create_group("lalinference")
            sampler = lalinference.create_group("lalinference_sampler")
            samples = sampler.create_dataset(
                "posterior_samples", data=lalinference_samples)
            f.close()
