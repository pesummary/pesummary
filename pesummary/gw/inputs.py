#! /usr/bin/env python

# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org> This program is free
# software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy as np
import h5py

from time import time

import pesummary
from pesummary.utils.utils import logger
from pesummary.gw.file.one_format import GWOneFormat
from pesummary.gw.file.existing import GWExistingFile
from pesummary.gw.file.lalinference import LALInferenceResultsFile
from pesummary.core.inputs import Input

__doc__ == "Classes to handle the command line inputs"


class GWInput(Input):
    """Super class to handle command line arguments

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    user: str
        The user who submitted the job
    add_to_existing: Bool
        Boolean to determine if you wish to add to a existing webpage
    existing: str
        Existing web directory
    webdir: str
        Directory to output all information
    baseurl: str
        The url path for the corresponding web directory
    inj_file: List
        List containing paths to the injection file
    result_files: list
        List containing paths to the results files which are being analysed
    config: list
        List containing paths to the configuration files used to generate each
        results files
    approximant: list
        List of approximants used in the analysis to generate each results
        files
    email: str
        The email address to notify when the job has completed
    sensitivity: Bool
        Boolean to determine if you wish to plot the sky sensitivity for
        different detector networks
    gracedb: str
        The gracedb of the event that produced the results files
    dump: Bool
        Boolean to determine if you wish to produce a dumped html page layout
    detectors: list
        List containing the detectors used to generate each results file
    labels: str
        A label for this summary page
    psd: str
        List of the psds used in the analysis
    """
    def __init__(self, opts):
        logger.info("Command line arguments: %s" % (opts))
        self.user = opts.user
        self.existing = opts.existing
        self.webdir = opts.webdir
        self.baseurl = opts.baseurl
        self.inj_file = opts.inj_file
        self.config = opts.config
        self.result_files = opts.samples
        self.email = opts.email
        self.add_to_existing = opts.add_to_existing
        self.dump = opts.dump
        self.hdf5 = opts.save_to_hdf5
        self.gracedb = opts.gracedb
        self.detectors = None
        self.calibration = opts.calibration
        self.approximant = opts.approximant
        self.sensitivity = opts.sensitivity
        self.psds = opts.psd
        self.existing_labels = []
        self.existing_parameters = []
        self.existing_samples = []
        self.existing_approximant = []
        self.make_directories()
        self.copy_files()
        self.labels = opts.labels

    @property
    def approximant(self):
        return self._approximant

    @approximant.setter
    def approximant(self, approximant):
        approximant_list = [None] * len(self.result_files)
        if not approximant:
            logger.warning("No approximant given. Waveform plots will not be "
                           "generated")
        else:
            approximant_list = approximant
        if approximant_list and len(approximant_list) != len(self.result_files):
            raise Exception("The number of results files does not match the "
                            "number of approximants")
        self._approximant = approximant_list

    @property
    def gracedb(self):
        return self._gracedb

    @gracedb.setter
    def gracedb(self, gracedb):
        self._gracedb = None
        if gracedb:
            self._gracedb = gracedb

    @property
    def detectors(self):
        return self._detectors

    @detectors.setter
    def detectors(self, detectors):
        detector_list = []
        if not detectors:
            for num, i in enumerate(self.result_files):
                params = self.parameters[num]
                individual_detectors = []
                for j in params:
                    if "optimal_snr" in j:
                        det = j.split("_optimal_snr")[0]
                        individual_detectors.append(det)
                individual_detectors = sorted(
                    [str(i) for i in individual_detectors])
                if individual_detectors:
                    detector_list.append("_".join(individual_detectors))
                else:
                    detector_list.append(None)
        else:
            detector_list = detectors
        logger.debug("The detector network is %s" % (detector_list))
        self._detectors = detector_list

    @property
    def psds(self):
        return self._psds

    @psds.setter
    def psds(self, psds):
        psd_list = []
        if psds:
            if isinstance(psds, dict):
                keys = psds.keys()
                for key in keys:
                    if isinstance(psds[key], str):
                        self._check_psd_extension(psds[key])
                    else:
                        for j in psds[key]:
                            self._check_psd_extension(j)
                psd_list = psds
            else:
                for i in psds:
                    self._check_psd_extension(i)
                    psd_list.append(i)
            self._psds = psd_list
        else:
            self._psds = None

    def _check_psd_extension(self, file):
        """Check that the file extension on the psd file can be read and
        understood by PESummary.

        Parameters
        ----------
        file: str
            path to the file that you would like to check
        """
        extension = file.split(".")[-1]
        if extension == "gz":
            print("convert to .dat file")
        elif extension == "dat":
            pass
        elif extension == "txt":
            pass
        else:
            raise Exception("PSD results file not understood")

    @property
    def calibration(self):
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        calibration_list = []
        if calibration:
            labels = None
            if isinstance(calibration, dict):
                keys = calibration.keys()
                for key in keys:
                    if isinstance(calibration[key], str):
                        self._check_calibration_file(calibration[key])
                    else:
                        for j in calibration[key]:
                            self._check_calibration_file(j)
                calibration_list = calibration
            else:
                for i in calibration:
                    self._check_calibration_file(i)
                    calibration_list.append(i)
            self._calibration = calibration_list
            self.calibration_labels = labels
        else:
            logger.debug("No calibration envelope file given. Checking the "
                         "results file")
            self._calibration_envelope = False
            label_list = []
            for i in self.result_files:
                try:
                    f = LALInferenceResultsFile(i)
                    data, labels = f.grab_calibration_data()
                    if data == []:
                        logger.info("Failed to grab calibration data from %s" % (i))
                        data, labels = None, None
                except Exception:
                    logger.info("Failed to grab calibration data from %s" % (i))
                    data, labels = None, None
                label_list.append(labels)
            calibration_list = data
            self._calibration = calibration_list
            self.calibration_labels = label_list

    def _check_calibration_file(self, file):
        """Check the contents of the calibration file to ensure that it is
        of the correct format.

        Parameters
        ----------
        file: str
            path to the calibration file
        """
        f = np.genfromtxt(file)
        if len(f[0]) != 7:
            raise Exception("Calibration envelope file not understood")
        pass

    @property
    def existing_approximant(self):
        return self._existing_approximant

    @existing_approximant.setter
    def existing_approximant(self, existing_approximant):
        self._existing_approximant = None
        if self.add_to_existing:
            existing = GWExistingFile(self.existing)
            self._existing_approximant = existing.existing_approximant

    @staticmethod
    def _IFO_from_file_name(file):
        """Return a guess of the IFO from the file name.

        Parameters
        ----------
        file: str
            the name of the file that you would like to make a guess for
        """
        file_name = file.split("/")[-1]
        if any(j in file_name for j in ["H1", "_0", "IFO0"]):
            ifo = "H1"
        elif any(j in file_name for j in ["L1", "_1", "IFO1"]):
            ifo = "L1"
        elif any(j in file_name for j in ["V1", "_2", "IFO2"]):
            ifo = "V1"
        else:
            ifo = file_name
        return ifo

    def convert_to_standard_format(self, results_file, injection_file=None,
                                   config_file=None):
        """Convert a results file to standard form.

        Parameters
        ----------
        results_file: str
            Path to the results file that you wish to convert to standard
            form
        injection_file: str, optional
            Path to the injection file that was used in the analysis to
            produce the results_file
        config_file: str, optional
            Path to the configuration file that was used
        """
        f = GWOneFormat(results_file, injection_file, config=config_file)
        f.generate_all_posterior_samples()
        parameters = f.parameters
        samples = f.samples
        inj_p = f.injection_parameters
        inj_value = f.injection_values
        injection = {i: j for i, j in zip(inj_p, inj_value)}
        return parameters, samples, injection

    def _default_labels(self):
        """Return the defaut labels given your detector network.
        """
        label_list = []
        for num, i in enumerate(self.result_files):
            if self.gracedb and self.detectors[num]:
                label_list.append("_".join(
                    [self.gracedb, self.detectors[num]]))
            elif self.gracedb:
                label_list.append(self.gracedb)
            elif self.detectors[num]:
                label_list.append(self.detectors[num])
            else:
                file_name = ".".join(i.split("/")[-1].split(".")[:-1])
                label_list.append("%s_%s" % (round(time()), file_name))

        duplicates = dict(set(
            (x, label_list.count(x)) for x in
            filter(lambda rec: label_list.count(rec) > 1, label_list)))
        for i in duplicates.keys():
            for j in range(duplicates[i]):
                ind = label_list.index(i)
                label_list[ind] += "_%s" % (j)
        if self.add_to_existing:
            for num, i in enumerate(label_list):
                if i in self.existing_labels:
                    ind = label_list.index(i)
                    label_list[ind] += "_%s" % (num)
        return label_list


class GWPostProcessing(pesummary.core.inputs.PostProcessing):
    """Class to extract parameters from the results files

    Parameters
    ----------
    inputs: argparser
        The parser containing the command line arguments
    colors: list, optional
        colors that you would like to use to display each results file in the
        webpage

    Attributes
    ----------
    parameters: list
        list of parameters that have posterior distributions for each results
        file
    injection_data: list
        list of dictionaries that contain the injection parameters and their
        injected value for each results file
    samples: list
        list of posterior samples for each parameter for each results file
    maxL_samples: list
        list of dictionaries that contain each parameter and their
        corresponding maximum likelihood value for each results file
    same_parameters: list
        List of parameters that all results files have sampled over
    """
    def __init__(self, inputs, colors="default"):
        self.inputs = inputs
        self.webdir = inputs.webdir
        self.baseurl = inputs.baseurl
        self.result_files = inputs.result_files
        self.dump = inputs.dump
        self.email = inputs.email
        self.user = inputs.user
        self.host = inputs.host
        self.config = inputs.config
        self.existing = inputs.existing
        self.add_to_existing = inputs.add_to_existing
        self.labels = inputs.labels
        self.hdf5 = inputs.hdf5
        self.existing_labels = inputs.existing_labels
        self.existing_parameters = inputs.existing_parameters
        self.existing_samples = inputs.existing_samples
        self.existing_meta_file = inputs.existing_meta_file
        self.colors = colors
        self.approximant = inputs.approximant
        self.detectors = inputs.detectors
        self.gracedb = inputs.gracedb
        self.calibration = inputs.calibration
        self.calibration_labels = inputs.calibration_labels
        self.sensitivity = inputs.sensitivity
        self.psds = inputs.psds
        self.psd_dict = False
        self.psd_list = False
        if isinstance(self.psds, dict):
            self.psd_dict = True
        else:
            self.psd_list = True
        self.calibration_dict = False
        self.calibration_list = False
        if isinstance(self.calibration, dict):
            self.calibration_dict = True
        else:
            self.calibration_list = True

        self.grab_data_map = {"existing_file": self._data_from_existing_file,
                              "standard_format": self._data_from_standard_format}

        self.parameters = inputs.parameters
        self.samples = inputs.samples
        self.injection_data = inputs.injection_data
        self.maxL_samples = []
        self.same_parameters = []

    @property
    def coherence_test(self):
        return False

    @property
    def maxL_samples(self):
        return self._maxL_samples

    @maxL_samples.setter
    def maxL_samples(self, maxL_samples):
        key_data = self._key_data()
        maxL_list = []
        for num, i in enumerate(self.parameters):
            dictionary = {j: key_data[num][j]["maxL"] for j in i}
            if self.approximant:
                dictionary["approximant"] = self.approximant[num]
            maxL_list.append(dictionary)
        self._maxL_samples = maxL_list

    @property
    def label_to_prepend_approximant(self):
        labels = [i[len(self.gracedb) + 1:] if self.gracedb else i for i in
                  self.labels]
        prepend = [None] * len(self.approximant)
        duplicates = dict(set(
            (x, self.approximant.count(x)) for x in filter(
                lambda rec: self.approximant.count(rec) > 1,
                self.approximant)))
        if len(duplicates.keys()) > 0:
            for num, i in enumerate(self.approximant):
                if i in duplicates.keys():
                    prepend[num] = labels[num]
        return prepend

    @property
    def psd_labels(self):
        if self.psds:
            return self._labels_from_dictionary(self.psds)
        return None

    @property
    def calibration_labels(self):
        return self._calibration_labels

    @calibration_labels.setter
    def calibration_labels(self, calibration_labels):
        if not calibration_labels and self.calibration:
            self._calibration_labels = self._labels_from_dictionary(
                self.calibration)
        elif self.calibration:
            self._calibration_labels = calibration_labels
        else:
            self._calibration_labels = None

    @property
    def psd_frequencies(self):
        return self._setup_psd_calibration(
            self.psds, self.psd_labels,
            self._grab_frequencies_from_psd_data_file)

    @property
    def psd_strains(self):
        return self._setup_psd_calibration(
            self.psds, self.psd_labels, self._grab_strains_from_psd_data_file)

    @property
    def calibration_envelopes(self):
        return self._setup_psd_calibration(
            self.calibration, self.calibration_labels,
            self._grab_calibration_data_from_data_file)

    @staticmethod
    def _IFO_from_file_name(file):
        """Return a guess of the IFO from the file name.

        Parameters
        ----------
        file: str
            the name of the file that you would like to make a guess for
        """
        file_name = file.split("/")[-1]
        if any(j in file_name for j in ["H1", "_0", "IFO0"]):
            ifo = "H1"
        elif any(j in file_name for j in ["L1", "_1", "IFO1"]):
            ifo = "L1"
        elif any(j in file_name for j in ["V1", "_2", "IFO2"]):
            ifo = "V1"
        else:
            ifo = file_name
        return ifo

    def _setup_psd_calibration(self, data, labels, executable):
        """Determine which result files correspond to which calibration/psd
        files.

        Parameters
        ----------
        data: list/dict
            list/dict containing paths to calibration/psd files
        labels: list
            list of labels corresponding to the calibration/psd files
        executable: PESummary object
            executable that is used to extract the data from the calibration/psd
            files
        """
        output = []
        if not isinstance(data, dict):
            for i in labels:
                temp = [executable(i) for i in data]
                output.append(temp)
        else:
            keys = list(data.keys())
            if isinstance(data[keys[0]], list):
                for idx in range(len(data[keys[0]])):
                    temp = [executable(data[i][idx]) for i in list(keys)]
                    output.append(temp)
            else:
                for i in labels:
                    temp = [executable(data[i]) for i in list(keys)]
                    output.append(temp)
        return output

    def _labels_from_dictionary(self, input):
        """Return the labels from either a list of a dictionary input

        Parameters
        ----------
        input: list/dict
            list/dict containing paths to files
        """
        if isinstance(input, dict):
            keys = list(input.keys())
            if isinstance(input[keys[0]], list):
                return [[i for i in input.keys()] for j in range(len(self.labels))]
            else:
                return [[i for i in input.keys()] for j in input]
        return [[self._IFO_from_file_name(i) for i in input] for j in
                range(len(self.labels))]

    def _key_data(self):
        """Grab the mean, median, maximum likelihood value and the standard
        deviation of each posteiror distribution for each results file.
        """
        key_data_list = []
        for num, i in enumerate(self.samples):
            data = {}
            likelihood_ind = self.parameters[num].index("log_likelihood")
            logL = [j[likelihood_ind] for j in i]
            for ind, j in enumerate(self.parameters[num]):
                index = self.parameters[num].index("%s" % (j))
                subset = [k[index] for k in i]
                data[j] = {"mean": np.mean(subset),
                           "median": np.median(subset),
                           "maxL": subset[logL.index(np.max(logL))],
                           "std": np.std(subset)}
            key_data_list.append(data)
        return key_data_list

    def _grab_frequencies_from_psd_data_file(self, file):
        """Return the frequencies stored in the psd data files

        Parameters
        ----------
        file: str
            path to the psd data file
        """
        fil = open(file)
        fil = fil.readlines()
        fil = [i.strip().split() for i in fil]
        return [float(i[0]) for i in fil]

    def _grab_strains_from_psd_data_file(self, file):
        """Return the strains stored in the psd data files

        Parameters
        ----------
        file: str
            path to the psd data file
        """
        fil = open(file)
        fil = fil.readlines()
        fil = [i.strip().split() for i in fil]
        return [float(i[1]) for i in fil]

    def _grab_calibration_data_from_data_file(self, file):
        """Return the data stored in the calibration data file

        Parameters
        ----------
        file: str
            path to the calibration data file
        """
        try:
            f = np.genfromtxt(file)
            return f
        except Exception:
            return file
