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
import sys
import logging
import contextlib
import time

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import h5py

from tqdm import tqdm as Basetqdm


class SamplesDict(dict):
    """Class to store the samples from a single run

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter
    """
    def __init__(self, parameters, samples):
        super(SamplesDict, self).__init__()
        self.parameters = parameters
        self.samples = samples
        lengths = [len(i) for i in samples]
        if len(np.unique(lengths)) > 1:
            raise Exception("Unequal number of samples for each parameter")
        self.make_dictionary()

    def __getitem__(self, key):
        """Return an object representing the specialization of SamplesDict
        by type arguments found in key.
        """
        if isinstance(key, slice):
            return SamplesDict(
                self.parameters,
                [i[key.start:key.stop:key.step] for i in self.samples]
            )
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(
                    "{} not in dictionary. The list of available keys are "
                    "{}".format(key, self.keys())
                )
        return super(SamplesDict, self).__getitem__(key)

    def __str__(self):
        """Print a summary of the information stored in the dictionary
        """
        def format_string(string, row):
            """Format a list into a table

            Parameters
            ----------
            string: str
                existing table
            row: list
                the row you wish to be written to a table
            """
            string += "{:<8}".format(row[0])
            for i in range(1, len(row)):
                if isinstance(row[i], str):
                    string += "{:<15}".format(row[i])
                elif isinstance(row[i], (float, int, np.int64, np.int32)):
                    string += "{:<15.6f}".format(row[i])
            string += "\n"
            return string

        string = ""
        string = format_string(string, ["idx"] + list(self.keys()))

        if self.number_of_samples < 8:
            for i in range(self.number_of_samples):
                string = format_string(
                    string, [i] + [item[i] for key, item in self.items()]
                )
        else:
            for i in range(4):
                string = format_string(
                    string, [i] + [item[i] for key, item in self.items()]
                )
            for i in range(2):
                string = format_string(string, ["."] * (len(self.keys()) + 1))
            for i in range(self.number_of_samples - 2, self.number_of_samples):
                string = format_string(
                    string, [i] + [item[i] for key, item in self.items()]
                )
        return string

    @property
    def maxL(self):
        return SamplesDict(
            self.parameters, [[item.maxL] for key, item in self.items()]
        )

    @property
    def minimum(self):
        return SamplesDict(
            self.parameters, [[item.minimum] for key, item in self.items()]
        )

    @property
    def maximum(self):
        return SamplesDict(
            self.parameters, [[item.maximum] for key, item in self.items()]
        )

    @property
    def median(self):
        return SamplesDict(
            self.parameters,
            [[item.average(type="median")] for key, item in self.items()]
        )

    @property
    def mean(self):
        return SamplesDict(
            self.parameters,
            [[item.average(type="mean")] for key, item in self.items()]
        )

    @property
    def number_of_samples(self):
        return len(self[self.parameters[0]])

    def to_pandas(self):
        """Convert a SamplesDict object to a pandas dataframe
        """
        from pandas import DataFrame

        return DataFrame(self)

    def to_structured_array(self):
        """Convert a SamplesDict object to a structured numpy array
        """
        return self.to_pandas().to_records(index=False, column_dtypes=np.float)

    def pop(self, parameter):
        """Delete a parameter from the SamplesDict

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to remove from the SamplesDict
        """
        if parameter not in self.parameters:
            logger.info(
                "{} not in SamplesDict. Unable to remove {}".format(
                    parameter, parameter
                )
            )
            return
        ind = self.parameters.index(parameter)
        self.parameters.remove(parameter)
        remove = self.samples[ind]
        if isinstance(self.samples, np.ndarray):
            samples = self.samples.tolist()
            remove = self.samples[ind].tolist()
        samples.remove(remove)
        if isinstance(self.samples, np.ndarray):
            self.samples = np.array(samples)
        return super(SamplesDict, self).pop(parameter)

    def downsample(self, number):
        """Downsample the samples stored in the SamplesDict class

        Parameters
        ----------
        number: int
            Number of samples you wish to downsample to
        """
        self.samples = resample_posterior_distribution(self.samples, number)
        self.make_dictionary()
        return self

    def discard_samples(self, number):
        """Remove the first n samples

        Parameters
        ----------
        number: int
            Number of samples that you wish to remove
        """
        self.make_dictionary(discard_samples=number)
        return self

    def make_dictionary(self, discard_samples=None):
        """Add the parameters and samples to the class
        """
        if "log_likelihood" in self.parameters:
            likelihoods = self.samples[self.parameters.index("log_likelihood")]
            likelihoods = likelihoods[discard_samples:]
        else:
            likelihoods = None
        if any(i in self.parameters for i in ["weights", "weight"]):
            ind = (
                self.parameters.index("weights") if "weights" in self.parameters
                else self.parameters.index("weight")
            )
            weights = self.samples[ind][discard_samples:]
        else:
            weights = None
        for key, val in zip(self.parameters, self.samples):
            self[key] = Array(
                val[discard_samples:], likelihood=likelihoods, weights=weights
            )


class Array(np.ndarray):
    """Class to add extra functions and methods to np.ndarray

    Parameters
    ----------
    input_aray: list/array
        input list/array

    Attributes
    ----------
    median: float
        median of the input array
    mean: float
        mean of the input array
    """
    __slots__ = ["standard_deviation", "minimum", "maximum", "maxL", "weights"]

    def __new__(cls, input_array, likelihood=None, weights=None):
        obj = np.asarray(input_array).view(cls)
        obj.standard_deviation = np.std(obj)
        obj.minimum = np.min(obj)
        obj.maximum = np.max(obj)
        obj.maxL = cls._maxL(obj, likelihood)
        obj.weights = weights
        return obj

    def __reduce__(self):
        pickled_state = super(Array, self).__reduce__()
        new_state = pickled_state[2] + tuple(
            [getattr(self, i) for i in self.__slots__]
        )
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.standard_deviation = state[-5]
        self.minimum = state[-4]
        self.maximum = state[-3]
        self.maxL = state[-2]
        self.weights = state[-1]
        super(Array, self).__setstate__(state[0:-5])

    def average(self, type="mean"):
        """Return the average of the array

        Parameters
        ----------
        type: str
            the method to average the array
        """
        if type == "mean":
            return self._mean(self, weights=self.weights)
        elif type == "median":
            return self._median(self, weights=self.weights)
        else:
            return None

    @staticmethod
    def _mean(array, weights=None):
        """Compute the mean from a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        """
        if weights is None:
            return np.mean(array)
        weights = np.array(weights).flatten() / float(sum(weights))
        return float(np.dot(np.array(array), weights))

    @staticmethod
    def _median(array, weights=None):
        """Compute the median from a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        """
        if weights is None:
            return np.median(array)
        return Array.percentile(array, weights=weights, percentile=0.5)

    @staticmethod
    def _maxL(array, likelihood=None):
        """Return the maximum likelihood value of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        likelihood: np.ndarray, optional
            likelihoods associated with each sample
        """
        if likelihood is not None:
            likelihood = list(likelihood)
            ind = likelihood.index(np.max(likelihood))
            return array[ind]
        return None

    @staticmethod
    def percentile(array, weights=None, percentile=None):
        """Compute the Nth percentile of a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        percentile: float, list
            list of percentiles to compute
        """
        if weights is None:
            return np.percentile(array, percentile)

        array, weights = np.array(array), np.array(weights)
        percentile_type = percentile
        if not isinstance(percentile, (list, np.ndarray)):
            percentile = [float(percentile)]
        percentile = np.array([float(i) for i in percentile])
        if not all(i < 1 for i in percentile):
            percentile *= 0.01
        ind_sorted = np.argsort(array)
        sorted_data = array[ind_sorted]
        sorted_weights = weights[ind_sorted]
        Sn = np.cumsum(sorted_weights)
        Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
        data = np.interp(percentile, Pn, sorted_data)
        if isinstance(percentile_type, (int, float, np.float64, np.float32)):
            return float(data[0])
        return data

    def confidence_interval(self, percentile=None):
        """Return the confidence interval of the array

        Parameters
        ----------
        percentile: int/list, optional
            Percentile or sequence of percentiles to compute, which must be
            between 0 and 100 inclusive
        """
        if percentile is not None:
            if isinstance(percentile, int):
                return self.percentile(self, self.weights, percentile)
            return np.array(
                [self.percentile(self, self.weights, i) for i in percentile]
            )
        return np.array(
            [self.percentile(self, self.weights, i) for i in [5, 95]]
        )

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.standard_deviation = getattr(obj, 'standard_deviation', None)
        self.minimum = getattr(obj, 'minimum', None)
        self.maximum = getattr(obj, 'maximum', None)
        self.maxL = getattr(obj, 'maxL', None)
        self.weights = getattr(obj, 'weights', None)


class _tqdm(Basetqdm):

    @property
    def format_dict(self):
        """Public API for read-only member access."""
        return dict(
            n=self.n, total=self.total,
            elapsed=self._time() - self.start_t
            if hasattr(self, 'start_t') else 0,
            asctime=time.strftime("%Y-%m-%d  %H:%M:%S"),
            ncols=self.dynamic_ncols(self.fp)
            if self.dynamic_ncols else self.ncols,
            prefix=self.desc, ascii=self.ascii, unit=self.unit,
            unit_scale=self.unit_scale,
            rate=1 / self.avg_time if self.avg_time else None,
            bar_format=self.bar_format, postfix=self.postfix,
            unit_divisor=self.unit_divisor)


def trange(*args, **kwargs):
    """
    A shortcut for tqdm(range(*args), **kwargs).
    """
    return _tqdm(range(*args), **kwargs)


def resample_posterior_distribution(posterior, nsamples):
    """Randomly draw nsamples from the posterior distribution

    Parameters
    ----------
    posterior: ndlist
        nd list of posterior samples. If you only want to resample one
        posterior distribution then posterior=[[1., 2., 3., 4.]]. For multiple
        posterior distributions then posterior=[[1., 2., 3., 4.], [1., 2., 3.]]
    nsamples: int
        number of samples that you wish to randomly draw from the distribution
    """
    if len(posterior) == 1:
        n, bins = np.histogram(posterior, bins=50)
        n = np.array([0] + [i for i in n])
        cdf = cumtrapz(n, bins, initial=0)
        cdf /= cdf[-1]
        icdf = interp1d(cdf, bins)
        samples = icdf(np.random.rand(nsamples))
    else:
        posterior = np.array([i for i in posterior])
        keep_idxs = np.random.choice(
            len(posterior[0]), nsamples, replace=False)
        samples = [i[keep_idxs] for i in posterior]
    return samples


def check_file_exists_and_rename(file_name):
    """Check to see if a file exists and if it does then rename the file

    Parameters
    ----------
    file_name: str
        proposed file name to store data
    """
    if os.path.isfile(file_name):
        import shutil

        old_file = "{}_old".format(file_name)
        while os.path.isfile(old_file):
            old_file += "_old"
        logger.warn(
            "The file '{}' already exists. Renaming the existing file to "
            "{} and saving the data to the requested file name".format(
                file_name, old_file
            )
        )
        shutil.move(file_name, old_file)


def check_condition(condition, error_message):
    """Raise an exception if the condition is not satisfied
    """
    if condition:
        raise Exception(error_message)


def rename_group_or_dataset_in_hf5_file(base_file, group=None, dataset=None):
    """Rename a group or dataset in an hdf5 file

    Parameters
    ----------
    group: list, optional
        a list containing the path to the group that you would like to change
        as the first argument and the new name of the group as the second
        argument
    dataset: list, optional
        a list containing the name of the dataset that you would like to change
        as the first argument and the new name of the dataset as the second
        argument
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The file %s does not exist" % (base_file))
    f = h5py.File(base_file, "a")
    if group:
        f[group[1]] = f[group[0]]
        del f[group[0]]
    elif dataset:
        f[dataset[1]] = f[dataset[0]]
        del f[dataset[0]]
    f.close()


def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


def guess_url(web_dir, host, user):
    """Guess the base url from the host name

    Parameters
    ----------
    web_dir: str
        path to the web directory where you want the data to be saved
    host: str
        the host name of the machine where the python interpreter is currently
        executing
    user: str
        the user that is current executing the python interpreter
    """
    ligo_data_grid = False
    if 'public_html' in web_dir:
        ligo_data_grid = True
    if ligo_data_grid:
        path = web_dir.split("public_html")[1]
        if "raven" in host or "arcca" in host:
            url = "https://geo2.arcca.cf.ac.uk/~{}".format(user)
        elif 'ligo-wa' in host:
            url = "https://ldas-jobs.ligo-wa.caltech.edu/~{}".format(user)
        elif 'ligo-la' in host:
            url = "https://ldas-jobs.ligo-la.caltech.edu/~{}".format(user)
        elif "cit" in host or "caltech" in host:
            url = "https://ldas-jobs.ligo.caltech.edu/~{}".format(user)
        elif 'uwm' in host or 'nemo' in host:
            url = "https://ldas-jobs.phys.uwm.edu/~{}".format(user)
        elif 'phy.syr.edu' in host:
            url = "https://sugar-jobs.phy.syr.edu/~{}".format(user)
        elif 'vulcan' in host:
            url = "https://galahad.aei.mpg.de/~{}".format(user)
        elif 'atlas' in host:
            url = "https://atlas1.atlas.aei.uni-hannover.de/~{}".format(user)
        elif 'iucca' in host:
            url = "https://ldas-jobs.gw.iucaa.in/~{}".format(user)
        elif 'hawk' in host:
            url = "https://ligo.gravity.cf.ac.uk/~{}".format(user)
        else:
            url = "https://{}/~{}".format(host, user)
        url += path
    else:
        url = "https://{}".format(web_dir)
    return url


def command_line_arguments():
    """Return the command line arguments
    """
    return sys.argv[1:]


def command_line_dict():
    """Return a dictionary of command line arguments
    """
    from pesummary.core.command_line import command_line
    from pesummary.gw.command_line import insert_gwspecific_option_group

    parser = command_line()
    insert_gwspecific_option_group(parser)
    opts = parser.parse_args()
    return vars(opts)


def gw_results_file(opts):
    """Determine if a GW results file is passed
    """
    cond1 = hasattr(opts, "gw") and opts.gw
    cond2 = hasattr(opts, "calibration") and opts.calibration
    cond3 = hasattr(opts, "gracedb") and opts.gracedb
    cond4 = hasattr(opts, "approximant") and opts.approximant
    cond5 = hasattr(opts, "psd") and opts.psd
    if cond1 or cond2 or cond3 or cond4 or cond5:
        return True
    else:
        return False


def functions(opts):
    """Return a dictionary of functions that are either specific to GW results
    files or core.
    """
    from pesummary.core.inputs import Input
    from pesummary.gw.inputs import GWInput
    from pesummary.core.file.meta_file import MetaFile
    from pesummary.gw.file.meta_file import GWMetaFile
    from pesummary.core.finish import FinishingTouches

    dictionary = {}
    dictionary["input"] = GWInput if gw_results_file(opts) else Input
    dictionary["MetaFile"] = GWMetaFile if gw_results_file(opts) else MetaFile
    dictionary["FinishingTouches"] = FinishingTouches
    return dictionary


def _logger_format():
    return '%(asctime)s %(name)s %(levelname)-8s: %(message)s'


def setup_logger():
    """Set up the logger output.
    """
    import tempfile

    if not os.path.isdir(".tmp/pesummary"):
        os.makedirs(".tmp/pesummary")
    dirpath = tempfile.mkdtemp(dir=".tmp/pesummary")
    level = 'INFO'
    if "-v" in sys.argv or "--verbose" in sys.argv:
        level = 'DEBUG'

    FORMAT = _logger_format()
    logger = logging.getLogger('PESummary')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('%s/pesummary.log' % (dirpath), mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel('INFO')
    formatter = logging.Formatter(FORMAT, datefmt='%Y-%m-%d  %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


def remove_tmp_directories():
    """Remove the temporary directories created by PESummary
    """
    import shutil
    from glob import glob
    import time

    directories = glob(".tmp/pesummary/*")

    for i in directories:
        if os.path.isdir(i):
            shutil.rmtree(i)
        elif os.path.isfile(i):
            os.remove(i)


def _add_existing_data(namespace):
    """Add existing data to namespace object
    """
    for num, i in enumerate(namespace.existing_labels):
        if hasattr(namespace, "labels") and i not in namespace.labels:
            namespace.labels.append(i)
        if hasattr(namespace, "samples") and i not in list(namespace.samples.keys()):
            namespace.samples[i] = namespace.existing_samples[i]
        if hasattr(namespace, "weights") and i not in list(namespace.weights.keys()):
            if namespace.existing_weights is None:
                namespace.weights[i] = None
            else:
                namespace.weights[i] = namespace.existing_weights[i]
        if hasattr(namespace, "injection_data"):
            if i not in list(namespace.injection_data.keys()):
                namespace.injection_data[i] = namespace.existing_injection_data[i]
        if hasattr(namespace, "file_versions"):
            if i not in list(namespace.file_versions.keys()):
                namespace.file_versions[i] = namespace.existing_file_version[i]
        if hasattr(namespace, "file_kwargs"):
            if i not in list(namespace.file_kwargs.keys()):
                namespace.file_kwargs[i] = namespace.existing_file_kwargs[i]
        if hasattr(namespace, "config"):
            if namespace.existing_config[num] not in namespace.config:
                namespace.config.append(namespace.existing_config[num])
        if hasattr(namespace, "priors"):
            if hasattr(namespace, "existing_priors"):
                for key, item in namespace.existing_priors.items():
                    if key in namespace.priors.keys():
                        for label in item.keys():
                            if label not in namespace.priors[key].keys():
                                namespace.priors[key][label] = item[label]
                    else:
                        namespace.priors.update({key: item})
        if hasattr(namespace, "approximant") and namespace.approximant is not None:
            if i not in list(namespace.approximant.keys()):
                if i in list(namespace.existing_approximant.keys()):
                    namespace.approximant[i] = namespace.existing_approximant[i]
        if hasattr(namespace, "psds") and namespace.psds is not None:
            if i not in list(namespace.psds.keys()):
                if i in list(namespace.existing_psd.keys()):
                    namespace.psds[i] = namespace.existing_psd[i]
                else:
                    namespace.psds[i] = {}
        if hasattr(namespace, "calibration") and namespace.calibration is not None:
            if i not in list(namespace.calibration.keys()):
                if i in list(namespace.existing_calibration.keys()):
                    namespace.calibration[i] = namespace.existing_calibration[i]
                else:
                    namespace.calibration[i] = {}
        if hasattr(namespace, "maxL_samples"):
            if i not in list(namespace.maxL_samples.keys()):
                namespace.maxL_samples[i] = {
                    key: val.maxL for key, val in namespace.samples[i].items()
                }
        if hasattr(namespace, "pepredicates_probs"):
            if i not in list(namespace.pepredicates_probs.keys()):
                from pesummary.gw.pepredicates import get_classifications

                namespace.pepredicates_probs[i] = get_classifications(
                    namespace.existing_samples[i]
                )
        if hasattr(namespace, "pastro_probs"):
            if i not in list(namespace.pastro_probs.keys()):
                from pesummary.gw.p_astro import get_probabilities

                em_bright = get_probabilities(namespace.existing_samples[i])
                namespace.pastro_probs[i] = {
                    "default": em_bright[0],
                    "population": em_bright[1]
                }
    if hasattr(namespace, "result_files"):
        number = len(namespace.labels)
        while len(namespace.result_files) < number:
            namespace.result_files.append(namespace.existing_metafile)
    parameters = [list(namespace.samples[i].keys()) for i in namespace.labels]
    namespace.same_parameters = list(
        set.intersection(*[set(l) for l in parameters])
    )
    namespace.same_samples = {
        param: {
            i: namespace.samples[i][param] for i in namespace.labels
        } for param in namespace.same_parameters
    }
    return namespace


def customwarn(message, category, filename, lineno, file=None, line=None):
    """
    """
    import sys
    import warnings

    sys.stdout.write(
        warnings.formatwarning("%s" % (message), category, filename, lineno)
    )


def determine_gps_time_and_window(maxL_samples, labels):
    """Determine the gps time and window to use in the spectrogram and
    omegascan plots
    """
    times = [
        maxL_samples[label]["geocent_time"] for label in labels
    ]
    gps_time = np.mean(times)
    time_range = np.max(times) - np.min(times)
    if time_range < 4.:
        window = 4.
    else:
        window = time_range * 1.5
    return gps_time, window


def number_of_columns_for_legend(labels):
    """Determine the number of columns to use in a legend

    Parameters
    ----------
    labels: list
        list of labels in the legend
    """
    max_length = np.max([len(i) for i in labels]) + 5.
    if max_length > 50.:
        return 1
    else:
        return int(50. / max_length)


class RedirectLogger(object):
    """Class to redirect the output from other codes to the `pesummary`
    logger

    Parameters
    ----------
    level: str, optional
        the level to display the messages
    """
    def __init__(self, code, level="Debug"):
        self.logger = logging.getLogger('PESummary')
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)
        self.code = code

    def isatty(self):
        pass

    def write(self, msg):
        """Write the message to stdout

        Parameters
        ----------
        msg: str
            the message you wish to be printed to stdout
        """
        if msg and not msg.isspace():
            self.logger.log(self.level, "[from %s] %s" % (self.code, msg))

    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redirector.__exit__(exc_type, exc_value, traceback)


def draw_conditioned_prior_samples(
    samples_dict, prior_samples_dict, conditioned, xlow, xhigh, N
):
    """Return a prior_dict that is conditioned on certain parameters

    Parameters
    ----------
    samples_dict: pesummary.utils.utils.SamplesDict
        SamplesDict containing the posterior samples
    prior_samples_dict: pesummary.utils.utils.SamplesDict
        SamplesDict containing the prior samples
    conditioned: list
        list of parameters that you wish to condition your prior on
    xlow: dict
        dictionary of lower bounds for each parameter
    xhigh: dict
        dictionary of upper bounds for each parameter
    N: int
        number of points to use within the grid
    """
    for param in conditioned:
        indices = _draw_conditioned_prior_samples(
            prior_samples_dict[param], samples_dict[param], xlow[param],
            xhigh[param], N
        )
        for key, val in prior_samples_dict.items():
            prior_samples_dict[key] = val[indices]

    return prior_samples_dict


def _draw_conditioned_prior_samples(
    prior_samples, posterior_samples, xlow, xhigh, xN=1000, N=1000
):
    """Return a list of indices for the conditioned prior via rejection
    sampling. The conditioned prior will then be `prior_samples[indicies]`.
    Code from Michael Puerrer.

    Parameters
    ----------
    prior_samples: np.ndarray
        array of prior samples that you wish to condition
    posterior_samples: np.ndarray
        array of posterior samples that you wish to condition on
    xlow: float
        lower bound for grid to be used
    xhigh: float
        upper bound for grid to be used
    xN: int, optional
        Number of points to use within the grid
    N: int, optional
        Number of samples to generate
    """
    from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde

    prior_KDE = Bounded_1d_kde(prior_samples)
    posterior_KDE = Bounded_1d_kde(posterior_samples)

    x = np.linspace(xlow, xhigh, xN)
    idx_nz = np.nonzero(posterior_KDE(x))
    pdf_ratio = prior_KDE(x)[idx_nz] / posterior_KDE(x)[idx_nz]
    M = 1.1 / min(pdf_ratio[np.where(pdf_ratio < 1)])

    indicies = []
    i = 0
    while i < N:
        x_i = np.random.choice(prior_samples)
        idx_i = np.argmin(np.abs(prior_samples - x_i))
        u = np.random.uniform()
        if u < posterior_KDE(x_i) / (M * prior_KDE(x_i)):
            indicies.append(idx_i)
            i += 1
    return indicies


def unzip(zip_file, outdir=".", overwrite=False):
    """Extract the data from a zipped file and save in outdir.

    Parameters
    ----------
    zip_file: str
        path to the file you wish to unzip
    outdir: str, optional
        path to the directory where you wish to save the unzipped file.
    overwrite: Bool, optional
        If True, overwrite a file that has the same name
    """
    import gzip
    import shutil
    from pathlib import Path

    f = Path(zip_file)
    file_name = f.stem
    out_file = os.path.join(outdir, file_name)
    if os.path.isfile(out_file) and not overwrite:
        raise FileExistsError(
            "The file '{}' already exists. Not overwriting".format(out_file)
        )
    with gzip.open(zip_file, 'rb') as input:
        with open(out_file, 'wb') as output:
            shutil.copyfileobj(input, output)
    return out_file


def iterator(
    iterable, desc="", tqdm=False, logger_output=True, total=None,
    logger_name="PESummary"
):
    """Return either a tqdm iterator, if tqdm installed, or a simple range

    Parameters
    ----------
    iterable: func
        iterable that you wish to iterate over
    desc: str, optional
        description for the tqdm bar
    tqdm: Bool, optional
        If True, a tqdm object is used. Otherwise simply returns the iterator.
    logger_output: Bool, optional
        If True, the tqdm progress bar interacts with logger
    total: float, optional
        total length of iterable
    logger_name: str, optional
        name of the logger you wish to use
    """
    if tqdm:
        try:
            FORMAT, DESC = None, None
            if logger_output:
                data = {"name": logger_name, "levelname": "INFO",
                        "message": desc, "asctime": "{asctime}"}
                FORMAT = _logger_format() % (data) + (
                    ' | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}'
                )
            elif len(desc):
                DESC = desc
            return _tqdm(
                iterable, total=total, desc=DESC, bar_format=FORMAT
            )
        except ImportError:
            return iterable
    else:
        return iterable


def _check_latex_install():
    from matplotlib import rcParams
    from distutils.spawn import find_executable

    if find_executable("latex"):
        rcParams["text.usetex"] = True
    else:
        rcParams["text.usetex"] = False


setup_logger()
logger = logging.getLogger('PESummary')
