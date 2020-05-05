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

import copy
import numpy as np
from pesummary.utils.utils import resample_posterior_distribution, logger
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.latex_labels import GWlatex_labels
import importlib

latex_labels.update(GWlatex_labels)


class SamplesDict(dict):
    """Class to store the samples from a single run

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter
    autoscale: Bool, optional
        If True, the posterior samples for each parameter are scaled to the
        same length

    Attributes
    ----------
    maxL: pesummary.utils.samples_dict.SamplesDict
        SamplesDict object containing the maximum likelihood sample keyed by
        the parameter
    minimum: pesummary.utils.samples_dict.SamplesDict
        SamplesDict object containing the minimum sample for each parameter
    maximum: pesummary.utils.samples_dict.SamplesDict
        SamplesDict object containing the maximum sample for each parameter
    median: pesummary.utils.samples_dict.SamplesDict
        SamplesDict object containining the median of each marginalized
        posterior distribution
    mean: pesummary.utils.samples_dict.SamplesDict
        SamplesDict object containing the mean of each marginalized posterior
        distribution
    number_of_samples: int
        Number of samples stored in the SamplesDict object
    latex_labels: dict
        Dictionary of latex labels for each parameter

    Methods
    -------
    to_pandas:
        Convert the SamplesDict object to a pandas DataFrame
    to_structured_array:
        Convert the SamplesDict object to a numpy structured array
    pop:
        Remove an entry from the SamplesDict object
    downsample:
        Downsample the samples stored in the SamplesDict object. See the
        pesummary.utils.utils.resample_posterior_distribution method
    discard_samples:
        Remove the first N samples from each distribution
    plot:
        Generate a plot based on the posterior samples stored

    Examples
    --------
    How the initialize the SamplesDict class

    >>> from pesummary.utils.samples_dict import SamplesDict
    >>> data = {
    ...     "a": [1, 1.2, 1.7, 1.1, 1.4, 0.8, 1.6],
    ...     "b": [10.2, 11.3, 11.6, 9.5, 8.6, 10.8, 10.9]
    ... }
    >>> dataset = SamplesDict(data)
    >>> parameters = ["a", "b"]
    >>> samples = [
    ...     [1, 1.2, 1.7, 1.1, 1.4, 0.8, 1.6],
    ...     [10.2, 11.3, 11.6, 9.5, 8.6, 10.8, 10.9]
    ... }
    >>> dataset = SamplesDict(parameters, samples)
    >>> fig = dataset.plot("a", type="hist", bins=30)
    >>> fig.show()
    """
    def __init__(self, *args, logger_warn="warn", autoscale=True):
        super(SamplesDict, self).__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            self.parameters = list(args[0].keys())
            self.samples = np.array(
                [args[0][param] for param in self.parameters]
            )
            for key, item in args[0].items():
                self[key] = Array(item)
        else:
            self.parameters, self.samples = args
            lengths = [len(i) for i in self.samples]
            if len(np.unique(lengths)) > 1 and autoscale:
                nsamples = np.min(lengths)
                getattr(logger, logger_warn)(
                    "Unequal number of samples for each parameter. "
                    "Restricting all posterior samples to have {} "
                    "samples".format(nsamples)
                )
                self.samples = [
                    dataset[:nsamples] for dataset in self.samples
                ]
            self.make_dictionary()
        self.latex_labels = {
            param: latex_labels[param] if param in latex_labels.keys() else
            param for param in self.parameters
        }

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
        samples = self.samples
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

    def plot(self, *args, type="marginalized_posterior", **kwargs):
        """Generate a plot for the posterior samples stored in SamplesDict

        Parameters
        ----------
        *args: tuple
            all arguments are passed to the plotting function
        type: str
            name of the plot you wish to make
        **kwargs: dict
            all additional kwargs are passed to the plotting function
        """
        plotting_map = {
            "marginalized_posterior": self._marginalized_posterior,
            "skymap": self._skymap,
            "hist": self._marginalized_posterior
        }
        if type not in plotting_map.keys():
            raise NotImplementedError(
                "The {} method is not currently implemented. The allowed "
                "plotting methods are {}".format(
                    type, ", ".join(plotting_map.keys())
                )
            )
        return plotting_map[type](*args, **kwargs)

    def _marginalized_posterior(self, parameter, module="core", **kwargs):
        """Wrapper for the `pesummary.core.plots.plot._1d_histogram_plot` or
        `pesummary.gw.plots.plot._1d_histogram_plot`

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to plot
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict
            all additional kwargs are passed to the `_1d_histogram_plot`
            function
        """
        module = importlib.import_module(
            "pesummary.{}.plots.plot".format(module)
        )
        return getattr(module, "_1d_histogram_plot")(
            parameter, self[parameter], self.latex_labels[parameter], **kwargs
        )

    def _skymap(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.plot._ligo_skymap_plot`
        function

        Parameters
        ----------
        **kwargs: dict
            All kwargs are passed to the `_ligo_skymap_plot` function
        """
        from pesummary.gw.plots.plot import _ligo_skymap_plot

        if "luminosity_distance" in self.keys():
            dist = self["luminosity_distance"]
        else:
            dist = None

        return _ligo_skymap_plot(self["ra"], self["dec"], dist=dist, **kwargs)


class _MultiDimensionalSamplesDict(dict):
    """Class to store multiple SamplesDict objects

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter for each chain
    label_prefix: str, optional
        prefix to use when distinguishing different analyses. The label is then
        '{label_prefix}_{num}' where num is the result file index. Default
        is 'dataset'
    transpose: Bool, optional
        True if the input is a transposed dictionary
    labels: list, optional
        the labels to use to distinguish different analyses. If provided
        label_prefix is ignored

    Attributes
    ----------
    T: pesummary.utils.samples_dict._MultiDimensionalSamplesDict
        Transposed _MultiDimensionalSamplesDict object keyed by parameters
        rather than label
    combine: pesummary.utils.samples_dict.SamplesDict
        Combine all samples from all analyses into a single SamplesDict object
    nsamples: int
        Total number of analyses stored in the _MultiDimensionalSamplesDict
        object
    number_of_samples: dict
        Number of samples stored in the _MultiDimensionalSamplesDict for each
        analysis
    total_number_of_samples: int
        Total number of samples stored across the multiple analyses
    minimum_number_of_samples: int
        The number of samples in the smallest analysis

    Methods
    -------
    samples:
        Return a list of samples stored in the _MultiDimensionalSamplesDict
        object for a given parameter
    """
    def __init__(
        self, *args, label_prefix="dataset", transpose=False, labels=None
    ):
        if labels is not None and len(np.unique(labels)) != len(labels):
            raise ValueError(
                "Please provide a unique set of labels for each analysis"
            )
        invalid_label_number_error = "Please provide a label for each analysis"
        self.labels = labels
        self.name = _MultiDimensionalSamplesDict
        self.transpose = transpose
        if len(args) == 1 and isinstance(args[0], dict):
            if transpose:
                parameters = list(args[0].keys())
                _labels = list(args[0][parameters[0]].keys())
                outer_iterator, inner_iterator = parameters, _labels
            else:
                _labels = list(args[0].keys())
                parameters = {
                    label: list(args[0][label].keys()) for label in _labels
                }
                outer_iterator, inner_iterator = _labels, parameters
            if labels is None:
                self.labels = _labels
            for num, dataset in enumerate(outer_iterator):
                if isinstance(inner_iterator, dict):
                    samples = np.array(
                        [args[0][dataset][param] for param in inner_iterator[dataset]]
                    )
                else:
                    samples = np.array(
                        [args[0][dataset][param] for param in inner_iterator]
                    )
                if transpose:
                    desc = parameters[num]
                    self[desc] = SamplesDict(
                        self.labels, samples, logger_warn="debug",
                        autoscale=False
                    )
                else:
                    if self.labels is not None:
                        desc = self.labels[num]
                    else:
                        desc = "{}_{}".format(label_prefix, num)
                    self[desc] = SamplesDict(parameters[self.labels[num]], samples)
        else:
            parameters, samples = args
            if labels is not None and len(labels) != len(samples):
                raise ValueError(invalid_label_number_error)
            for num, dataset in enumerate(samples):
                if labels is not None:
                    desc = labels[num]
                else:
                    desc = "{}_{}".format(label_prefix, num)
                self[desc] = SamplesDict(parameters, dataset)
        if self.labels is None:
            self.labels = [
                "{}_{}".format(label_prefix, num) for num, _ in
                enumerate(samples)
            ]
        self.parameters = parameters

    @property
    def T(self):
        _params = sorted([param for param in self[self.labels[0]].keys()])
        if not all(sorted(self[label].keys()) == _params for label in self.labels):
            raise ValueError(
                "Unable to transpose as not all samples have the same parameters"
            )
        return self.name({
            param: {
                label: dataset[param] for label, dataset in self.items()
            } for param in self[self.labels[0]].keys()
        }, transpose=True)

    @property
    def combine(self):
        if self.transpose:
            data = SamplesDict({
                param: np.concatenate(
                    [self[param][key] for key in self[param].keys()]
                ) for param in self.parameters
            }, logger_warn="debug")
        else:
            data = SamplesDict({
                param: np.concatenate(
                    [self[key][param] for key in self.keys()]
                ) for param in self.parameters
            }, logger_warn="debug")
        return data

    @property
    def nsamples(self):
        if self.transpose:
            parameters = list(self.keys())
            return len(self[parameters[0]])
        return len(self)

    @property
    def number_of_samples(self):
        if self.transpose:
            return {
                label: len(self[iterator][label]) for iterator, label in zip(
                    self.keys(), self.labels
                )
            }
        return {
            label: self[iterator].number_of_samples for iterator, label in zip(
                self.keys(), self.labels
            )
        }

    @property
    def total_number_of_samples(self):
        return np.sum([length for length in self.number_of_samples.values()])

    @property
    def minimum_number_of_samples(self):
        return np.min([length for length in self.number_of_samples.values()])

    def samples(self, parameter):
        if self.transpose:
            samples = [self[parameter][label] for label in self.labels]
        else:
            samples = [self[label][parameter] for label in self.labels]
        return samples


class MCMCSamplesDict(_MultiDimensionalSamplesDict):
    """Class to store the mcmc chains from a single run

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter for each chain
    transpose: Bool, optional
        True if the input is a transposed dictionary

    Attributes
    ----------
    T: pesummary.utils.samples_dict.MCMCSamplesDict
        Transposed MCMCSamplesDict object keyed by parameters rather than
        chain
    average: pesummary.utils.samples_dict.SamplesDict
        The mean of each sample across multiple chains. If the chains are of
        different lengths, all chains are resized to the minimum number of
        samples
    combine: pesummary.utils.samples_dict.SamplesDict
        Combine all samples from all chains into a single SamplesDict object
    nchains: int
        Total number of chains stored in the MCMCSamplesDict object
    number_of_samples: dict
        Number of samples stored in the MCMCSamplesDict for each chain
    total_number_of_samples: int
        Total number of samples stored across the multiple chains
    minimum_number_of_samples: int
        The number of samples in the smallest chain

    Methods
    -------
    discard_samples:
        Discard the first N samples for each chain
    burnin:
        Remove the first N samples as burnin. For different algorithms
        see pesummary.core.file.mcmc.algorithms
    gelman_rubin: float
        Return the Gelman-Rubin statistic between the chains for a given
        parameter. See pesummary.utils.utils.gelman_rubin
    samples:
        Return a list of samples stored in the MCMCSamplesDict object for a
        given parameter

    Examples
    --------
    Initializing the MCMCSamplesDict class

    >>> from pesummary.utils.samplesdict import MCMCSamplesDict
    >>> data = {
    ...     "chain_0": {
    ...         "a": [1, 1.2, 1.7, 1.1, 1.4, 0.8, 1.6],
    ...         "b": [10.2, 11.3, 11.6, 9.5, 8.6, 10.8, 10.9]
    ...     },
    ...     "chain_1": {
    ...         "a": [0.8, 0.5, 1.7, 1.4, 1.2, 1.7, 0.9],
    ...         "b": [10, 10.5, 10.4, 9.6, 8.6, 11.6, 16.2]
    ...     }
    ... }
    >>> dataset = MCMCSamplesDict(data)
    >>> parameters = ["a", "b"]
    >>> samples = [
    ...     [
    ...         [1, 1.2, 1.7, 1.1, 1.4, 0.8, 1.6],
    ...         [10.2, 11.3, 11.6, 9.5, 8.6, 10.8, 10.9]
    ...     ], [
    ...         [0.8, 0.5, 1.7, 1.4, 1.2, 1.7, 0.9],
    ...         [10, 10.5, 10.4, 9.6, 8.6, 11.6, 16.2]
    ...     ]
    ... ]
    >>> dataset = MCMCSamplesDict(parameter, samples)
    """
    def __init__(self, *args, transpose=False):
        single_chain_error = (
            "This class requires more than one mcmc chain to be passed. "
            "As only one dataset is available, please use the SamplesDict "
            "class."
        )
        super(MCMCSamplesDict, self).__init__(
            *args, transpose=transpose, label_prefix="chain"
        )
        self.name = MCMCSamplesDict
        if len(self.labels) == 1:
            raise ValueError(single_chain_error)
        self.chains = self.labels
        self.nchains = self.nsamples

    @property
    def average(self):
        if self.transpose:
            data = SamplesDict({
                param: np.mean(
                    [
                        self[param][key][:self.minimum_number_of_samples] for
                        key in self[param].keys()
                    ], axis=0
                ) for param in self.parameters
            }, logger_warn="debug")
        else:
            data = SamplesDict({
                param: np.mean(
                    [
                        self[key][param][:self.minimum_number_of_samples] for
                        key in self.keys()
                    ], axis=0
                ) for param in self.parameters
            }, logger_warn="debug")
        return data

    def discard_samples(self, number):
        """Remove the first n samples

        Parameters
        ----------
        number: int/dict
            Number of samples that you wish to remove across all chains or a
            dictionary containing the number of samples to remove per chain
        """
        if isinstance(number, int):
            number = {chain: number for chain in self.keys()}
        for chain in self.keys():
            self[chain].discard_samples(number[chain])
        return self

    def burnin(self, *args, algorithm="burnin_by_step_number", **kwargs):
        """Remove the first N samples as burnin

        Parameters
        ----------
        algorithm: str, optional
            The algorithm you wish to use to remove samples as burnin. Default
            is 'burnin_by_step_number'. See
            `pesummary.core.file.mcmc.algorithms` for list of available
            algorithms
        """
        from pesummary.core.file import mcmc

        if algorithm not in mcmc.algorithms:
            raise ValueError(
                "{} is not a valid algorithm for removing samples as "
                "burnin".format(algorithm)
            )
        arguments = [self] + [i for i in args]
        return getattr(mcmc, algorithm)(*arguments, **kwargs)

    def gelman_rubin(self, parameter, decimal=5):
        """Return the gelman rubin statistic between chains for a given
        parameter

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to return the gelman rubin statistic
            for
        decimal: int
            number of decimal places to keep when rounding
        """
        from pesummary.utils.utils import gelman_rubin as _gelman_rubin

        return _gelman_rubin(self.samples(parameter), decimal=decimal)


class MultiAnalysisSamplesDict(_MultiDimensionalSamplesDict):
    """Class to samples from multiple analyses

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter for each chain
    labels: list, optional
        the labels to use to distinguish different analyses.
    transpose: Bool, optional
        True if the input is a transposed dictionary

    Attributes
    ----------
    T: pesummary.utils.samples_dict.MultiAnalysisSamplesDict
        Transposed MultiAnalysisSamplesDict object keyed by parameters
        rather than label
    combine: pesummary.utils.samples_dict.SamplesDict
        Combine all samples from all analyses into a single SamplesDict object
    nsamples: int
        Total number of analyses stored in the MultiAnalysisSamplesDict
        object
    number_of_samples: dict
        Number of samples stored in the MultiAnalysisSamplesDict for each
        analysis
    total_number_of_samples: int
        Total number of samples stored across the multiple analyses
    minimum_number_of_samples: int
        The number of samples in the smallest analysis

    Methods
    -------
    js_divergence: float
        Return the JS divergence between two posterior distributions for a
        given parameter. See pesummary.utils.utils.jension_shannon_divergence
    ks_statistic: float
        Return the KS statistic between two posterior distributions for a
        given parameter. See pesummary.utils.utils.kolmogorov_smirnov_test
    samples:
        Return a list of samples stored in the MCMCSamplesDict object for a
        given parameter
    """
    def __init__(self, *args, labels=None, transpose=False):
        if labels is None and not isinstance(args[0], dict):
            raise ValueError(
                "Please provide a unique label for each analysis"
            )
        super(MultiAnalysisSamplesDict, self).__init__(
            *args, labels=labels, transpose=transpose
        )
        self.name = MultiAnalysisSamplesDict

    def js_divergence(self, parameter, decimal=5):
        """Return the JS divergence between the posterior samples for
        a given parameter

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to return the gelman rubin statistic
            for
        decimal: int
            number of decimal places to keep when rounding
        """
        from pesummary.utils.utils import jension_shannon_divergence

        return jension_shannon_divergence(
            self.samples(parameter), decimal=decimal
        )

    def ks_statistic(self, parameter, decimal=5):
        """Return the KS statistic between the posterior samples for
        a given parameter

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to return the gelman rubin statistic
            for
        decimal: int
            number of decimal places to keep when rounding
        """
        from pesummary.utils.utils import kolmogorov_smirnov_test

        return kolmogorov_smirnov_test(
            self.samples(parameter), decimal=decimal
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
