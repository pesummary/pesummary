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
from pesummary.utils.decorators import docstring_subfunction
from pesummary.utils.array import Array
from pesummary.utils.dict import Dict
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary import conf
import importlib

latex_labels.update(GWlatex_labels)


class SamplesDict(Dict):
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
    key_data: dict
        dictionary containing the key data associated with each array
    number_of_samples: int
        Number of samples stored in the SamplesDict object
    latex_labels: dict
        Dictionary of latex labels for each parameter
    available_plots: list
        list of plots which the user may user to display the contained posterior
        samples

    Methods
    -------
    from_file:
        Initialize the SamplesDict class with the contents of a file
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
    generate_all_posterior_samples:
        Convert the posterior samples in the SamplesDict object according to
        a conversion function

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
        super(SamplesDict, self).__init__(
            *args, value_class=Array, make_dict_kwargs={"autoscale": autoscale},
            logger_warn=logger_warn, latex_labels=latex_labels
        )

    def __getitem__(self, key):
        """Return an object representing the specialization of SamplesDict
        by type arguments found in key.
        """
        if isinstance(key, slice):
            return SamplesDict(
                self.parameters,
                [i[key.start:key.stop:key.step] for i in self.samples]
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

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Initialize the SamplesDict class with the contents of a result file

        Parameters
        ----------
        filename: str
            path to the result file you wish to load.
        **kwargs: dict
            all kwargs are passed to the pesummary.io.read function
        """
        from pesummary.io import read

        return read(filename, **kwargs).samples_dict

    @property
    def key_data(self):
        return {param: value.key_data for param, value in self.items()}

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

    @property
    def plotting_map(self):
        existing = super(SamplesDict, self).plotting_map
        modified = existing.copy()
        modified.update(
            {
                "marginalized_posterior": self._marginalized_posterior,
                "skymap": self._skymap,
                "hist": self._marginalized_posterior,
                "corner": self._corner,
                "spin_disk": self._spin_disk,
                "2d_kde": self._2d_kde,
                "triangle": self._triangle,
                "reverse_triangle": self._reverse_triangle,
            }
        )
        return modified

    def to_pandas(self, **kwargs):
        """Convert a SamplesDict object to a pandas dataframe
        """
        from pandas import DataFrame

        return DataFrame(self, **kwargs)

    def to_structured_array(self, **kwargs):
        """Convert a SamplesDict object to a structured numpy array
        """
        return self.to_pandas(**kwargs).to_records(
            index=False, column_dtypes=np.float
        )

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
        samples = self.samples
        self.samples = np.delete(samples, ind, axis=0)
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

    def make_dictionary(self, discard_samples=None, autoscale=True):
        """Add the parameters and samples to the class
        """
        lengths = [len(i) for i in self.samples]
        if len(np.unique(lengths)) > 1 and autoscale:
            nsamples = np.min(lengths)
            getattr(logger, self.logger_warn)(
                "Unequal number of samples for each parameter. "
                "Restricting all posterior samples to have {} "
                "samples".format(nsamples)
            )
            self.samples = [
                dataset[:nsamples] for dataset in self.samples
            ]
        if "log_likelihood" in self.parameters:
            likelihoods = self.samples[self.parameters.index("log_likelihood")]
            likelihoods = likelihoods[discard_samples:]
        else:
            likelihoods = None
        if "log_prior" in self.parameters:
            priors = self.samples[self.parameters.index("log_prior")]
            priors = priors[discard_samples:]
        else:
            priors = None
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
                val[discard_samples:], likelihood=likelihoods, prior=priors,
                weights=weights
            )

    @docstring_subfunction([
        'pesummary.core.plots.plot._1d_histogram_plot',
        'pesummary.gw.plots.plot._1d_histogram_plot',
        'pesummary.gw.plots.plot._ligo_skymap_plot',
        'pesummary.gw.plots.publication.spin_distribution_plots',
        'pesummary.core.plots.plot._make_corner_plot',
        'pesummary.gw.plots.plot._make_corner_plot'
    ])
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
        return super(SamplesDict, self).plot(*args, type=type, **kwargs)

    def generate_all_posterior_samples(self, function=None, **kwargs):
        """Convert samples stored in the SamplesDict according to a conversion
        function

        Parameters
        ----------
        function: func, optional
            function to use when converting posterior samples. Must take a
            dictionary as input and return a dictionary of converted posterior
            samples. Default `pesummary.gw.file.conversions._Conversion
        **kwargs: dict, optional
            All additional kwargs passed to function
        """
        if function is None:
            from pesummary.gw.file.conversions import _Conversion
            function = _Conversion
        _samples = self.copy()
        _keys = list(_samples.keys())
        converted_samples = function(_samples, **kwargs)
        for key, item in converted_samples.items():
            if key not in _keys:
                self[key] = item
        return

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

    def _spin_disk(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.spin_distribution_plots`
        function
        """
        from pesummary.gw.plots.publication import spin_distribution_plots

        required = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        if not all(param in self.keys() for param in required):
            raise ValueError(
                "The spin disk plot requires samples for the following "
                "parameters: {}".format(", ".join(required))
            )
        samples = [self[param] for param in required]
        return spin_distribution_plots(required, samples, None, **kwargs)

    def _corner(self, module="core", parameters=None, **kwargs):
        """Wrapper for the `pesummary.core.plots.plot._make_corner_plot` or
        `pesummary.gw.plots.plot._make_corner_plot` function

        Parameters
        ----------
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict
            all additional kwargs are passed to the `_make_corner_plot`
            function
        """
        module = importlib.import_module(
            "pesummary.{}.plots.plot".format(module)
        )
        _parameters = None
        if parameters is not None:
            _parameters = [param for param in parameters if param in self.keys()]
            if not len(_parameters):
                raise ValueError(
                    "None of the chosen parameters are in the posterior "
                    "samples table. Please choose other parameters to plot"
                )
        return getattr(module, "_make_corner_plot")(
            self, self.latex_labels, corner_parameters=_parameters, **kwargs
        )[0]

    def _2d_kde(self, parameters, module="core", **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.twod_contour_plot` or
        `pesummary.core.plots.publication.twod_contour_plot` function

        Parameters
        ----------
        parameters: list
            list of length 2 giving the parameters you wish to plot
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict, optional
            all additional kwargs are passed to the `twod_contour_plot` function
        """
        _module = importlib.import_module(
            "pesummary.{}.plots.publication".format(module)
        )
        if module == "gw":
            return getattr(_module, "twod_contour_plots")(
                parameters, [[self[parameters[0]], self[parameters[1]]]],
                [None], {
                    parameters[0]: self.latex_labels[parameters[0]],
                    parameters[1]: self.latex_labels[parameters[1]]
                }, **kwargs
            )
        return getattr(_module, "twod_contour_plot")(
            self[parameters[0]], self[parameters[1]],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], **kwargs
        )

    def _triangle(self, parameters, **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.core.plots.publication import triangle_plot

        return triangle_plot(
            [self[parameters[0]]], [self[parameters[1]]],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], **kwargs
        )

    def _reverse_triangle(self, parameters, **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.reverse_triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.core.plots.publication import reverse_triangle_plot

        return reverse_triangle_plot(
            [self[parameters[0]]], [self[parameters[1]]],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], **kwargs
        )

    def classification(self, prior=None):
        """Return the classification probabilities

        Parameters
        ----------
        prior: str, optional
            prior you wish to use when generating the classification
            probabilities.
        """
        from pesummary.gw.pepredicates import get_classifications
        from pesummary.gw.p_astro import get_probabilities

        _prior = ["default", "population", None]
        if prior not in _prior:
            raise ValueError(
                "Unrecognised prior. Prior must be either: {}".format(
                    ", ".join(_prior)
                )
            )
        classifications = get_classifications(self)
        embright = get_probabilities(self)
        classifications["default"].update(embright[0])
        classifications["population"].update(embright[1])
        if prior is not None:
            return classifications[prior]
        return classifications

    def _waveform_args(self, ind=0, longAscNodes=0., eccentricity=0.):
        """Arguments to be passed to waveform generation

        Parameters
        ----------
        ind: int, optional
            index for the sample you wish to plot
        longAscNodes: float, optional
            longitude of ascending nodes, degenerate with the polarization
            angle. Default 0.
        eccentricity: float, optional
            eccentricity at reference frequency. Default 0.
        """
        from lal import MSUN_SI, PC_SI

        _samples = {key: value[ind] for key, value in self.items()}
        required = [
            "mass_1", "mass_2", "luminosity_distance", "iota"
        ]
        if not all(param in _samples.keys() for param in required):
            raise ValueError(
                "Unable to generate a waveform. Please add samples for "
                + ", ".join(required)
            )
        waveform_args = [
            _samples["mass_1"] * MSUN_SI, _samples["mass_2"] * MSUN_SI
        ]
        spins = [
            _samples[param] if param in _samples.keys() else 0. for param in
            ["spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z"]
        ]
        waveform_args += spins
        phase = _samples["phase"] if "phase" in _samples.keys() else 0.
        waveform_args += [
            _samples["luminosity_distance"] * PC_SI * 10**6,
            _samples["iota"], phase
        ]
        waveform_args += [longAscNodes, eccentricity, 0.]
        return waveform_args, _samples

    def _project_waveform(self, ifo, hp, hc, ra, dec, psi, time):
        """Project a waveform onto a given detector

        Parameters
        ----------
        ifo: str
            name of the detector you wish to project the waveform onto
        hp: np.ndarray
            plus gravitational wave polarization
        hc: np.ndarray
            cross gravitational wave polarization
        ra: float
            right ascension to be passed to antenna response function
        dec: float
            declination to be passed to antenna response function
        psi: float
            polarization to be passed to antenna response function
        time: float
            time to be passed to antenna response function
        """
        import importlib

        mod = importlib.import_module("pesummary.gw.plots.plot")
        func = getattr(mod, "__antenna_response")
        antenna = func(ifo, ra, dec, psi, time)
        ht = hp * antenna[0] + hc * antenna[1]
        return ht

    def fd_waveform(
        self, approximant, delta_f, f_low, f_high, f_ref=20., project=None,
        ind=0, longAscNodes=0., eccentricity=0., LAL_parameters=None
    ):
        """Generate a gravitational wave in the frequency domain

        Parameters
        ----------
        approximant: str
            name of the approximant to use when generating the waveform
        delta_f: float
            spacing between frequency samples
        f_low: float
            frequency to start evaluating the waveform
        f_high: float
            frequency to stop evaluating the waveform
        f_ref: float, optional
            reference frequency
        project: str, optional
            name of the detector to project the waveform onto. If None,
            the plus and cross polarizations are returned. Default None
        ind: int, optional
            index for the sample you wish to plot
        longAscNodes: float, optional
            longitude of ascending nodes, degenerate with the polarization
            angle. Default 0.
        eccentricity: float, optional
            eccentricity at reference frequency. Default 0.
        LAL_parameters: dict, optional
            LAL dictioanry containing accessory parameters. Default None
        """
        from gwpy.frequencyseries import FrequencySeries
        import lalsimulation as lalsim

        waveform_args, _samples = self._waveform_args(
            ind=ind, longAscNodes=longAscNodes, eccentricity=eccentricity
        )
        approx = lalsim.GetApproximantFromString(approximant)
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            *waveform_args, delta_f, f_low, f_high, f_ref, LAL_parameters, approx
        )
        hp = FrequencySeries(hp.data.data, df=hp.deltaF, f0=0.)
        hc = FrequencySeries(hc.data.data, df=hc.deltaF, f0=0.)
        if project is None:
            return {"h_plus": hp, "h_cross": hc}
        ht = self._project_waveform(
            project, hp, hc, _samples["ra"], _samples["dec"], _samples["psi"],
            _samples["geocent_time"]
        )
        return ht

    def td_waveform(
        self, approximant, delta_t, f_low, f_ref=20., project=None, ind=0,
        longAscNodes=0., eccentricity=0., LAL_parameters=None
    ):
        """Generate a gravitational wave in the time domain

        Parameters
        ----------
        approximant: str
            name of the approximant to use when generating the waveform
        delta_t: float
            spacing between frequency samples
        f_low: float
            frequency to start evaluating the waveform
        f_ref: float, optional
            reference frequency
        project: str, optional
            name of the detector to project the waveform onto. If None,
            the plus and cross polarizations are returned. Default None
        ind: int, optional
            index for the sample you wish to plot
        longAscNodes: float, optional
            longitude of ascending nodes, degenerate with the polarization
            angle. Default 0.
        eccentricity: float, optional
            eccentricity at reference frequency. Default 0.
        LAL_parameters: dict, optional
            LAL dictioanry containing accessory parameters. Default None
        """
        from gwpy.timeseries import TimeSeries
        import lalsimulation as lalsim
        from astropy.units import Quantity

        waveform_args, _samples = self._waveform_args(
            ind=ind, longAscNodes=longAscNodes, eccentricity=eccentricity
        )
        approx = lalsim.GetApproximantFromString(approximant)
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            *waveform_args, delta_t, f_low, f_ref, LAL_parameters, approx
        )
        hp = TimeSeries(hp.data.data, dt=hp.deltaT, t0=hp.epoch)
        hc = TimeSeries(hc.data.data, dt=hc.deltaT, t0=hc.epoch)
        if project is None:
            return {"h_plus": hp, "h_cross": hc}
        ht = self._project_waveform(
            project, hp, hc, _samples["ra"], _samples["dec"], _samples["psi"],
            _samples["geocent_time"]
        )
        ht.times = (
            Quantity(ht.times, unit="s")
            + Quantity(_samples["{}_time".format(project)], unit="s")
        )
        return ht

    def _maxL_waveform(self, func, *args, **kwargs):
        """Return the maximum likelihood waveform in a given domain

        Parameters
        ----------
        func: function
            function you wish to use when generating the maximum likelihood
            waveform
        *args: tuple
            all args passed to func
        **kwargs: dict
            all kwargs passed to func
        """
        ind = np.argmax(self["log_likelihood"])
        kwargs["ind"] = ind
        return func(*args, **kwargs)

    def maxL_td_waveform(self, *args, **kwargs):
        """Generate the maximum likelihood gravitational wave in the time domain

        Parameters
        ----------
        approximant: str
            name of the approximant to use when generating the waveform
        delta_t: float
            spacing between frequency samples
        f_low: float
            frequency to start evaluating the waveform
        f_ref: float, optional
            reference frequency
        project: str, optional
            name of the detector to project the waveform onto. If None,
            the plus and cross polarizations are returned. Default None
        longAscNodes: float, optional
            longitude of ascending nodes, degenerate with the polarization
            angle. Default 0.
        eccentricity: float, optional
            eccentricity at reference frequency. Default 0.
        LAL_parameters: dict, optional
            LAL dictioanry containing accessory parameters. Default None
        """
        return self._maxL_waveform(self.td_waveform, *args, **kwargs)

    def maxL_fd_waveform(self, *args, **kwargs):
        """Generate the maximum likelihood gravitational wave in the frequency
        domain

        Parameters
        ----------
        approximant: str
            name of the approximant to use when generating the waveform
        delta_f: float
            spacing between frequency samples
        f_low: float
            frequency to start evaluating the waveform
        f_high: float
            frequency to stop evaluating the waveform
        f_ref: float, optional
            reference frequency
        project: str, optional
            name of the detector to project the waveform onto. If None,
            the plus and cross polarizations are returned. Default None
        longAscNodes: float, optional
            longitude of ascending nodes, degenerate with the polarization
            angle. Default 0.
        eccentricity: float, optional
            eccentricity at reference frequency. Default 0.
        LAL_parameters: dict, optional
            LAL dictioanry containing accessory parameters. Default None
        """
        return self._maxL_waveform(self.fd_waveform, *args, **kwargs)


class _MultiDimensionalSamplesDict(Dict):
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
        self.latex_labels = self._latex_labels()

    def _latex_labels(self):
        """
        """
        return {
            param: latex_labels[param] if param in latex_labels.keys() else
            param for param in self.total_list_of_parameters
        }

    def __setitem__(self, key, value):
        _value = value
        if not isinstance(value, SamplesDict):
            _value = SamplesDict(value)
        super(_MultiDimensionalSamplesDict, self).__setitem__(key, _value)
        try:
            if key not in self.labels:
                parameters = list(value.keys())
                samples = np.array([value[param] for param in parameters])
                self.parameters[key] = parameters
                self.labels.append(key)
                self.latex_labels = self._latex_labels()
        except (AttributeError, TypeError):
            pass

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

    @property
    def total_list_of_parameters(self):
        if isinstance(self.parameters, dict):
            _parameters = [item for item in self.parameters.values()]
            _flat_parameters = [
                item for sublist in _parameters for item in sublist
            ]
        elif isinstance(self.parameters, list):
            if np.array(self.parameters).ndim > 1:
                _flat_parameters = [
                    item for sublist in self.parameters for item in sublist
                ]
            else:
                _flat_parameters = self.parameters
        return list(set(_flat_parameters))

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

    @property
    def key_data(self):
        data = {}
        for param, value in self.combine.items():
            data[param] = value.key_data
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
    available_plots: list
        list of plots which the user may user to display the contained posterior
        samples

    Methods
    -------
    from_files:
        Initialize the MultiAnalysisSamplesDict class with the contents of
        multiple files
    js_divergence: float
        Return the JS divergence between two posterior distributions for a
        given parameter. See pesummary.utils.utils.jensen_shannon_divergence
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

    @classmethod
    def from_files(cls, filenames, **kwargs):
        """Initialize the MultiAnalysisSamplesDict class with the contents of
        multiple result files

        Parameters
        ----------
        filenames: dict
            dictionary containing the path to the result file you wish to load
            as the item and a label associated with each result file as the key
        **kwargs: dict
            all kwargs are passed to the pesummary.io.read function
        """
        from pesummary.io import read
        from pesummary.core.inputs import _Input

        samples = {}
        for label, filename in filenames.items():
            _samples = read(filename, **kwargs).samples_dict
            if _Input.is_pesummary_metafile(filename):
                _stored_labels = _samples.keys()
                cond1 = any(
                    _label in filenames.keys() for _label in _stored_labels
                )
                cond2 = any(
                    _label in samples.keys() for _label in _stored_labels
                )
                if cond1 or cond2:
                    raise ValueError(
                        "One or more of the labels stored in the PESummary "
                        "meta file matches another label. Please provide unique "
                        "labels for each dataset"
                    )
                samples.update(_samples)
            else:
                samples[label] = _samples
        return cls(samples)

    @property
    def plotting_map(self):
        return {
            "hist": self._marginalized_posterior,
            "corner": self._corner,
            "triangle": self._triangle,
            "reverse_triangle": self._reverse_triangle,
            "violin": self._violin,
            "2d_kde": self._2d_kde
        }

    @property
    def available_plots(self):
        return list(self.plotting_map.keys())

    @docstring_subfunction([
        'pesummary.core.plots.plot._1d_comparison_histogram_plot',
        'pesummary.gw.plots.plot._1d_comparison_histogram_plot',
        'pesummary.core.plots.publication.triangle_plot',
        'pesummary.core.plots.publication.reverse_triangle_plot'
    ])
    def plot(
        self, *args, type="hist", labels="all", colors=None, latex_friendly=True,
        **kwargs
    ):
        """Generate a plot for the posterior samples stored in
        MultiDimensionalSamplesDict

        Parameters
        ----------
        *args: tuple
            all arguments are passed to the plotting function
        type: str
            name of the plot you wish to make
        labels: list
            list of analyses that you wish to include in the plot
        colors: list
            list of colors to use for each analysis
        latex_friendly: Bool, optional
            if True, make the labels latex friendly. Default True
        **kwargs: dict
            all additional kwargs are passed to the plotting function
        """
        if type not in self.plotting_map.keys():
            raise NotImplementedError(
                "The {} method is not currently implemented. The allowed "
                "plotting methods are {}".format(
                    type, ", ".join(self.available_plots)
                )
            )

        if labels == "all":
            labels = self.labels
        elif isinstance(labels, list):
            for label in labels:
                if label not in self.labels:
                    raise ValueError(
                        "'{}' is not a stored analysis. The available analyses "
                        "are: '{}'".format(label, ", ".join(self.labels))
                    )
        else:
            raise ValueError(
                "Please provide a list of analyses that you wish to plot"
            )
        if colors is None:
            colors = list(conf.colorcycle)
            while len(colors) < len(labels):
                colors += colors

        kwargs["labels"] = labels
        kwargs["colors"] = colors
        kwargs["latex_friendly"] = latex_friendly
        return self.plotting_map[type](*args, **kwargs)

    def _marginalized_posterior(
        self, parameter, module="core", labels="all", colors=None, **kwargs
    ):
        """Wrapper for the
        `pesummary.core.plots.plot._1d_comparison_histogram_plot` or
        `pesummary.gw.plots.plot._comparison_1d_histogram_plot`

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to plot
        module: str, optional
            module you wish to use for the plotting
        labels: list
            list of analyses that you wish to include in the plot
        colors: list
            list of colors to use for each analysis
        **kwargs: dict
            all additional kwargs are passed to the
            `_1d_comparison_histogram_plot` function
        """
        module = importlib.import_module(
            "pesummary.{}.plots.plot".format(module)
        )
        return getattr(module, "_1d_comparison_histogram_plot")(
            parameter, [self[label][parameter] for label in labels],
            colors, self.latex_labels[parameter], labels, **kwargs
        )

    def _base_triangle(self, parameters, labels="all"):
        """Check that the parameters are valid for the different triangle
        plots available

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        labels: list
            list of analyses that you wish to include in the plot
        """
        samples = [self[label] for label in labels]
        if len(parameters) > 2:
            raise ValueError("Function is only 2d")
        condition = set(
            label for num, label in enumerate(labels) for param in parameters if
            param not in samples[num].keys()
        )
        if len(condition):
            raise ValueError(
                "{} and {} are not available for the following "
                " analyses: {}".format(
                    parameters[0], parameters[1], ", ".join(condition)
                )
            )
        return samples

    def _triangle(self, parameters, labels="all", **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        labels: list
            list of analyses that you wish to include in the plot
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.core.plots.publication import triangle_plot

        samples = self._base_triangle(parameters, labels=labels)
        return triangle_plot(
            [_samples[parameters[0]] for _samples in samples],
            [_samples[parameters[1]] for _samples in samples],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], labels=labels, **kwargs
        )

    def _reverse_triangle(self, parameters, labels="all", **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.reverse_triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        labels: list
            list of analyses that you wish to include in the plot
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.core.plots.publication import reverse_triangle_plot

        samples = self._base_triangle(parameters, labels=labels)
        return reverse_triangle_plot(
            [_samples[parameters[0]] for _samples in samples],
            [_samples[parameters[1]] for _samples in samples],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], labels=labels, **kwargs
        )

    def _violin(
        self, parameter, labels="all", priors=None, latex_labels=GWlatex_labels,
        **kwargs
    ):
        """Wrapper for the `pesummary.gw.plots.publication.violin_plots`
        function

        Parameters
        ----------
        parameter: str, optional
            name of the parameter you wish to generate a violin plot for
        labels: list
            list of analyses that you wish to include in the plot
        priors: MultiAnalysisSamplesDict, optional
            prior samples for each analysis. If provided, the right hand side
            of each violin will show the prior
        latex_labels: dict, optional
            dictionary containing the latex label associated with parameter
        **kwargs: dict
            all additional kwargs are passed to the `violin_plots` function
        """
        from pesummary.gw.plots.publication import violin_plots

        _labels = [label for label in labels if parameter in self[label].keys()]
        if not len(_labels):
            raise ValueError(
                "{} is not in any of the posterior samples tables. Please "
                "choose another parameter to plot".format(parameter)
            )
        elif len(_labels) != len(labels):
            no = list(set(labels) - set(_labels))
            logger.warn(
                "Unable to generate a violin plot for {} because {} is not "
                "in their posterior samples table".format(
                    " or ".join(no), parameter
                )
            )
        samples = [self[label][parameter] for label in _labels]
        if priors is not None and not all(
                label in priors.keys() for label in _labels
        ):
            raise ValueError("Please provide prior samples for all labels")
        elif priors is not None and not all(
                parameter in priors[label].keys() for label in _labels
        ):
            raise ValueError(
                "Please provide prior samples for {} for all labels".format(
                    parameter
                )
            )
        elif priors is not None:
            from pesummary.core.plots.violin import split_dataframe

            priors = [priors[label][parameter] for label in _labels]
            samples = split_dataframe(samples, priors, _labels)
            palette = kwargs.get("palette", None)
            left, right = "color: white", "pastel"
            if palette is not None and not isinstance(palette, dict):
                right = palette
            elif palette is not None and all(
                    side in palette.keys() for side in ["left", "right"]
            ):
                left, right = palette["left"], palette["right"]
            kwargs.update(
                {
                    "split": True, "x": "label", "y": "data", "hue": "side",
                    "palette": {"right": right, "left": left}
                }
            )
        return violin_plots(
            parameter, samples, _labels, latex_labels, **kwargs
        )

    def _corner(self, module="core", labels="all", parameters=None, **kwargs):
        """Wrapper for the `pesummary.core.plots.plot._make_comparison_corner_plot`
        or `pesummary.gw.plots.plot._make_comparison_corner_plot` function

        Parameters
        ----------
        module: str, optional
            module you wish to use for the plotting
        labels: list
            list of analyses that you wish to include in the plot
        **kwargs: dict
            all additional kwargs are passed to the `_make_comparison_corner_plot`
            function
        """
        module = importlib.import_module(
            "pesummary.{}.plots.plot".format(module)
        )
        _samples = {label: self[label] for label in labels}
        _parameters = None
        if parameters is not None:
            _parameters = [
                param for param in parameters if all(
                    param in posterior for posterior in _samples.values()
                )
            ]
            if not len(_parameters):
                raise ValueError(
                    "None of the chosen parameters are in all of the posterior "
                    "samples tables. Please choose other parameters to plot"
                )
        return getattr(module, "_make_comparison_corner_plot")(
            _samples, self.latex_labels, corner_parameters=_parameters, **kwargs
        )

    def _2d_kde(
        self, parameters, module="core", labels="all", plot_density=None,
        **kwargs
    ):
        """Wrapper for the
        `pesummary.gw.plots.publication.comparison_twod_contour_plot` or
        `pesummary.core.plots.publication.comparison_twod_contour_plot` function

        Parameters
        ----------
        parameters: list
            list of length 2 giving the parameters you wish to plot
        module: str, optional
            module you wish to use for the plotting
        labels: list
            list of analyses that you wish to include in the plot
        **kwargs: dict, optional
            all additional kwargs are passed to the
            `comparison_twod_contour_plot` function
        """
        _module = importlib.import_module(
            "pesummary.{}.plots.publication".format(module)
        )
        samples = self._base_triangle(parameters, labels=labels)
        if plot_density is not None and plot_density not in labels:
            raise ValueError(
                "Unable to plot the density for '{}'. Please choose from: "
                "{}".format(plot_density, ", ".join(labels))
            )
        if module == "gw":
            return getattr(_module, "twod_contour_plots")(
                parameters, [
                    [self[label][param] for param in parameters] for label in
                    labels
                ], labels, {
                    parameters[0]: self.latex_labels[parameters[0]],
                    parameters[1]: self.latex_labels[parameters[1]]
                }, **kwargs
            )
        return getattr(_module, "comparison_twod_contour_plot")(
            [_samples[parameters[0]] for _samples in samples],
            [_samples[parameters[1]] for _samples in samples],
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], labels=labels,
            plot_density=plot_density, **kwargs
        )

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
        from pesummary.utils.utils import jensen_shannon_divergence

        return jensen_shannon_divergence(
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
