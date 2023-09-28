# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import importlib

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _Base(object):
    """Base class for generating classification probabilities

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities

    Attributes
    ----------
    available_plots: list
        list of available plotting types

    Methods
    -------
    classification:
        return a dictionary containing the classification probabilities. These
        probabilities can either be generated from the raw samples or samples
        reweighted to a population inferred prior
    dual_classification:
        return a dictionary containing the classification probabilities
        generated from the raw samples ('default') and samples reweighted to
        a population inferred prior ('population')
    plot:
        generate a plot showing the classification probabilities
    """
    def __init__(self, samples):
        self.module = self.check_for_install()
        if not isinstance(samples, dict):
            raise ValueError("Please provide samples as dictionary")
        if not all(_p in samples.keys() for _p in self.required_parameters):
            from pesummary.utils.samples_dict import SamplesDict
            samples = SamplesDict(samples)
            samples.generate_all_posterior_samples(disable_remnant=True)
        if not all(_p in samples.keys() for _p in self.required_parameters):
            raise ValueError(
                "Failed to compute classification probabilities because "
                "the following parameters are required: {}".format(
                    ", ".join(self.required_parameters)
                )
            )
        self.samples = self._convert_samples(samples)

    @classmethod
    def from_file(cls, filename):
        """Initiate the classification class with samples stored in file

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
        """
        from pesummary.io import read
        f = read(filename)
        samples = f.samples_dict
        return cls(samples)

    @classmethod
    def classification_from_file(cls, filename, **kwargs):
        """Initiate the classification class with samples stored in file and
        return a dictionary containing the classification probabilities

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
        **kwargs: dict, optional
            all kwargs passed to cls.classification()
        """
        _cls = cls.from_file(filename)
        return _cls.classification(**kwargs)

    @classmethod
    def dual_classification_from_file(cls, filename, seed=123456789):
        """Initiate the classification class with samples stored in file and
        return a dictionary containing the classification probabilities
        generated from the raw samples ('default') and samples reweighted to
        a population inferred prior ('population')

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
        seed: int, optional
            random seed to use when reweighing to a population inferred prior
        """
        _cls = cls.from_file(filename)
        return _cls.dual_classification(seed=seed)

    @property
    def required_parameters(self):
        return ["mass_1_source", "mass_2_source", "a_1", "a_2"]

    @property
    def available_plots(self):
        return ["bar"]

    @staticmethod
    def round_probabilities(ptable, rounding=5):
        """Round the entries of a probability table

        Parameters
        ----------
        ptable: dict
            probability table
        rounding: int
            number of decimal places to round the entries of the probability
            table
        """
        for key, value in ptable.items():
            ptable[key] = np.round(value, rounding)
        return ptable

    def check_for_install(self, package=None):
        """Check that the required package is installed. If the package
        is not installed, raise an ImportError

        Parameters
        ----------
        package: str, optional
            name of package to check for install. Default None
        """
        if package is None:
            package = self.package
        if isinstance(package, str):
            package = [package]
        _not_available = []
        for _package in package:
            try:
                return importlib.import_module(_package)
            except ModuleNotFoundError:
                _not_available.append(_package)
        if len(_not_available):
            raise ImportError(
                "Unable to import {}. Unable to compute classification "
                "probabilities".format(" or ".join(package))
            )

    def _resample_to_population_prior(self, samples=None):
        """Use the pepredicates.rewt_approx_massdist_redshift function to
        reweight a pandas DataFrame to a population informed prior

        Parameters
        ----------
        samples: dict, optional
            pandas DataFrame containing posterior samples
        """
        import copy
        if not self.__class__.__name__ == "PEPredicates":
            _module = self.check_for_install(package="pepredicates")
        else:
            _module = self.module
        if samples is None:
            samples = self.samples
        _samples = copy.deepcopy(samples)
        if not all(param in _samples.keys() for param in ["redshift", "dist"]):
            raise ValueError(
                "Samples for redshift and distance required for population "
                "reweighting"
            )
        return _module.rewt_approx_massdist_redshift(_samples)

    def dual_classification(self, seed=123456789):
        """Return a dictionary containing the classification probabilities
        generated from the raw samples ('default') and samples reweighted to
        a population inferred prior ('population')

        Parameters
        ----------
        seed: int, optional
            random seed to use when reweighing to a population inferred prior
        """
        return {
            "default": self.classification(),
            "population": self.classification(population=True, seed=seed)
        }

    def classification(self, population=False, return_samples=False, seed=123456789):
        """return a dictionary containing the classification probabilities.
        These probabilities can either be generated from the raw samples or
        samples reweighted to a population inferred prior

        Parameters
        ----------
        population: Bool, optional
            if True, reweight the samples to a population informed prior and
            then calculate classification probabilities. Default False
        return_samples: Bool, optional
            if True, return the samples used as well as the classification
            probabilities
        seed: int, optional
            random seed to use when reweighing to a population inferred prior
        """
        if not population:
            ptable = self._compute_classification_probabilities()
            if return_samples:
                return self.samples, ptable
            return ptable
        np.random.seed(seed)
        _samples = PEPredicates._convert_samples(self.samples)
        reweighted_samples = self._resample_to_population_prior(
            samples=_samples
        )
        ptable = self._compute_classification_probabilities(
            samples=self._convert_samples(reweighted_samples)
        )
        if return_samples:
            return _samples, ptable
        return ptable

    def _compute_classification_probabilities(self, samples=None):
        """Base function to compute classification probabilities

        Parameters
        ----------
        samples: dict, optional
            samples to use for computing the classification probabilities
            Default None.
        """
        if samples is None:
            samples = self.samples
        return samples, {}

    def plot(
        self, samples=None, probabilities=None, type="bar", population=False
    ):
        """Generate a plot showing the classification probabilities

        Parameters
        ----------
        samples: dict, optional
            samples to use for plotting. Default None
        probabilities: dict, optional
            dictionary giving the classification probabilities. Default None
        type: str, optional
            type of plot to produce
        population: Bool, optional
            if True, reweight the posterior samples to a population informed
            prior before computing the classification probabilities for
            plotting. Only used when probabilities=None. Default False
        """
        if type not in self.available_plots:
            raise ValueError(
                "Unknown plot '{}'. Please select a plot from {}".format(
                    type, ", ".join(self.available_plots)
                )
            )
        if (probabilities is None) or ((samples is None) and population):
            s, p = self.classification(
                population=population, return_samples=True
            )
            if probabilities is None:
                probabilities = p
            if ((samples is None) and population):
                samples = s
        return getattr(self, "_{}_plot".format(type))(samples, probabilities)

    def _bar_plot(self, samples, probabilities):
        """Generate a bar plot showing classification probabilities

        Parameters
        ----------
        samples: dict
            samples to use for plotting
        probabilities: dict
            dictionary giving the classification probabilities.
        """
        from pesummary.gw.plots.plot import _classification_plot
        return _classification_plot(probabilities)


class PEPredicates(_Base):
    """Class for generating source classification probabilities, i.e.
    the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS), or a binary originating from the mass gap, p(MassGap)

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities

    Attributes
    ----------
    available_plots: list
        list of available plotting types

    Methods
    -------
    classification:
        return a dictionary containing the classification probabilities. These
        probabilities can either be generated from the raw samples or samples
        reweighted to a population inferred prior
    dual_classification:
        return a dictionary containing the classification probabilities
        generated from the raw samples ('default') and samples reweighted to
        a population inferred prior ('population')
    plot:
        generate a plot showing the classification probabilities
    """
    def __init__(self, samples):
        self.package = "pepredicates"
        super(PEPredicates, self).__init__(samples)

    @property
    def _default_probabilities(self):
        return {
            'BNS': self.module.BNS_p, 'NSBH': self.module.NSBH_p,
            'BBH': self.module.BBH_p, 'MassGap': self.module.MG_p
        }

    @property
    def available_plots(self):
        _plots = super(PEPredicates, self).available_plots
        _plots.extend(["pepredicates"])
        return _plots

    @staticmethod
    def mapping(reverse=False):
        _mapping = {
            "mass_1_source": "m1_source", "mass_2_source": "m2_source",
            "luminosity_distance": "dist", "redshift": "redshift",
            "a_1": "a1", "a_2": "a2", "tilt_1": "tilt1", "tilt_2": "tilt2"
        }
        if reverse:
            return {item: key for key, item in _mapping.items()}
        return _mapping

    @staticmethod
    def _convert_samples(samples):
        """Convert dictionary of posterior samples to required form
        needed for pepredicates

        Parameters
        ----------
        samples: dict
            samples to use for computing the classification probabilities
        """
        import pandas as pd
        mapping = PEPredicates.mapping()
        reverse = PEPredicates.mapping(reverse=True)
        if not all(param in samples.keys() for param in reverse.keys()):
            _samples = {
                new: samples[original] for original, new in mapping.items()
                if original in samples.keys()
            }
        else:
            _samples = samples.copy()
        return pd.DataFrame.from_dict(_samples)

    def _compute_classification_probabilities(self, samples=None, rounding=5):
        """Compute classification probabilities

        Parameters
        ----------
        samples: dict, optional
            samples to use for computing the classification probabilities
            Default None.
        rounding: int, optional
            number of decimal places to round entries of probability table.
            Default 5
        """
        samples, _ = super()._compute_classification_probabilities(
            samples=samples
        )
        ptable = self.module.predicate_table(
            self._default_probabilities, samples
        )
        if rounding is not None:
            return self.round_probabilities(ptable, rounding=rounding)
        return ptable

    def _pepredicates_plot(self, samples, probabilities):
        """Generate the a plot using the pepredicates.plot_predicates function
        showing classification probabilities

        Parameters
        ----------
        samples: dict
            samples to use for plotting
        probabilities: dict
            dictionary giving the classification probabilities.
        """
        from pesummary.core.plots.figure import ExistingFigure
        if samples is None:
            from pesummary.utils.utils import logger
            logger.debug(
                "No samples provided for plotting. Using cached array."
            )
            samples = self.samples
        idxs = {
            "BBH": self.module.is_BBH(samples),
            "BNS": self.module.is_BNS(samples),
            "NSBH": self.module.is_NSBH(samples),
            "MassGap": self.module.is_MG(samples)
        }
        return ExistingFigure(
            self.module.plot_predicates(
                idxs, samples, probs=probabilities
            )
        )


class PAstro(_Base):
    """Class for generating EM-Bright classification probabilities, i.e.
    the probability that the binary has a neutron star, p(HasNS), and
    the probability that the remnant is observable, p(HasRemnant).

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities

    Attributes
    ----------
    available_plots: list
        list of available plotting types

    Methods
    -------
    classification:
        return a dictionary containing the classification probabilities. These
        probabilities can either be generated from the raw samples or samples
        reweighted to a population inferred prior
    dual_classification:
        return a dictionary containing the classification probabilities
        generated from the raw samples ('default') and samples reweighted to
        a population inferred prior ('population')
    plot:
        generate a plot showing the classification probabilities
    """
    def __init__(self, samples):
        self.package = "ligo.em_bright.em_bright"
        super(PAstro, self).__init__(samples)

    @property
    def required_parameters(self):
        _parameters = super(PAstro, self).required_parameters
        _parameters.extend(["tilt_1", "tilt_2"])
        return _parameters

    @staticmethod
    def _convert_samples(samples):
        """Convert dictionary of posterior samples to required form
        needed for ligo.computeDiskMass

        Parameters
        ----------
        samples: dict
            samples to use for computing the classification probabilities
        """
        _samples = {}
        try:
            reverse = PEPredicates.mapping(reverse=True)
            for key, item in reverse.items():
                if key in samples.keys():
                    samples[item] = samples.pop(key)
        except KeyError:
            pass
        for key, item in samples.items():
            _samples[key] = np.asarray(item)
        return _samples

    def _compute_classification_probabilities(self, samples=None, rounding=5):
        """Compute classification probabilities

        Parameters
        ----------
        samples: dict, optional
            samples to use for computing the classification probabilities
            Default None.
        rounding: int, optional
            number of decimal places to round entries of probability table.
            Default 5
        """
        samples, _ = super()._compute_classification_probabilities(
            samples=samples
        )
        probs = self.module.source_classification_pe_from_table(samples)
        ptable = {"HasNS": probs[0], "HasRemnant": probs[1]}
        if rounding is not None:
            return self.round_probabilities(ptable, rounding=rounding)
        return ptable


class Classify(_Base):
    """Class for generating source classification and EM-Bright probabilities,
    i.e. the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS), or a binary originating from the mass gap, p(MassGap), the
    probability that the binary has a neutron star, p(HasNS), and the
    probability that the remnant is observable, p(HasRemnant).
    """
    @property
    def required_parameters(self):
        _parameters = super(Classify, self).required_parameters
        _parameters.extend(["tilt_1", "tilt_2"])
        return _parameters

    def check_for_install(self, *args, **kwargs):
        pass

    def _convert_samples(self, samples):
        return samples

    def classification(self, **kwargs):
        """return a dictionary containing the classification probabilities.
        These probabilities can either be generated from the raw samples or
        samples reweighted to a population inferred prior

        Parameters
        ----------
        population: Bool, optional
            if True, reweight the samples to a population informed prior and
            then calculate classification probabilities. Default False
        return_samples: Bool, optional
            if True, return the samples used as well as the classification
            probabilities
        """
        probs = PEPredicates(self.samples).classification(**kwargs)
        pastro = PAstro(self.samples).classification(**kwargs)
        probs.update(pastro)
        return probs


def classify(*args, **kwargs):
    """Generate source classification and EM-Bright probabilities,
    i.e. the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS), or a binary originating from the mass gap, p(MassGap), the
    probability that the binary has a neutron star, p(HasNS), and the
    probability that the remnant is observable, p(HasRemnant).

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities
    population: Bool, optional
        if True, reweight the samples to a population informed prior and
        then calculate classification probabilities. Default False
    return_samples: Bool, optional
        if True, return the samples used as well as the classification
        probabilities
    """
    return Classify(*args).classification(**kwargs)
