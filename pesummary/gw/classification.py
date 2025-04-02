# Licensed under an MIT style license -- see LICENSE.md

import importlib
import os
import numpy as np
from scipy.special import logsumexp
from pesummary.gw.cosmology import hubble_distance, hubble_parameter
from pesummary.utils.utils import logger

__author__ = [
    "Anarya Ray <anarya.ray@ligo.org>",
    "Charlie Hoy <charlie.hoy@ligo.org>"
]

class _Base():
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
        self.samples = {key: np.array(value) for key, value in samples.items()}

    @property
    def required_parameters(self):
        return ["mass_1_source", "mass_2_source"]

    @property
    def available_plots(self):
        return ["bar"]

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Initiate the classification class with samples stored in file

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
        """
        from pesummary.io import read
        f = read(filename)
        samples = f.samples_dict
        return cls(samples, **kwargs)

    @classmethod
    def classification_from_file(cls, filename, **kwargs):
        """Initiate the classification class with samples stored in file and
        return a dictionary containing the classification probabilities

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
        **kwargs: dict, optional
            all kwargs passed to cls.from_file()
        """
        _cls = cls.from_file(filename, **kwargs)
        return _cls.classification()

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

    def plot(self, probabilities, type="bar"):
        """Generate a plot showing the classification probabilities

        Parameters
        ----------
        probabilities: dict
            dictionary giving the classification probabilities
        type: str, optional
            type of plot to produce
        """
        if type not in self.available_plots:
            raise ValueError(
                "Unknown plot '{}'. Please select a plot from {}".format(
                    type, ", ".join(self.available_plots)
                )
            )
        return getattr(self, "_{}_plot".format(type))(probabilities)

    def _bar_plot(self, probabilities):
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

    def save_to_file(self, file_name, probabilities, outdir="./", **kwargs):
        """Save classification data to json file

        Parameters
        ----------
        file_name: str
            name of the file name that you wish to use
        probabilities: dict
            dictionary of probabilities you wish to save to file
        """
        from pesummary.io import write
        write(
            list(probabilities.keys()), list(probabilities.values()),
            file_format="json", outdir=outdir, filename=file_name,
            dataset_name=None, indent=None, **kwargs
        )


class PAstro(_Base):
    """Class for generating source classification probabilities, i.e.
    the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS). We use a rate and evidence based estimate, as detailed in
    https://dcc.ligo.org/LIGO-G2301521 and described below:
    
    The probability for a given classification is simply:

    ..math ::
        fraction = \frac{R_{\alpha}Z_{\alpha}}{\sum_{\beta}R_{\beta}Z_{\beta}}
        P(H_{\alpha}|d) = (1 - P_{\text{Terr}}^{pipeline}) fraction

    where :math:`Z_{\alpha}` is the Bayesian evidence for each category, estimated as,

    ..math ::
        fraction = \frac{p(m_{1s,i},m_{2s,i},z_i|\alpha)}{p(m_{1d,i}m_{2d,i},d_{L,i})\times \frac{dd_L}{dz}\frac{1}{(1+z_i)^2}}
        Z_{\alpha}=\frac{Z_{PE}}{N_{samp}}\sum_{i\sim\text{posterior}}^{N_{samp}} fraction

    and we use the following straw-person population prior for classifying the sources
    into different astrophysical categories

    ..math ::
        fraction = \frac{m_{1s}^{\alpha}m_{2s}^{\beta}}{\text{min}(m_{1s},m_{2s,max})^{\beta+1}-m_{2s,min}^{\beta+1}}
        p(m_{1s},m_{2s},z|\alpha)  \propto fraction \frac{dV_c}{dz}\frac{1}{1+z}

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities
    category_data: dict, optional
        dictionary of summary data (rates and population hyper parameters) for each
        category. Default None
    distance_prior: class, optional
        class describing the distance prior used when generating the posterior
        samples. It must have a method `ln_prob` for returning the log prior
        probability for a given distance. Default
        bilby.gw.prior.UniformSourceFrame
    cosmology: str, optional
        cosmology you wish to use. Default Planck15
    terrestrial_probability: float, optional
        probability that the observed gravitational-wave is of terrestrial
        origin. Default None.
    catch_terrestrial_probability_error: bool, optional
        catch the ValueError raised when no terrestrial_probability is provided.
        If True, terrestrial_probability is set to 0. Default False

    Attributes
    ----------
    available_plots: list
        list of available plotting types

    Methods
    -------
    classification:
        return a dictionary containing the classification probabilities
    """
    defaults = {"BBH": None, "BNS": None, "NSBH": None}
    def __init__(
        self, samples, category_data=None, distance_prior=None,
        cosmology="Planck15", terrestrial_probability=None,
        catch_terrestrial_probability_error=False
    ):
        self.package = ["bilby.gw.prior"]
        super(PAstro, self).__init__(samples)
        self.distance_prior = distance_prior
        if distance_prior is None:
            logger.debug(
                f"No distance prior provided. Assuming the posterior samples "
                f"were obtained with a 'UniformSourceFrame' prior (with a "
                f"{cosmology} cosmology), as defined in 'bilby'."
            )
            self.distance_prior = self.module.UniformSourceFrame(
                minimum=float(np.min(self.samples["luminosity_distance"]) * 0.5),
                maximum=float(np.max(self.samples["luminosity_distance"]) * 1.5),
                name="luminosity_distance", unit="Mpc",
                cosmology=cosmology
            )
        if category_data is not None and os.path.isfile(category_data):
            import yaml
            with open(category_data, "r") as f:
                config = yaml.full_load(f)
            category_data = config["pop_prior"]
            for key, value in config["Rates"].items():
                category_data[key]["rate"] = float(value)
        self.category_data = category_data
        self.cosmology = cosmology
        self.terrestrial_probability = terrestrial_probability
        self.catch_terrestrial_probability_error = catch_terrestrial_probability_error

    @property
    def required_parameters(self):
        params = super(PAstro, self).required_parameters
        params.extend(["luminosity_distance", "redshift"])
        return params

    def _salpeter_prior(self, alpha, m1_bounds, m2_bounds, zmax, beta):
        """Calculate and return the log probabilities assuming a Salpeter population
        prior

        Parameters
        ----------
        alpha: float
            index of the powerlaw for the primary mass prior
        m1_bounds: list
            list of length 2 which contains the minimum (index 0) and maximum (index 1)
            primary mass
        m2_bounds: list
            list of length 2 which contains the minimum (index 0) and maximum (index 1)
            secondary mass
        zmax: float
            maximum redshift
        beta: float
            index of the powerlaw for the secondary mass prior
        """
        if alpha != -1:
            upper = m1_bounds[1]**(1. + alpha)
            lower = m1_bounds[0]**(1. + alpha)
            log_m1_norm = np.log((1. + alpha) / (upper - lower))
        else:
            log_m1_norm = -np.log(np.log(m1_bounds[1] / m1_bounds[0]))
        m2_max = np.min(
            np.array(
                [
                    m2_bounds[1] * np.ones(len(self.samples["mass_1_source"])),
                    self.samples["mass_1_source"]
                ]
            ), axis=0
        )
        if beta != -1:
            upper = m2_max**(1. + beta)
            lower = m2_bounds[0]**(1. + beta)
            log_m2_norm = np.log((1. + beta) / (upper - lower))
        else:
            log_m2_norm = -np.log(np.log(m2_max / m2_bounds[0]))
        z_prior = self.module.UniformSourceFrame(
            name="redshift", minimum=0., maximum=zmax, unit=None
        )
        logprob = (
            alpha * np.log(self.samples["mass_1_source"]) +
            beta * np.log(self.samples["mass_2_source"]) +
            log_m1_norm + log_m2_norm +
            z_prior.ln_prob(self.samples["redshift"])
        )
        logprob[np.isnan(logprob)] = -np.inf
        logprob += np.log(
            (
                (self.samples["mass_1_source"] >= self.samples["mass_2_source"]) *
                (self.samples["mass_1_source"] <= m1_bounds[1]) *
                (self.samples["mass_1_source"] >= m1_bounds[0]) *
                (m2_max >= self.samples["mass_2_source"]) *
                (self.samples["mass_2_source"] >= m2_bounds[0])
            ).astype(int)
        )
        return logprob

    def classification(self, rounding=5):
        if self.category_data is None:
            raise ValueError(
                "No category data provided to estimate rate weighted evidence. "
                "Unable to calculate source probabilities."
            )
        required_data = [
            "rate", "alpha", "m1_bounds", "m2_bounds", "zmax", "beta"
        ]
        for value in self.category_data.values():
            if not all(_ in value.keys() for _ in required_data):
                raise ValueError(
                    "Please provide {} for each category".format(
                    ", ".join(required_data)
                )
            )
        if self.terrestrial_probability is None:
            if self.catch_terrestrial_probability_error:
                logger.debug(
                    "Setting terrestrial probability to 0 for classification "
                    "probabilities"
                )
                self.terrestrial_probability = 0.
            else:
                raise ValueError(
                    "Please provide a terrestrial probability in order to calculate "
                    "classification probabilities. Alternatively pass the kwarg "
                    "catch_terrestrial_probability_error=True"
                )
        elif self.terrestrial_probability >= 1:
            raise ValueError(
                "Terrestrial probability >= 1 meaning that there is no "
                "probability that the source is a BBH, NSBH or BNS"
            )
        # evaluate population prior
        pop_log_priors = {
            category: self._salpeter_prior(
                config["alpha"], config["m1_bounds"], config["m2_bounds"],
                config["zmax"], config["beta"]
            ) for category, config in self.category_data.items()
        }
        # evaluate pe-prior
        pe_log_prior = self.distance_prior.ln_prob(
            self.samples["luminosity_distance"]
        )
    
        # evaluate detector frame to source frame jacobian
        hd = hubble_distance(self.cosmology)
        hp = hubble_parameter(self.cosmology, self.samples["redshift"])
        ddL_dz = (
            self.samples["luminosity_distance"] / (1 + self.samples["redshift"]) +
            (1. + self.samples["redshift"]) * hd / hp
        )
        log_jacobian = -np.log(ddL_dz) - 2. * np.log1p(self.samples["redshift"])
    
        #compute evidence
        rate_weighted_evidence = {
            category: config["rate"] * np.exp(
                logsumexp(
                    pop_log_priors[category] - pe_log_prior -
                    np.log(len(self.samples["mass_1_source"])) + log_jacobian
                )
            ) for category, config in self.category_data.items()
        }
        #compute p_astro
        total_evidence = np.sum(list(rate_weighted_evidence.values()))
        ptable = {
            category: (1. - self.terrestrial_probability) * rz / total_evidence
            for category, rz in rate_weighted_evidence.items()
        }
        ptable["Terrestrial"] = self.terrestrial_probability
        if rounding is not None:
            return self.round_probabilities(ptable, rounding=rounding)
        return ptable

    def _samples_plot(self, probabilities):
        """Generate a sample distribution plot showing classification
        probabilities

        Parameters
        ----------
        probabilities: dict
            dictionary giving the classification probabilities.
        """
        from pesummary.gw.plots.plot import _classification_samples_plot
        return _classification_samples_plot(
            self.samples["mass_1_source"], self.samples["mass_2_source"],
            probabilities
        )


class EMBright(_Base):
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
        return a dictionary containing the classification probabilities
    """
    defaults = {"HasNS": None, "HasRemnant": None, "HasMassGap": None}
    def __init__(self, samples, **kwargs):
        self.package = ["ligo.em_bright.em_bright"]
        super(EMBright, self).__init__(samples)

    @property
    def required_parameters(self):
        params = super(EMBright, self).required_parameters
        params.extend(["a_1", "a_2", "tilt_1", "tilt_2"])
        return params

    def classification(self, rounding=5, **kwargs):
        probs = self.module.source_classification_pe_from_table(self.samples)
        ptable = {"HasNS": probs[0], "HasRemnant": probs[1], "HasMassGap": probs[2]}
        if rounding is not None:
            return self.round_probabilities(ptable, rounding=rounding)
        return ptable


class Classify(_Base):
    """Class for generating source classification and EM-Bright probabilities,
    i.e. the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS), the probability that the binary has a neutron star, p(HasNS), and
    the probability that the remnant is observable, p(HasRemnant).
    """
    @property
    def required_parameters(self):
        params = super(Classify, self).required_parameters
        params.extend(
            ["luminosity_distance", "redshift", "a_1", "a_2", "tilt_1", "tilt_2"]
        )
        return params

    def check_for_install(self, *args, **kwargs):
        pass

    @classmethod
    def classification_from_file(cls, filename, **kwargs):
        """Initiate the classification class with samples stored in file and
        return a dictionary containing the classification probabilities

        Parameters
        ----------
        filename: str
            path to file you wish to initiate the classification class with
            path to file you wish to initiate the classification class with
        **kwargs: dict, optional
            all kwargs passed to cls.from_file()
        """
        _cls = PAstro.from_file(filename, **kwargs)
        pastro = _cls.classification()
        _cls = EMBright.from_file(filename, **kwargs)
        embright = _cls.classification()
        pastro.update(embright)
        return pastro

    def classification(self, rounding=5, **kwargs):
        """return a dictionary containing the classification probabilities.
        """
        probs = PAstro(self.samples, **kwargs).classification(
            rounding=rounding
        )
        pastro = EMBright(self.samples, **kwargs).classification(
            rounding=rounding
        )
        probs.update(pastro)
        return probs


def classify(*args, **kwargs):
    """Generate source classification and EM-Bright probabilities,
    i.e. the probability that it is consistent with originating from a binary
    black hole, p(BBH), neutron star black hole, p(NSBH), binary neutron star,
    p(BNS), the probability that the binary has a neutron star, p(HasNS), and
    the probability that the remnant is observable, p(HasRemnant).

    Parameters
    ----------
    samples: dict
        dictionary of posterior samples to use for generating classification
        probabilities
    """
    return Classify(*args).classification(**kwargs)
