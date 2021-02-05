# Licensed under an MIT style license -- see LICENSE.md

try:
    import pepredicates as pep
    PEP = True
except ImportError:
    PEP = False

import numpy as np
import pandas as pd
from pesummary.core.plots.figure import ExistingFigure
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def get_classifications(samples):
    """Return the classifications from a dictionary of samples

    Parameters
    ----------
    samples: dict
        dictionary of samples
    """
    from pesummary.utils.utils import RedirectLogger

    default_error = (
        "Failed to generate source classification probabilities because {}"
    )
    try:
        with RedirectLogger("PEPredicates", level="DEBUG") as redirector:
            parameters = list(samples.keys())
            samples = [
                [samples[parameter][j] for parameter in parameters] for j in
                range(len(samples[parameters[0]]))
            ]
            data = PEPredicates.classifications(samples, parameters)
        classifications = {
            "default": data[0], "population": data[1]
        }
    except ImportError:
        logger.warning(
            default_error.format("'PEPredicates' is not installed")
        )
        classifications = None
    except Exception as e:
        logger.warning(default_error.format("%s" % (e)))
        classifications = None
    return classifications


class PEPredicates(object):
    """Class to handle the PEPredicates package
    """
    @staticmethod
    def default_predicates():
        """Set the default possibilities
        """
        default = {
            'BNS': pep.BNS_p,
            'NSBH': pep.NSBH_p,
            'BBH': pep.BBH_p,
            'MassGap': pep.MG_p}
        return default

    @staticmethod
    def check_for_install():
        """Check that predicates is installed
        """
        if not PEP:
            raise ImportError(
                "Failed to import 'predicates' and therefore unable to "
                "calculate astro/terrestrial probabilities")

    @staticmethod
    def convert_to_PEPredicated_data_frame(samples, parameters):
        """Convert the inputs to a pandas data frame compatible with
        PEPredicated

        Parameters
        ----------
        samples: list
            list of samples for a specific result file
        parameters: list
            list of parameters for a specific result file
        """
        PEPredicates.check_for_install()
        psamps = pd.DataFrame()

        mapping = {"mass_1_source": "m1_source",
                   "mass_2_source": "m2_source",
                   "luminosity_distance": "dist",
                   "redshift": "redshift",
                   "a_1": "a1",
                   "a_2": "a2"}

        if not all(i in parameters for i in list(mapping.keys())):
            raise Exception(
                "Failed to generate classification probabilities because not "
                "all required parameters have been provided.")
        for num, i in enumerate(list(mapping.keys())):
            psamps[mapping[i]] = [j[parameters.index(i)] for j in samples]
        return psamps

    @staticmethod
    def resample_to_population(samples):
        """Return samples that have been resampled to a sensibile population

        Parameters
        ----------
        samples: list
            list of samples for a specific result file
        """
        PEPredicates.check_for_install()
        return pep.rewt_approx_massdist_redshift(samples)

    @staticmethod
    def default_classification(samples, parameters):
        """Return the source classification probabilities using the default
        prior used

        Parameters
        ----------
        samples: list
            list of samples for a specific result file
        """
        PEPredicates.check_for_install()
        core_samples = PEPredicates.convert_to_PEPredicated_data_frame(
            samples, parameters)
        ptable = pep.predicate_table(
            PEPredicates.default_predicates(), core_samples)
        for key, value in ptable.items():
            ptable[key] = np.round(value, 5)
        return ptable

    @staticmethod
    def population_classification(samples, parameters):
        """Return the source classification probabilities using a population
        prior

        Parameters
        ----------
        samples: list
            list of samples for a specific result file
        """
        PEPredicates.check_for_install()
        core_samples = PEPredicates.convert_to_PEPredicated_data_frame(
            samples, parameters)
        psamps_resamples = PEPredicates.resample_to_population(core_samples)
        ptable = pep.predicate_table(
            PEPredicates.default_predicates(), psamps_resamples)
        for key, value in ptable.items():
            ptable[key] = np.round(value, 5)
        return ptable

    @staticmethod
    def classifications(samples, parameters):
        """Return the source classification probabilities using both the default
        prior used in the analysis and the population prior
        """
        pop = PEPredicates.population_classification(samples, parameters)
        default = PEPredicates.default_classification(samples, parameters)
        return default, pop

    @staticmethod
    def plot(samples, parameters, population_prior=True):
        """Make a plot of the samples classified by type

        Parameters
        ----------
        samples: list
            list of samples for a specific result file
        """
        logger.debug("Generating the PEPredicates plot")
        PEPredicates.check_for_install()
        core_samples = PEPredicates.convert_to_PEPredicated_data_frame(
            samples, parameters)
        if population_prior:
            psamps_resamples = PEPredicates.resample_to_population(core_samples)
        else:
            psamps_resamples = core_samples
        ptable = {"BBH": pep.is_BBH(psamps_resamples),
                  "BNS": pep.is_BNS(psamps_resamples),
                  "NSBH": pep.is_NSBH(psamps_resamples),
                  "MassGap": pep.is_MG(psamps_resamples)}
        fig = ExistingFigure(pep.plot_predicates(ptable, psamps_resamples))
        return fig
