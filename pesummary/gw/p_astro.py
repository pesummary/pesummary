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

try:
    from ligo.computeDiskMass import computeDiskMass
    PASTRO = True
except ImportError:
    PASTRO = False

import numpy as np
from pesummary.utils.utils import logger


def get_probabilities(samples):
    """Return `HasNS` and `HasRemnant` probabilities from a dictionary of
    samples

    Parameters
    ----------
    samples: dict
        dictionary of samples
    """
    from pesummary.utils.utils import RedirectLogger

    default_error = (
        "Failed to generate `HasNS` and `HasRemnant` probabilities because {}"
    )
    try:
        with RedirectLogger("p_astro", level="DEBUG") as redirector:
            data = PAstro.classifications(samples)
    except ImportError:
        logger.warn(default_error.format("'p_astro' is not installed"))
        data = None
    except Exception as e:
        logger.warn(default_error.format("%s" % (e)))
        data = None
    return data


class PAstro(object):
    """Class to handle the p_astro package
    """
    @staticmethod
    def check_for_install():
        """Check that p_astro is installed
        """
        if not PASTRO:
            raise ImportError(
                "Failed to import 'p_astro' packages and therefore unable to "
                "calculate `HasNS` and `HasRemnant` probabilities"
            )

    @staticmethod
    def classifications(samples):
        """Calculate the `HasNS` and `HasRemnant` probabilities

        Parameters
        ----------
        samples: dict
            dictionary of samples
        """
        PAstro.check_for_install()
        required_params = ["mass_1", "mass_2", "a_1", "a_2"]
        parameters = list(samples.keys())

        if not all(i in parameters for i in required_params):
            raise Exception(
                "Failed to generate `HasNS` and `HasRemnant` probabilities "
                "because not all required parameters have been provided."
            )
        M_rem = computeDiskMass(
            samples["mass_1"], samples["mass_2"], samples["a_1"], samples["a_2"]
        )
        prediction_ns = float(
            np.sum(samples["mass_2"] <= 3.0) / len(samples["mass_2"])
        )
        prediction_em = float(np.sum(M_rem > 0) / len(M_rem))
        return {"HasNS": prediction_ns, "HasRemnant": prediction_em}
