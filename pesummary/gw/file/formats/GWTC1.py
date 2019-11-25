# Copyright (C) 2019  Charlie Hoy <charlie.hoy@ligo.org>
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

import h5py
import numpy as np
try:
    from glue.ligolw import ligolw
    from glue.ligolw import lsctables
    from glue.ligolw import utils as ligolw_utils
    GLUE = True
except ImportError:
    GLUE = False

from pesummary.gw.file.formats.base_read import GWRead
from pesummary.gw.file import conversions as con
from pesummary.utils.utils import logger


class GWTC1(GWRead):
    """PESummary wrapper of the GWTC1 sample release

    Attributes
    ----------
    path_to_results_file: str
        path to the results file you wish to load in with `GWTC1`
    """
    def __init__(self, path_to_results_file, injection_file=None):
        super(GWTC1, self).__init__(path_to_results_file)
        self.load(self._grab_data_from_GWTC1_file)

    @classmethod
    def load_file(cls, path, injection_file=None):
        if not os.path.isfile(path):
            raise IOError("%s does not exist" % (path))
        if injection_file and not os.path.isfile(injection_file):
            raise IOError("%s does not exist" % (path))
        return cls(path, injection_file=injection_file)

    @staticmethod
    def grab_extra_kwargs(path):
        """
        """
        return {"sampler": {}, "meta_data": {}}

    @staticmethod
    def grab_priors(obj):
        """
        """
        from pesummary.utils.utils import SamplesDict

        keys = list(obj.keys())
        if "prior" in keys or "priors" in keys:
            data = obj["prior"] if "prior" in keys else obj["priors"]
            parameters = list(data.dtype.names)
            samples = [list(i) for i in data]
            return SamplesDict(parameters, np.array(samples).T)
        logger.warn(
            "Failed to draw prior samples because there is not an entry for "
            "'prior' or 'priors' in the result file"
        )
        return {}

    @staticmethod
    def _grab_data_from_GWTC1_file(path):
        """
        """
        f = h5py.File(path)
        keys = list(f.keys())
        if "Overall_posterior" in keys or "overall_posterior" in keys:
            data = \
                f["overall_posterior"] if "overall_posterior" in keys else \
                f["Overall_posterior"]
        else:
            f.close()
            raise Exception(
                "Failed to read in result file because there was no group "
                "called 'Overall_posterior' or 'overall_posterior'"
            )

        parameters = list(data.dtype.names)
        samples = [list(i) for i in data]
        extra_kwargs = GWTC1.grab_extra_kwargs(path)
        extra_kwargs["sampler"]["nsamples"] = len(samples)
        prior_samples = GWTC1.grab_priors(f)
        version = None
        f.close()
        return {
            "parameters": parameters,
            "samples": samples,
            "injection": None,
            "version": version,
            "kwargs": extra_kwargs,
            "prior": prior_samples
        }

    @property
    def calibration_data_in_results_file(self):
        """
        """
        return None
