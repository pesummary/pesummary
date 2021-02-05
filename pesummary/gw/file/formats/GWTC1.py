# Licensed under an MIT style license -- see LICENSE.md

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

from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead
from pesummary.gw import conversions as con
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def open_GWTC1(path, path_to_samples=None, **kwargs):
    """Grab the parameters and samples in a bilby file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    path_to_samples: str, optional
        path to the group containing the posterior samples you wish to load
    """
    f = h5py.File(path, 'r')
    keys = list(f.keys())
    if path_to_samples is not None:
        data = f[path_to_samples]
    elif "Overall_posterior" in keys or "overall_posterior" in keys:
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
    data = {
        "parameters": parameters,
        "samples": samples,
        "injection": None,
        "version": version,
        "kwargs": extra_kwargs
    }
    if len(prior_samples):
        data["prior"] = {"samples": prior_samples}
    return data


class GWTC1(GWSingleAnalysisRead):
    """PESummary wrapper of the GWTC1 sample release

    Attributes
    ----------
    path_to_results_file: str
        path to the results file you wish to load in with `GWTC1`
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples
    """
    def __init__(self, path_to_results_file, injection_file=None, **kwargs):
        super(GWTC1, self).__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_GWTC1_file)

    @classmethod
    def load_file(cls, path, injection_file=None, **kwargs):
        if injection_file and not os.path.isfile(injection_file):
            raise IOError("%s does not exist" % (path))
        return super(GWTC1, cls).load_file(
            path, injection_file=injection_file, **kwargs
        )

    @staticmethod
    def grab_extra_kwargs(path):
        """
        """
        return {"sampler": {}, "meta_data": {}}

    @staticmethod
    def grab_priors(obj):
        """
        """
        from pesummary.utils.samples_dict import SamplesDict

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
    def _grab_data_from_GWTC1_file(path, path_to_samples=None, **kwargs):
        """
        """
        return open_GWTC1(path, path_to_samples=path_to_samples, **kwargs)

    @property
    def calibration_data_in_results_file(self):
        """
        """
        return None
