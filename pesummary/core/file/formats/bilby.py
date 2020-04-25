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
import numpy as np
from pesummary.core.file.formats.base_read import Read
from pesummary.core.plots.latex_labels import latex_labels
from pesummary import conf
from pesummary.utils.utils import logger


class Bilby(Read):
    """PESummary wrapper of `bilby` (https://git.ligo.org/lscsoft/bilby). The
    path_to_results_file argument will be passed directly to
    `bilby.core.result.read_in_result`. All functions therefore use `bilby`
    methods and requires `bilby` to be installed.

    Attributes
    ----------
    path_to_results_file: str
        path to the results file that you wish to read in with `bilby`.

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    injection_parameters: dict
        dictionary of injection parameters extracted from the result file
    prior: dict
        dictionary of prior samples keyed by parameters. The prior functions
        are evaluated for 5000 samples.

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, path_to_results_file):
        super(Bilby, self).__init__(path_to_results_file)
        self.load(self._grab_data_from_bilby_file)

    @classmethod
    def load_file(cls, path):
        if not os.path.isfile(path):
            raise Exception("%s does not exist" % (path))
        return cls(path)

    @staticmethod
    def grab_priors(bilby_object, nsamples=5000):
        """Draw samples from the prior functions stored in the bilby file
        """
        from pesummary.utils.samples_dict import Array

        f = bilby_object
        try:
            samples = f.priors.sample(size=nsamples)
            priors = {key: Array(samples[key]) for key in samples}
        except Exception as e:
            logger.info("Failed to draw prior samples because {}".format(e))
            priors = {}
        return priors

    @staticmethod
    def grab_extra_kwargs(bilby_object):
        """Grab any additional information stored in the lalinference file
        """
        f = bilby_object
        kwargs = {"sampler": {
            conf.log_evidence: np.round(f.log_evidence, 2),
            conf.log_evidence_error: np.round(f.log_evidence_err, 2),
            conf.log_bayes_factor: np.round(f.log_bayes_factor, 2),
            conf.log_noise_evidence: np.round(f.log_noise_evidence, 2)},
            "meta_data": {}}
        return kwargs

    @staticmethod
    def _grab_data_from_bilby_file(path):
        """Load the results file using the `bilby` library
        """
        from bilby.core.result import read_in_result

        bilby_object = read_in_result(filename=path)
        posterior = bilby_object.posterior
        # Drop all non numeric bilby data outputs
        posterior = posterior.select_dtypes(include=[float, int])
        parameters = list(posterior.keys())
        samples = posterior.to_numpy().real
        injection = bilby_object.injection_parameters
        if injection is None:
            injection = {i: j for i, j in zip(
                parameters, [float("nan")] * len(parameters))}
        else:
            for i in parameters:
                if i not in injection.keys():
                    injection[i] = float("nan")

        if all(i for i in (
               bilby_object.constraint_parameter_keys,
               bilby_object.search_parameter_keys,
               bilby_object.fixed_parameter_keys)):
            for key in (
                    bilby_object.constraint_parameter_keys
                    + bilby_object.search_parameter_keys
                    + bilby_object.fixed_parameter_keys):
                if key not in latex_labels:
                    label = bilby_object.get_latex_labels_from_parameter_keys(
                        [key])[0]
                    latex_labels[key] = label
        try:
            extra_kwargs = Bilby.grab_extra_kwargs(bilby_object)
        except Exception:
            extra_kwargs = {"sampler": {}, "meta_data": {}}
        extra_kwargs["sampler"]["nsamples"] = len(samples)
        try:
            version = bilby_object.version
        except Exception as e:
            version = None
        prior_samples = Bilby.grab_priors(bilby_object, nsamples=len(samples))
        return {
            "parameters": parameters,
            "samples": samples,
            "injection": injection,
            "version": version,
            "kwargs": extra_kwargs,
            "prior": prior_samples
        }

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass
