# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from ....utils.parameters import Parameters

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _LazyRead(object):
    """Base class for a lazy read object

    Parameters
    ----------
    drop_non_numeric: bool, optional
        if True, remove all non_numeric values from the posterior samples
        table
    remove_nan_likelihood_samples: bool, optional
        if True. remove all rows in the posterior samples table that have 'nan'
        likelihood

    Attributes
    ----------
    parameters: pesummary.utils.parameters.Parameters
        list of parameters in the posterior samples table
    samples: np.ndarray
        2D array of samples in the posterior samples table
    samples_dict: pesummary.utils.samples_dict.SamplesDict
        dictionary displaying the posterior samples table.
    """
    def __init__(
        self, *args, drop_non_numeric=True, remove_nan_likelihood_samples=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mcmc_samples = False
        self.drop_non_numeric = drop_non_numeric
        self.remove_nan_likelihood_samples = remove_nan_likelihood_samples

    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            _ = self.samples
        self._parameters = [str(i) for i in self._parameters]
        return Parameters(self._parameters)

    @property
    def samples(self):
        if hasattr(self, "_samples"):
            return self._samples
        self.grab_samples_from_file()
        if not self._is_array:
            self._convert_samples_to_array()
        if self.drop_non_numeric:
            self.remove_non_numeric_values_from_samples()
        if self.remove_nan_likelihood_samples:
            self.remove_nan_likelihood_values_from_samples()
        return self._samples

    @property
    def _is_array(self):
        if not np.issubdtype(type(self._samples[0]), np.number):
            return True
        return False

    def _convert_samples_to_array(self):
        self._samples_dict = {
            key: [item] for key, item in self.samples_dict.items()
        }

    def remove_non_numeric_values_from_samples(self):
        numeric_cols = []
        for i in range(self._samples.shape[1]):
            try:
                original_dtype = self._samples[:, i].dtype
                # this will fail if non-numeric dtype
                self._samples[:, i].astype(np.complex128)
                numeric_cols.append(i)
                self._samples[:, i].astype(original_dtype)
            except (ValueError, TypeError):
                from pesummary.utils.utils import logger
                logger.warning(
                    f"Removing '{self._parameters[i]}' from the posterior "
                    f"table as it contains non-numeric values. To prevent "
                    f"this, pass 'drop_non_numeric=False' when loading the "
                    f"file."
                )
                continue
        if not len(numeric_cols):
            self._parameters = []
            self._samples = []
        else:
            self._parameters = self._parameters[numeric_cols]
            self._samples = self._samples[:, numeric_cols]

    def remove_nan_likelihood_values_from_samples(self):
        if "log_likelihood" not in self._parameters:
            return
        ind = self.parameters.index("log_likelihood")
        likelihoods = self._samples[:,ind]
        inds = np.isnan(likelihoods)
        if not sum(inds):
            return
        msg = (
            f"Posterior table contains {sum(inds)} samples with 'nan' log "
            f"likelihood. "
        )
        msg += "Removing samples from posterior table."
        logger.warn(msg)
        self._samples = self._samples[~inds,:]

    @property
    def samples_dict(self):
        # if samples_dict has already been initialized, use this
        if hasattr(self, "_samples_dict"):
            return self._samples_dict
        return self._generate_samples_dict()

    def _generate_samples_dict(self):
        from pesummary.utils.samples_dict import (
            SamplesDict, MCMCSamplesDict
        )
        if self.mcmc_samples:
            return MCMCSamplesDict(
                self.parameters, [np.array(i).T for i in self.samples]
            )
        return SamplesDict(self.parameters, np.array(self.samples).T)

    def __repr__(self):
        return self.summary()

    def _parameter_summary(self, parameters, parameters_to_show=4):
        """Return a summary of the parameter stored

        Parameters
        ----------
        parameters: list
            list of parameters to create a summary for
        parameters_to_show: int, optional
            number of parameters to show. Default 4.
        """
        params = parameters
        if len(parameters) > parameters_to_show:
            params = parameters[:2] + ["..."] + parameters[-2:]
        return ", ".join(params)

    def summary(
        self, parameters_to_show=4, show_parameters=True, show_nsamples=True
    ):
        """Return a summary of the contents of the file

        Parameters
        ----------
        parameters_to_show: int, optional
            number of parameters to show. Default 4
        show_parameters: Bool, optional
            if True print a list of the parameters stored
        show_nsamples: Bool, optional
            if True print how many samples are stored in the file
        """
        string = ""
        if self.filename is not None:
            string += "file: {}\n".format(self.filename)
        string += "cls: {}.{}\n".format(
            self.__class__.__module__, self.__class__.__name__
        )
        if show_nsamples:
            string += "nsamples: {}\n".format(len(self.samples))
        if show_parameters:
            string += "parameters: {}".format(
                self._parameter_summary(
                    self.parameters, parameters_to_show=parameters_to_show
                )
            )
        return string
