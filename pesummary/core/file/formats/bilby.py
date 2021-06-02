# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from pesummary.core.file.formats.base_read import SingleAnalysisRead
from pesummary.core.plots.latex_labels import latex_labels
from pesummary import conf
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _load_bilby(path):
    """Wrapper for `bilby.core.result.read_in_result`

    Parameters
    ----------
    path: str
        path to the bilby result file you wish to load
    """
    from bilby.core.result import read_in_result
    return read_in_result(filename=path)


def read_bilby(
    path, disable_prior=False, complex_params=[], latex_dict=latex_labels,
    nsamples_for_prior=None, _bilby_class=None, **kwargs
):
    """Grab the parameters and samples in a bilby file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `bilby` result file.
        Default False
    complex_params: list, optional
        list of parameters stored in the bilby result file which are complex
        and you wish to store the `amplitude` and `angle` as seperate
        posterior distributions
    latex_dict: dict, optional
        list of latex labels for each parameter
    nsamples_for_prior: int, optional
        number of samples to draw from the analytic priors
    """
    if _bilby_class is None:
        _bilby_class = Bilby
    bilby_object = _load_bilby(path)
    posterior = bilby_object.posterior
    _original_keys = posterior.keys()
    for key in _original_keys:
        for param in complex_params:
            if param in key and any(np.iscomplex(posterior[key])):
                posterior[key + "_amp"] = abs(posterior[key])
                posterior[key + "_angle"] = np.angle(posterior[key])
                posterior[key] = np.real(posterior[key])
            elif param in key:
                posterior[key] = np.real(posterior[key])
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
            if key not in latex_dict:
                label = bilby_object.get_latex_labels_from_parameter_keys(
                    [key])[0]
                latex_dict[key] = label
    try:
        extra_kwargs = _bilby_class.grab_extra_kwargs(bilby_object)
    except Exception:
        extra_kwargs = {"sampler": {}, "meta_data": {}}
    extra_kwargs["sampler"]["nsamples"] = len(samples)
    extra_kwargs["sampler"]["pe_algorithm"] = "bilby"
    try:
        version = bilby_object.version
        if isinstance(version, (list, np.ndarray)):
            version = version[0]
    except Exception as e:
        version = None

    data = {
        "parameters": parameters,
        "samples": samples.tolist(),
        "injection": injection,
        "version": version,
        "kwargs": extra_kwargs
    }
    if bilby_object.meta_data is not None:
        if "command_line_args" in bilby_object.meta_data.keys():
            data["config"] = {
                "config": bilby_object.meta_data["command_line_args"]
            }
    if not disable_prior:
        logger.debug("Drawing prior samples from bilby result file")
        if nsamples_for_prior is None:
            nsamples_for_prior = len(samples)
        prior_samples = Bilby.grab_priors(
            bilby_object, nsamples=nsamples_for_prior
        )
        data["prior"] = {"samples": prior_samples}
        if len(prior_samples):
            data["prior"]["analytic"] = prior_samples.analytic
    else:
        try:
            _prior = bilby_object.priors
            data["prior"] = {
                "samples": {},
                "analytic": {key: str(item) for key, item in _prior.items()}
            }
        except (AttributeError, KeyError):
            pass
    return data


def to_bilby(
    parameters, samples, label=None, analytic_priors=None, cls=None,
    meta_data=None, **kwargs
):
    """Convert a set of samples to a bilby object

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    """
    from bilby.core.result import Result
    from bilby.core.prior import Prior, PriorDict
    from pandas import DataFrame

    if cls is None:
        cls = Result
    if analytic_priors is not None:
        priors = PriorDict._get_from_json_dict(analytic_priors)
        search_parameters = priors.keys()
    else:
        priors = {param: Prior() for param in parameters}
        search_parameters = parameters
    posterior_data_frame = DataFrame(samples, columns=parameters)
    bilby_object = cls(
        search_parameter_keys=search_parameters, samples=samples, priors=priors,
        posterior=posterior_data_frame, label="pesummary_%s" % label,
    )
    return bilby_object


def _write_bilby(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    extension="json", save=True, analytic_priors=None, cls=None,
    meta_data=None, **kwargs
):
    """Write a set of samples to a bilby file

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    extension: str, optional
        file extension for the bilby result file. Default json.
    save: Bool, optional
        if True, save the bilby object to file
    """
    bilby_object = to_bilby(
        parameters, samples, label=None, analytic_priors=None, cls=None,
        meta_data=None, **kwargs
    )
    if save:
        _filename = os.path.join(outdir, filename)
        bilby_object.save_to_file(filename=_filename, extension=extension)
    else:
        return bilby_object


def write_bilby(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    extension="json", save=True, analytic_priors=None, cls=None,
    meta_data=None, labels=None, **kwargs
):
    """Write a set of samples to a bilby file

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    outdir: str, optional
        directory to write the dat file
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    extension: str, optional
        file extension for the bilby result file. Default json.
    save: Bool, optional
        if True, save the bilby object to file
    """
    from pesummary.io.write import _multi_analysis_write

    func = _write_bilby
    if not save:
        func = to_bilby
    return _multi_analysis_write(
        func, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, extension=extension,
        save=save, analytic_priors=analytic_priors, cls=cls,
        meta_data=meta_data, file_format="bilby", labels=labels,
        _return=True, **kwargs
    )


def config_from_file(path):
    """Extract the configuration file stored within a bilby result file

    Parameters
    ----------
    path: str
        path to the bilby result file you wish to load
    """
    bilby_object = _load_bilby(path)
    return config_from_object(bilby_object)


def config_from_object(bilby_object):
    """Extract the configuration file stored within a `bilby.core.result.Result`
    object (or alike)

    Parameters
    ----------
    bilby_object: bilby.core.result.Result (or alike)
        a bilby.core.result.Result object (or alike) you wish to extract the
        configuration file from
    """
    config = {}
    if bilby_object.meta_data is not None:
        if "command_line_args" in bilby_object.meta_data.keys():
            config = {
                "config": bilby_object.meta_data["command_line_args"]
            }
    return config


def prior_samples_from_file(path, cls="PriorDict", nsamples=5000, **kwargs):
    """Return a dict of prior samples from a `bilby` prior file

    Parameters
    ----------
    path: str
        path to a `bilby` prior file
    cls: str, optional
        class you wish to read in the prior file
    nsamples: int, optional
        number of samples to draw from a prior file. Default 5000
    """
    from bilby.core import prior

    if isinstance(cls, str):
        cls = getattr(prior, cls)
    _prior = cls(filename=path)
    samples = _prior.sample(size=nsamples)
    return _bilby_prior_dict_to_pesummary_samples_dict(samples, prior=_prior)


def prior_samples_from_bilby_object(bilby_object, nsamples=5000, **kwargs):
    """Return a dict of prior samples from a `bilby.core.result.Result`
    object

    Parameters
    ----------
    bilby_object: bilby.core.result.Result
        a bilby.core.result.Result object you wish to draw prior samples from
    nsamples: int, optional
        number of samples to draw from a prior file. Default 5000
    """
    samples = bilby_object.priors.sample(size=nsamples)
    return _bilby_prior_dict_to_pesummary_samples_dict(
        samples, prior=bilby_object.priors
    )


def _bilby_prior_dict_to_pesummary_samples_dict(samples, prior=None):
    """Return a pesummary.utils.samples_dict.SamplesDict object from a bilby
    priors dict
    """
    from pesummary.utils.samples_dict import SamplesDict

    _samples = SamplesDict(samples)
    if prior is not None:
        analytic = {key: str(item) for key, item in prior.items()}
        setattr(_samples, "analytic", analytic)
    return _samples


class Bilby(SingleAnalysisRead):
    """PESummary wrapper of `bilby` (https://git.ligo.org/lscsoft/bilby). The
    path_to_results_file argument will be passed directly to
    `bilby.core.result.read_in_result`. All functions therefore use `bilby`
    methods and requires `bilby` to be installed.

    Parameters
    ----------
    path_to_results_file: str
        path to the results file that you wish to read in with `bilby`.
    disable_prior: Bool, optional
        if True, do not collect prior samples from the `bilby` result file.
        Default False

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
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples

    Methods
    -------
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(Bilby, self).__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_bilby_file, **kwargs)

    @staticmethod
    def grab_priors(bilby_object, nsamples=5000):
        """Draw samples from the prior functions stored in the bilby file
        """
        try:
            return prior_samples_from_bilby_object(
                bilby_object, nsamples=nsamples
            )
        except Exception as e:
            logger.info("Failed to draw prior samples because {}".format(e))
            return {}

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
            "meta_data": {}, "other": f.meta_data}
        return kwargs

    @staticmethod
    def _grab_data_from_bilby_file(path, **kwargs):
        """Load the results file using the `bilby` library
        """
        return read_bilby(path, **kwargs)

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass
