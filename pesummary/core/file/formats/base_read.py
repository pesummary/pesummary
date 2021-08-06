# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
import h5py
from pesummary.utils.parameters import MultiAnalysisParameters, Parameters
from pesummary.utils.samples_dict import (
    MultiAnalysisSamplesDict, SamplesDict, MCMCSamplesDict, Array
)
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _downsample(samples, number, extra_kwargs=None):
    """Downsample a posterior table

    Parameters
    ----------
    samples: 2d list
        list of posterior samples where the columns correspond to a given
        parameter
    number: int
        number of posterior samples you wish to downsample to
    extra_kwargs: dict, optional
        dictionary of kwargs to modify
    """
    from pesummary.utils.utils import resample_posterior_distribution
    import copy

    _samples = np.array(samples).T
    if number > len(_samples[0]):
        raise ValueError(
            "Failed to downsample the posterior samples to {} because "
            "there are only {} samples stored in the file.".format(
                number, len(_samples[0])
            )
        )
    _samples = np.array(resample_posterior_distribution(_samples, number))
    if extra_kwargs is None:
        return _samples.T.tolist()
    _extra_kwargs = copy.deepcopy(extra_kwargs)
    _extra_kwargs["sampler"]["nsamples"] = number
    return _samples.T.tolist(), _extra_kwargs


class Read(object):
    """Base class to read in a results file

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

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
    pe_algorithm: str
        name of the algorithm used to generate the posterior samples

    Methods
    -------
    downsample:
        downsample the posterior samples stored in the result file
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    """
    def __init__(self, path_to_results_file, **kwargs):
        self.path_to_results_file = path_to_results_file
        self.mcmc_samples = False
        self.extension = self.extension_from_path(self.path_to_results_file)
        self.converted_parameters = []

    @classmethod
    def load_file(cls, path, **kwargs):
        """Initialize the class with a file

        Parameters
        ----------
        path: str
            path to the result file you wish to load
        **kwargs: dict, optional
            all kwargs passed to the class
        """
        if not os.path.isfile(path):
            raise FileNotFoundError("%s does not exist" % (path))
        return cls(path, **kwargs)

    @staticmethod
    def load_from_function(function, path_to_file, **kwargs):
        """Load a file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your file
        path_to_file: str
            path to the file that you wish to load
        kwargs: dict
            all kwargs are passed to the function
        """
        return function(path_to_file, **kwargs)

    @staticmethod
    def check_for_weights(parameters, samples):
        """Check to see if the samples are weighted

        Parameters
        ----------
        parameters: list
            list of parameters stored in the result file
        samples: np.ndarray
            array of samples for each parameter
        """
        likely_names = ["weights", "weight"]
        if any(i in parameters for i in likely_names):
            ind = (
                parameters.index("weights") if "weights" in parameters else
                parameters.index("weight")
            )
            return Array(np.array(samples).T[ind])
        return None

    @property
    def pe_algorithm(self):
        try:
            return self.extra_kwargs["sampler"]["pe_algorithm"]
        except KeyError:
            return None

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
        if self.path_to_results_file is not None:
            string += "file: {}\n".format(self.path_to_results_file)
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

    attrs = {
        "input_version": "version", "extra_kwargs": "kwargs",
        "priors": "prior", "analytic": "analytic", "labels": "labels",
        "config": "config", "weights": "weights", "history": "history",
        "description": "description"
    }

    def _load(self, function, **kwargs):
        """Extract the data from a file using a given function

        Parameters
        ----------
        function: func
            function you wish to use to extract the data
        **kwargs: dict, optional
            optional kwargs to pass to the load function
        """
        return self.load_from_function(
            function, self.path_to_results_file, **kwargs
        )

    def load(self, function, _data=None, **kwargs):
        """Load a results file according to a given function

        Parameters
        ----------
        function: func
            callable function that will load in your results file
        """
        self.data = _data
        if _data is None:
            self.data = self._load(function, **kwargs)
        if isinstance(self.data["parameters"][0], list):
            _cls = MultiAnalysisParameters
        else:
            _cls = Parameters
        self.parameters = _cls(self.data["parameters"])
        self.converted_parameters = []
        self.samples = self.data["samples"]
        if "mcmc_samples" in self.data.keys():
            self.mcmc_samples = self.data["mcmc_samples"]
        if "injection" in self.data.keys():
            if isinstance(self.data["injection"], dict):
                self.injection_parameters = {
                    key.decode("utf-8") if isinstance(key, bytes) else key: val
                    for key, val in self.data["injection"].items()
                }
            elif isinstance(self.data["injection"], list):
                self.injection_parameters = [
                    {
                        key.decode("utf-8") if isinstance(key, bytes) else
                        key: val for key, val in i.items()
                    } for i in self.data["injection"]
                ]
            else:
                self.injection_parameters = self.data["injection"]
        for new_attr, existing_attr in self.attrs.items():
            if existing_attr in self.data.keys():
                setattr(self, new_attr, self.data[existing_attr])
            else:
                setattr(self, new_attr, None)
        if self.input_version is None:
            self.input_version = self._default_version
        if self.extra_kwargs is None:
            self.extra_kwargs = self._default_kwargs
        if self.description is None:
            self.description = self._default_description
        if self.weights is None:
            self.weights = self.check_for_weights(
                self.data["parameters"], self.data["samples"]
            )

    @staticmethod
    def extension_from_path(path):
        """Return the extension of the file from the file path

        Parameters
        ----------
        path: str
            path to the results file
        """
        extension = path.split(".")[-1]
        return extension

    @staticmethod
    def guess_path_to_samples(path):
        """Guess the path to the posterior samples stored in an hdf5 object

        Parameters
        ----------
        path: str
            path to the results file
        """
        def _find_name(name, item):
            c1 = "posterior_samples" in name or "posterior" in name
            c2 = isinstance(item, (h5py._hl.dataset.Dataset, np.ndarray))
            try:
                c3 = isinstance(item, h5py._hl.group.Group) and isinstance(
                    item[0], (float, int, np.number)
                )
            except (TypeError, AttributeError):
                c3 = False
            c4 = (
                isinstance(item, h5py._hl.group.Group) and "parameter_names" in
                item.keys() and "samples" in item.keys()
            )
            if c1 and c3:
                paths.append(name)
            elif c1 and c4:
                return paths.append(name)
            elif c1 and c2:
                return paths.append(name)

        f = h5py.File(path, 'r')
        paths = []
        f.visititems(_find_name)
        f.close()
        if len(paths) == 1:
            return paths[0]
        elif len(paths) > 1:
            raise ValueError(
                "Found multiple posterior sample tables in '{}': {}. Not sure "
                "which to load.".format(
                    path, ", ".join(paths)
                )
            )
        else:
            raise ValueError(
                "Unable to find a posterior samples table in '{}'".format(path)
            )

    def generate_all_posterior_samples(self, **kwargs):
        """Empty function
        """
        pass

    def add_fixed_parameters_from_config_file(self, config_file):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        pass

    def add_injection_parameters_from_file(self, injection_file, **kwargs):
        """
        """
        self.injection_parameters = self._add_injection_parameters_from_file(
            injection_file, self._grab_injection_parameters_from_file,
            **kwargs
        )

    def _grab_injection_parameters_from_file(self, path, **kwargs):
        """
        """
        from pesummary.core.file.injection import Injection

        data = Injection.read(path, **kwargs).samples_dict
        for i in self.parameters:
            if i not in data.keys():
                data[i] = float("nan")
        return data

    def _add_injection_parameters_from_file(self, injection_file, function, **kwargs):
        """Add the injection parameters from file

        Parameters
        ----------
        injection_file: str
            path to injection file
        function: func
            funcion you wish to use to extract the information from the
            injection file
        """
        return function(injection_file, **kwargs)

    def write(
        self, package="core", file_format="dat", extra_kwargs=None,
        file_versions=None, **kwargs
    ):
        """Save the data to file

        Parameters
        ----------
        package: str, optional
            package you wish to use when writing the data
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        from pesummary.io import write

        if file_format == "pesummary" and np.array(self.parameters).ndim > 1:
            args = [self.samples_dict]
        else:
            args = [self.parameters, self.samples]
        if extra_kwargs is None:
            extra_kwargs = self.extra_kwargs
        if file_versions is None:
            file_versions = self.input_version
        if file_format == "ini":
            kwargs["file_format"] = "ini"
            return write(getattr(self, "config", None), **kwargs)
        else:
            return write(
                *args, package=package, file_versions=file_versions,
                file_kwargs=extra_kwargs, file_format=file_format, **kwargs
            )

    def downsample(self, number):
        """Downsample the posterior samples stored in the result file
        """
        self.samples, self.extra_kwargs = _downsample(
            self.samples, number, extra_kwargs=self.extra_kwargs
        )

    @staticmethod
    def latex_table(samples, parameter_dict=None, labels=None):
        """Return a latex table displaying the passed data.

        Parameters
        ----------
        samples_dict: list
            list of pesummary.utils.utils.SamplesDict objects
        parameter_dict: dict, optional
            dictionary of parameters that you wish to include in the latex
            table. The keys are the name of the parameters and the items are
            the descriptive text. If None, all parameters are included
        """
        table = (
            "\\begin{table}[hptb]\n\\begin{ruledtabular}\n\\begin{tabular}"
            "{l %s}\n" % ("c " * len(samples))
        )
        if labels:
            table += (
                " & " + " & ".join(labels)
            )
            table += "\\\ \n\\hline \\\ \n"
        data = {i: i for i in samples[0].keys()}
        if parameter_dict is not None:
            import copy

            data = copy.deepcopy(parameter_dict)
            for param in parameter_dict.keys():
                if not all(param in samples_dict.keys() for samples_dict in samples):
                    logger.warning(
                        "{} not in list of parameters. Not adding to "
                        "table".format(param)
                    )
                    data.pop(param)

        for param, desc in data.items():
            table += "{}".format(desc)
            for samples_dict in samples:
                median = samples_dict[param].average(type="median")
                confidence = samples_dict[param].confidence_interval()
                table += (
                    " & $%s^{+%s}_{-%s}$" % (
                        np.round(median, 2),
                        np.round(confidence[1] - median, 2),
                        np.round(median - confidence[0], 2)
                    )
                )
            table += "\\\ \n"
        table += (
            "\\end{tabular}\n\\end{ruledtabular}\n\\caption{}\n\\end{table}"
        )
        return table

    @staticmethod
    def latex_macros(
        samples, parameter_dict=None, labels=None, rounding="smart"
    ):
        """Return a latex table displaying the passed data.

        Parameters
        ----------
        samples_dict: list
            list of pesummary.utils.utils.SamplesDict objects
        parameter_dict: dict, optional
            dictionary of parameters that you wish to generate macros for. The
            keys are the name of the parameters and the items are the latex
            macros name you wish to use. If None, all parameters are included.
        rounding: int, optional
            decimal place for rounding. Default uses the
            `pesummary.utils.utils.smart_round` function to round according to
            the uncertainty
        """
        macros = ""
        data = {i: i for i in samples[0].keys()}
        if parameter_dict is not None:
            import copy

            data = copy.deepcopy(parameter_dict)
            for param in parameter_dict.keys():
                if not all(param in samples_dict.keys() for samples_dict in samples):
                    logger.warning(
                        "{} not in list of parameters. Not generating "
                        "macro".format(param)
                    )
                    data.pop(param)
        for param, desc in data.items():
            for num, samples_dict in enumerate(samples):
                if labels:
                    description = "{}{}".format(desc, labels[num])
                else:
                    description = desc

                median = samples_dict[param].average(type="median")
                confidence = samples_dict[param].confidence_interval()
                if rounding == "smart":
                    from pesummary.utils.utils import smart_round

                    median, upper, low = smart_round([
                        median, confidence[1] - median, median - confidence[0]
                    ])
                else:
                    median = np.round(median, rounding)
                    low = np.round(median - confidence[0], rounding)
                    upper = np.round(confidence[1] - median, rounding)
                macros += (
                    "\\def\\%s{$%s_{-%s}^{+%s}$}\n" % (
                        description, median, low, upper
                    )
                )
                macros += (
                    "\\def\\%smedian{$%s$}\n" % (description, median)
                )
                macros += (
                    "\\def\\%supper{$%s$}\n" % (
                        description, np.round(median + upper, 9)
                    )
                )
                macros += (
                    "\\def\\%slower{$%s$}\n" % (
                        description, np.round(median - low, 9)
                    )
                )
        return macros


class SingleAnalysisRead(Read):
    """Base class to read in a results file which contains a single analyses

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: list
        list of parameters stored in the file
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file
    input_version: str
        version of the result file passed
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file

    Methods
    -------
    downsample:
        downsample the posterior samples stored in the result file
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    reweight_samples:
        reweight the posterior and/or samples according to a new prior
    """
    def __init__(self, *args, **kwargs):
        super(SingleAnalysisRead, self).__init__(*args, **kwargs)

    @property
    def samples_dict(self):
        if self.mcmc_samples:
            return MCMCSamplesDict(
                self.parameters, [np.array(i).T for i in self.samples]
            )
        return SamplesDict(self.parameters, np.array(self.samples).T)

    @property
    def _default_version(self):
        return "No version information found"

    @property
    def _default_kwargs(self):
        _kwargs = {"sampler": {}, "meta_data": {}}
        _kwargs["sampler"]["nsamples"] = len(self.data["samples"])
        return _kwargs

    @property
    def _default_description(self):
        return "No description found"

    def _add_fixed_parameters_from_config_file(self, config_file, function):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        function: func
            function you wish to use to extract the information from the
            configuration file
        """
        self.data[0], self.data[1] = function(self.parameters, self.samples, config_file)

    def _add_marginalized_parameters_from_config_file(self, config_file, function):
        """Search the configuration file and add marginalized parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        function: func
            function you wish to use to extract the information from the
            configuration file
        """
        self.data[0], self.data[1] = function(self.parameters, self.samples, config_file)

    def to_latex_table(self, parameter_dict=None, save_to_file=None):
        """Make a latex table displaying the data in the result file.

        Parameters
        ----------
        parameter_dict: dict, optional
            dictionary of parameters that you wish to include in the latex
            table. The keys are the name of the parameters and the items are
            the descriptive text. If None, all parameters are included
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )

        table = self.latex_table([self.samples_dict], parameter_dict)
        if save_to_file is None:
            print(table)
        elif os.path.isfile("{}".format(save_to_file)):
            logger.warning(
                "File {} already exists. Printing to stdout".format(save_to_file)
            )
            print(table)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([table])

    def generate_latex_macros(
        self, parameter_dict=None, save_to_file=None, rounding="smart"
    ):
        """Generate a list of latex macros for each parameter in the result
        file

        Parameters
        ----------
        labels: list, optional
            list of labels that you want to include in the table
        parameter_dict: dict, optional
            dictionary of parameters that you wish to generate macros for. The
            keys are the name of the parameters and the items are the latex
            macros name you wish to use. If None, all parameters are included.
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        rounding: int, optional
            number of decimal points to round the latex macros
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )

        macros = self.latex_macros(
            [self.samples_dict], parameter_dict, rounding=rounding
        )
        if save_to_file is None:
            print(macros)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([macros])

    def to_dat(self, **kwargs):
        """Save the PESummary results file object to a dat file

        Parameters
        ----------
        kwargs: dict
            all kwargs passed to the pesummary.core.file.formats.dat.write_dat
            function
        """
        return self.write(file_format="dat", **kwargs)

    def reweight_samples(self, function, **kwargs):
        """Reweight the posterior and/or prior samples according to a new prior
        """
        if self.mcmc_samples:
            return ValueError("Cannot currently reweight MCMC chains")
        _samples = self.samples_dict
        new_samples = _samples.reweight(function, **kwargs)
        self.parameters = Parameters(new_samples.parameters)
        self.samples = np.array(new_samples.samples).T
        self.extra_kwargs["sampler"].update(
            {
                "nsamples": new_samples.number_of_samples,
                "nsamples_before_reweighting": _samples.number_of_samples
            }
        )
        self.extra_kwargs["meta_data"]["reweighting"] = function
        if not hasattr(self, "priors"):
            return
        if (self.priors is None) or ("samples" not in self.priors.keys()):
            return
        prior_samples = self.priors["samples"]
        if not len(prior_samples):
            return
        new_prior_samples = prior_samples.reweight(function, **kwargs)
        self.priors["samples"] = new_prior_samples


class MultiAnalysisRead(Read):
    """Base class to read in a results file which contains multiple analyses

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load

    Attributes
    ----------
    parameters: 2d list
        list of parameters for each analysis
    samples: 3d list
        list of samples stored in the result file for each analysis
    samples_dict: dict
        dictionary of samples stored in the result file keyed by analysis label
    input_version: str
        version of the result files passed
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file

    Methods
    -------
    downsample:
        downsample the posterior samples stored in the result file
    to_dat:
        save the posterior samples to a .dat file
    to_latex_table:
        convert the posterior samples to a latex table
    generate_latex_macros:
        generate a set of latex macros for the stored posterior samples
    reweight_samples:
        reweight the posterior and/or samples according to a new prior
    """
    def __init__(self, *args, **kwargs):
        super(MultiAnalysisRead, self).__init__(*args, **kwargs)

    @property
    def samples_dict(self):
        if self.mcmc_samples:
            outdict = MCMCSamplesDict(
                self.parameters[0], [np.array(i).T for i in self.samples[0]]
            )
        else:
            if all("log_likelihood" in i for i in self.parameters):
                likelihood_inds = [self.parameters[idx].index("log_likelihood")
                                   for idx in range(len(self.labels))]
                likelihoods = [[i[likelihood_inds[idx]] for i in self.samples[idx]]
                               for idx, label in enumerate(self.labels)]
            else:
                likelihoods = [None] * len(self.labels)
            outdict = MultiAnalysisSamplesDict({
                label:
                    SamplesDict(
                        self.parameters[idx], np.array(self.samples[idx]).T
                    ) for idx, label in enumerate(self.labels)
            })
        return outdict

    @property
    def _default_version(self):
        return ["No version information found"] * len(self.parameters)

    @property
    def _default_kwargs(self):
        _kwargs = [{"sampler": {}, "meta_data": {}}] * len(self.parameters)
        for num, ss in enumerate(self.data["samples"]):
            _kwargs[num]["sampler"]["nsamples"] = len(ss)
        return _kwargs

    @property
    def _default_description(self):
        return {label: "No description found" for label in self.labels}

    def write(self, package="core", file_format="dat", **kwargs):
        """Save the data to file

        Parameters
        ----------
        package: str, optional
            package you wish to use when writing the data
        kwargs: dict, optional
            all additional kwargs are passed to the pesummary.io.write function
        """
        return super(MultiAnalysisRead, self).write(
            package=package, file_format=file_format,
            extra_kwargs=self.kwargs_dict, file_versions=self.version_dict,
            **kwargs
        )

    @property
    def kwargs_dict(self):
        return {
            label: kwarg for label, kwarg in zip(self.labels, self.extra_kwargs)
        }

    @property
    def version_dict(self):
        return {
            label: version for label, version in zip(self.labels, self.input_version)
        }

    def summary(self, *args, parameters_to_show=4, **kwargs):
        """Return a summary of the contents of the file

        Parameters
        ----------
        parameters_to_show: int, optional
            number of parameters to show. Default 4
        """
        string = super(MultiAnalysisRead, self).summary(
            show_parameters=False, show_nsamples=False
        )
        string += "analyses: {}\n\n".format(", ".join(self.labels))
        for num, label in enumerate(self.labels):
            string += "{}\n".format(label)
            string += "-" * len(label) + "\n"
            string += "description: {}\n".format(self.description[label])
            string += "nsamples: {}\n".format(len(self.samples[num]))
            string += "parameters: {}\n\n".format(
                self._parameter_summary(
                    self.parameters[num], parameters_to_show=parameters_to_show
                )
            )
        return string[:-2]

    def downsample(self, number):
        """Downsample the posterior samples stored in the result file
        """
        for num, ss in enumerate(self.samples):
            self.samples[num], self.extra_kwargs[num] = _downsample(
                ss, number, extra_kwargs=self.extra_kwargs[num]
            )

    def to_latex_table(self, labels="all", parameter_dict=None, save_to_file=None):
        """Make a latex table displaying the data in the result file.

        Parameters
        ----------
        labels: list, optional
            list of labels that you want to include in the table
        parameter_dict: dict, optional
            dictionary of parameters that you wish to include in the latex
            table. The keys are the name of the parameters and the items are
            the descriptive text. If None, all parameters are included
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )
        if labels != "all" and isinstance(labels, str) and labels not in self.labels:
            raise ValueError("The label %s does not exist." % (labels))
        elif labels == "all":
            labels = list(self.labels)
        elif isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels, list):
            for ll in labels:
                if ll not in list(self.labels):
                    raise ValueError("The label %s does not exist." % (ll))

        table = self.latex_table(
            [self.samples_dict[label] for label in labels], parameter_dict,
            labels=labels
        )
        if save_to_file is None:
            print(table)
        elif os.path.isfile("{}".format(save_to_file)):
            logger.warning(
                "File {} already exists. Printing to stdout".format(save_to_file)
            )
            print(table)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([table])

    def generate_latex_macros(
        self, labels="all", parameter_dict=None, save_to_file=None,
        rounding=2
    ):
        """Generate a list of latex macros for each parameter in the result
        file

        Parameters
        ----------
        labels: list, optional
            list of labels that you want to include in the table
        parameter_dict: dict, optional
            dictionary of parameters that you wish to generate macros for. The
            keys are the name of the parameters and the items are the latex
            macros name you wish to use. If None, all parameters are included.
        save_to_file: str, optional
            name of the file you wish to save the latex table to. If None, print
            to stdout
        rounding: int, optional
            number of decimal points to round the latex macros
        """
        import os

        if save_to_file is not None and os.path.isfile("{}".format(save_to_file)):
            raise FileExistsError(
                "The file {} already exists.".format(save_to_file)
            )
        if labels != "all" and isinstance(labels, str) and labels not in self.labels:
            raise ValueError("The label %s does not exist." % (labels))
        elif labels == "all":
            labels = list(self.labels)
        elif isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels, list):
            for ll in labels:
                if ll not in list(self.labels):
                    raise ValueError("The label %s does not exist." % (ll))

        macros = self.latex_macros(
            [self.samples_dict[i] for i in labels], parameter_dict,
            labels=labels, rounding=rounding
        )
        if save_to_file is None:
            print(macros)
        else:
            with open(save_to_file, "w") as f:
                f.writelines([macros])

    def reweight_samples(self, function, labels=None, **kwargs):
        """Reweight the posterior and/or prior samples according to a new prior

        Parameters
        ----------
        labels: list, optional
            list of analyses you wish to reweight. Default reweight all
            analyses
        """
        _samples_dict = self.samples_dict
        for idx, label in enumerate(self.labels):
            if labels is not None and label not in labels:
                continue
            new_samples = _samples_dict[label].reweight(function, **kwargs)
            self.parameters[idx] = Parameters(new_samples.parameters)
            self.samples[idx] = np.array(new_samples.samples).T
            self.extra_kwargs[idx]["sampler"].update(
                {
                    "nsamples": new_samples.number_of_samples,
                    "nsamples_before_reweighting": (
                        _samples_dict[label].number_of_samples
                    )
                }
            )
            self.extra_kwargs[idx]["meta_data"]["reweighting"] = function
            if not hasattr(self, "priors"):
                continue
            if "samples" not in self.priors.keys():
                continue
            prior_samples = self.priors["samples"][label]
            if not len(prior_samples):
                continue
            new_prior_samples = prior_samples.reweight(function, **kwargs)
            self.priors["samples"][label] = new_prior_samples
