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

from pesummary.gw.file.formats.base_read import GWRead, GWSingleAnalysisRead
from pesummary.gw import conversions as con
from pesummary.utils.utils import logger
from pesummary.utils.decorators import open_config
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
SAMPLER_KWARGS = {
    "log_bayes_factor": conf.log_bayes_factor,
    "bayes_factor": conf.bayes_factor,
    "log_evidence": conf.log_evidence,
    "evidence": conf.evidence,
    "log_prior_volume": conf.log_prior_volume,
    "sampleRate": "sample_rate",
    "segmentLength": "segment_length"
}

META_DATA = {
    "flow": "f_low",
    "f_low": "f_low",
    "fref": "f_ref",
    "f_ref": "f_ref",
    "LAL_PNORDER": "pn_order",
    "LAL_APPROXIMANT": "approximant",
    "number_of_live_points": "number_of_live_points",
    "segmentLength": "segment_length",
    "segmentStart": "segment_start",
    "sampleRate": "sample_rate",
}


def open_lalinference(path):
    """Grab the parameters and samples in a lalinference file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    f = h5py.File(path, 'r')
    path_to_samples = GWRead.guess_path_to_samples(path)
    lalinference_names = list(f[path_to_samples].dtype.names)
    samples = [list(i) for i in f[path_to_samples]]

    if "logdistance" in lalinference_names:
        lalinference_names.append("luminosity_distance")
        for num, i in enumerate(samples):
            samples[num].append(
                np.exp(i[lalinference_names.index("logdistance")]))
    if "costheta_jn" in lalinference_names:
        lalinference_names.append("theta_jn")
        for num, i in enumerate(samples):
            samples[num].append(
                np.arccos(i[lalinference_names.index("costheta_jn")]))
    extra_kwargs = LALInference.grab_extra_kwargs(path)
    extra_kwargs["sampler"]["nsamples"] = len(samples)
    extra_kwargs["sampler"]["pe_algorithm"] = "lalinference"
    try:
        version = f[path_to_samples].attrs["VERSION"].decode("utf-8")
    except Exception as e:
        version = None
    return {
        "parameters": lalinference_names,
        "samples": samples,
        "injection": None,
        "version": version,
        "kwargs": extra_kwargs
    }


class LALInference(GWSingleAnalysisRead):
    """PESummary wrapper of `lalinference`
    (https://git.ligo.org/lscsoft/lalsuite/lalinference).

    Parameters
    ----------
    path_to_results_file: str
        path to the results file you wish to load in with `LALInference`

    Attributes
    ----------
    parameters: list
        list of parameters stored in the result file
    converted_parameters: list
        list of parameters that have been derived from the sampled distributions
    samples: 2d list
        list of samples stored in the result file
    samples_dict: dict
        dictionary of samples stored in the result file keyed by parameters
    input_version: str
        version of the result file passed.
    extra_kwargs: dict
        dictionary of kwargs that were extracted from the result file
    converted_parameters: list
        list of parameters that have been added
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
    to_lalinference:
        convert the posterior samples to a lalinference result file
    generate_all_posterior_samples:
        generate all posterior distributions that may be derived from
        sampled distributions
    """
    def __init__(self, path_to_results_file, injection_file=None, **kwargs):
        super(LALInference, self).__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_lalinference_file)

    @classmethod
    def load_file(cls, path, injection_file=None, **kwargs):
        if injection_file and not os.path.isfile(injection_file):
            raise IOError("%s does not exist" % (path))
        return super(LALInference, cls).load_file(
            path, injection_file=injection_file, **kwargs
        )

    @staticmethod
    def guess_path_to_sampler(path):
        """Guess the path to the sampler group in a LALInference results file

        Parameters
        ----------
        path: str
            path to the LALInference results file
        """
        def _find_name(name):
            c1 = "lalinference_nest" in name or "lalinference_mcmc" in name
            c2 = "lalinference_nest/" not in name and "lalinference_mcmc/" not in name
            if c1 and c2:
                return name

        f = h5py.File(path, 'r')
        _path = f.visit(_find_name)
        f.close()
        return _path

    @staticmethod
    def _parameters_in_lalinference_file(path):
        """Return the parameter names stored in the LALInference results file

        Parameters
        ----------
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        parameters = list(f[path_to_samples].dtype.names)
        f.close()
        return parameters

    @staticmethod
    def _samples_in_lalinference_file(path):
        """
        """
        f = h5py.File(path, 'r')
        path_to_samples = GWRead.guess_path_to_samples(path)
        samples = [list(i) for i in f[path_to_samples]]
        return samples

    @property
    def calibration_spline_posterior(self):
        if not any("_spcal_amp" in i for i in self.parameters):
            return super(LALInference, self).calibration_spline_posterior
        keys_amp = np.sort(
            [param for param in self.parameters if "_spcal_amp" in param]
        )
        keys_phase = np.sort(
            [param for param in self.parameters if "_spcal_phase" in param]
        )
        IFOs = np.unique(
            [
                param.split("_")[0] for param in self.parameters if
                "_spcal_" in param
            ]
        )
        log_frequencies = {ifo: [] for ifo in IFOs}
        for key, value in self.extra_kwargs["other"].items():
            if "_spcal_logfreq" in key:
                cond = (
                    key.replace("logfreq", "freq") not in
                    self.extra_kwargs["other"].keys()
                )
                if cond:
                    log_frequencies[key.split("_")[0]].append(float(value))
            elif "_spcal_freq" in key:
                log_frequencies[key.split("_")[0]].append(np.log(float(value)))
        amp_params = {ifo: [] for ifo in IFOs}
        phase_params = {ifo: [] for ifo in IFOs}
        zipped = zip(
            [keys_amp, keys_phase], [amp_params, phase_params]
        )
        _samples = self.samples_dict
        for keys, dictionary in zipped:
            for key in keys:
                ifo = key.split("_")[0]
                ind = self.parameters.index(key)
                dictionary[ifo].append(_samples[key])
        return log_frequencies, amp_params, phase_params

    @staticmethod
    def grab_extra_kwargs(path):
        """Grab any additional information stored in the lalinference file
        """
        kwargs = {"sampler": {}, "meta_data": {}, "other": {}}
        path_to_samples = GWRead.guess_path_to_samples(path)
        path_to_sampler = LALInference.guess_path_to_sampler(path)
        f = h5py.File(path, 'r')

        attributes = dict(f[path_to_sampler].attrs.items())
        for kwarg, item in attributes.items():
            if kwarg in list(SAMPLER_KWARGS.keys()) and kwarg == "evidence":
                kwargs["sampler"][conf.log_evidence] = np.round(np.log(item), 2)
            elif kwarg in list(SAMPLER_KWARGS.keys()) and kwarg == "bayes_factor":
                kwargs["sampler"][conf.log_bayes_factor] = np.round(
                    np.log(item), 2
                )
            elif kwarg in list(SAMPLER_KWARGS.keys()):
                kwargs["sampler"][SAMPLER_KWARGS[kwarg]] = np.round(item, 2)
            else:
                kwargs["other"][kwarg] = item

        attributes = dict(f[path_to_samples].attrs.items())
        for kwarg, item in attributes.items():
            if kwarg in list(META_DATA.keys()) and kwarg == "LAL_APPROXIMANT":
                try:
                    from lalsimulation import GetStringFromApproximant

                    kwargs["meta_data"]["approximant"] = \
                        GetStringFromApproximant(
                            int(attributes["LAL_APPROXIMANT"])
                    )
                except Exception:
                    kwargs["meta_data"]["approximant"] = \
                        int(attributes["LAL_APPROXIMANT"])
            elif kwarg in list(META_DATA.keys()):
                kwargs["meta_data"][META_DATA[kwarg]] = item
            else:
                kwargs["other"][kwarg] = item
        f.close()
        return kwargs

    @staticmethod
    def _grab_data_from_lalinference_file(path):
        """
        """
        return open_lalinference(path)

    def add_fixed_parameters_from_config_file(self, config_file):
        """Search the conifiguration file and add fixed parameters to the
        list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        self._add_fixed_parameters_from_config_file(
            config_file, self._add_fixed_parameters)

    def add_marginalized_parameters_from_config_file(self, config_file):
        """Search the configuration file and add the marginalized parameters
        to the list of parameters and samples

        Parameters
        ----------
        config_file: str
            path to the configuration file
        """
        self._add_marginalized_parameters_from_config_file(
            config_file, self._add_marginalized_parameters)

    @staticmethod
    @open_config(index=2)
    def _add_fixed_parameters(parameters, samples, config_file):
        """Open a LALInference configuration file and add the fixed parameters
        to the list of parameters and samples

        Parameters
        ----------
        parameters: list
            list of existing parameters
        samples: list
            list of existing samples
        config_file: str
            path to the configuration file
        """
        from pesummary.gw.file.standard_names import standard_names

        config = config_file
        if not config.error:
            fixed_data = None
            if "engine" in config.sections():
                fixed_data = {
                    key.split("fix-")[1]: item for key, item in
                    config.items("engine") if "fix" in key}
            if fixed_data is not None:
                for i in fixed_data.keys():
                    fixed_parameter = i
                    fixed_value = fixed_data[i]
                    try:
                        param = standard_names[fixed_parameter]
                        if param in parameters:
                            pass
                        else:
                            parameters.append(param)
                            for num in range(len(samples)):
                                samples[num].append(float(fixed_value))
                    except Exception:
                        if fixed_parameter == "logdistance":
                            if "luminosity_distance" not in parameters:
                                parameters.append(standard_names["distance"])
                                for num in range(len(samples)):
                                    samples[num].append(float(fixed_value))
                        if fixed_parameter == "costheta_jn":
                            if "theta_jn" not in parameters:
                                parameters.append(standard_names["theta_jn"])
                                for num in range(len(samples)):
                                    samples[num].append(float(fixed_value))
        return parameters, samples

    @staticmethod
    @open_config(index=2)
    def _add_marginalized_parameters(parameters, samples, config_file):
        """Open a LALInference configuration file and add the marginalized
        parameters to the list of parameters and samples

        Parameters
        ----------
        parameters: list
            list of existing parameters
        samples: list
            list of existing samples
        config_file: str
            path to the configuration file
        """
        from pesummary.gw.file.standard_names import standard_names

        config = config_file
        if not config.error:
            fixed_data = None
            if "engine" in config.sections():
                marg_par = {
                    key.split("marg")[1]: item for key, item in
                    config.items("engine") if "marg" in key}
            for i in marg_par.keys():
                if "time" in i and "geocent_time" not in parameters:
                    if "marginalized_geocent_time" in parameters:
                        ind = parameters.index("marginalized_geocent_time")
                        parameters.remove(parameters[ind])
                        parameters.append("geocent_time")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warning("You have marginalized over time and "
                                       "there are no time samples. Manually "
                                       "setting time to 100000s")
                        parameters.append("geocent_time")
                        for num, j in enumerate(samples):
                            samples[num].append(float(100000))
                if "phi" in i and "phase" not in parameters:
                    if "marginalized_phase" in parameters:
                        ind = parameters.index("marginalized_phase")
                        parameters.remove(parameters[ind])
                        parameters.append("phase")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warning("You have marginalized over phase and "
                                       "there are no phase samples. Manually "
                                       "setting the phase to be 0")
                        parameters.append("phase")
                        for num, j in enumerate(samples):
                            samples[num].append(float(0))
                if "dist" in i and "luminosity_distance" not in parameters:
                    if "marginalized_distance" in parameters:
                        ind = parameters.index("marginalized_distance")
                        parameters.remove(parameters[ind])
                        parameters.append("luminosity_distance")
                        for num, j in enumerate(samples):
                            samples[num].append(float(j[ind]))
                            del j[ind]
                    else:
                        logger.warning("You have marginalized over distance and "
                                       "there are no distance samples. Manually "
                                       "setting distance to 100Mpc")
                        parameters.append("luminosity_distance")
                        for num, j in enumerate(samples):
                            samples[num].append(float(100.0))
            return parameters, samples
        return parameters, samples


def _write_lalinference(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    sampler="lalinference_nest", dat=False, **kwargs
):
    """Write a set of samples in LALInference file format

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    outdir: str
        The directory where you would like to write the lalinference file
    label: str
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str
        The name of the file that you wish to write
    overwrite: Bool
        If True, an existing file of the same name will be overwritten
    sampler: str
        The sampler which you wish to store in the result file. This may either
        be 'lalinference_nest' or 'lalinference_mcmc'
    dat: Bool
        If True, a LALInference dat file is produced
    """
    from pesummary.gw.file.standard_names import lalinference_map
    from pesummary.utils.samples_dict import SamplesDict
    import copy

    _samples = copy.deepcopy(samples)
    _parameters = copy.deepcopy(parameters)
    _samples = SamplesDict(_parameters, np.array(_samples).T.tolist())
    if not filename and not label:
        from time import time

        label = round(time())
    if not filename:
        extension = "dat" if dat else "hdf5"
        filename = "lalinference_{}.{}".format(label, extension)

    if os.path.isfile(os.path.join(outdir, filename)) and not overwrite:
        raise FileExistsError(
            "The file '{}' already exists in the directory {}".format(
                filename, outdir
            )
        )
    reverse_map = {item: key for key, item in lalinference_map.items()}
    no_key = []
    for param in _parameters:
        if param in reverse_map.keys() and reverse_map[param] in _parameters:
            logger.warning(
                "The LALInference name for '{}' is '{}'. '{}' already found "
                "in the posterior table. Keeping both entries".format(
                    param, reverse_map[param], reverse_map[param]
                )
            )
        elif param in reverse_map.keys():
            _samples[reverse_map[param]] = _samples.pop(param)
        elif param not in lalinference_map.keys():
            no_key.append(param)
    if len(no_key):
        logger.info(
            "Unable to find a LALInference name for the parameters: {}. "
            "Keeping the PESummary name.".format(", ".join(no_key))
        )
    lalinference_samples = _samples.to_structured_array()
    if dat:
        np.savetxt(
            os.path.join(outdir, filename), lalinference_samples,
            delimiter="\t", comments="",
            header="\t".join(lalinference_samples.dtype.names)
        )
    else:
        with h5py.File(os.path.join(outdir, filename), "w") as f:
            lalinference = f.create_group("lalinference")
            sampler = lalinference.create_group(sampler)
            samples = sampler.create_dataset(
                "posterior_samples", data=lalinference_samples
            )


def write_lalinference(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    sampler="lalinference_nest", dat=False, **kwargs
):
    """Write a set of samples in LALInference file format

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: 2d list
        list of samples. Columns correspond to a given parameter
    outdir: str
        The directory where you would like to write the lalinference file
    label: str
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str
        The name of the file that you wish to write
    overwrite: Bool
        If True, an existing file of the same name will be overwritten
    sampler: str
        The sampler which you wish to store in the result file. This may either
        be 'lalinference_nest' or 'lalinference_mcmc'
    dat: Bool
        If True, a LALInference dat file is produced
    """
    from pesummary.io.write import _multi_analysis_write

    _multi_analysis_write(
        _write_lalinference, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, sampler=sampler, dat=dat,
        file_format="lalinference", **kwargs
    )
