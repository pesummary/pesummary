# Licensed under an MIT style license -- see LICENSE.md

error_msg = (
    "Unable to install '{}'. You will not be able to use some of the inbuilt "
    "functions."
)
import numpy as np
from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead
from pesummary import conf
from pesummary.core.file.formats.ini import read_ini
from pesummary.utils.utils import logger
try:
    from pycbc.inference.io import loadfile
except ImportError:
    logger.warning(error_msg.format("pycbc"))

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

def read_pycbc(path, **kwargs):
    """Grab the parameters and samples in a pycbc file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    with loadfile(path, "r") as f:
        params = list(f["samples"].keys())
        _samples = f.read_samples(params)
        samples = {key: _samples[key] for key in _samples.dtype.names}
        try:
            from pycbc.conversions import snr_from_loglr
            samples["network_matched_filter_snr"] = snr_from_loglr(
                samples["loglikelihood"] - _samples.lognl
            )
        except AttributeError:
            pass
        try:
            config = read_ini(f.read_config_file(return_cp=False).read())
        except (KeyError, IndexError):
            # no config file stored
            config = None
        extra_kwargs = {
            "sampler": {}, "meta_data": {}, "other": dict(f.attrs)
        }
    try:
        extra_kwargs["sampler"][conf.log_evidence] = np.round(
            extra_kwargs["other"].pop("log_evidence"), 2
        )
        extra_kwargs["sampler"][conf.log_evidence_error] = np.round(
            extra_kwargs["other"].pop("dlog_evidence"), 2
        )
    except KeyError:
        pass

    low_freqs = [
        item for key, item in extra_kwargs["other"].items() if
        "likelihood_low_freq" in key
    ]
    if len(low_freqs):
        extra_kwargs["meta_data"]["f_low"] = np.min(low_freqs)
    try:
        extra_kwargs["meta_data"]["f_ref"] = extra_kwargs["other"].pop("f_ref")
    except KeyError:
        pass

    data = {
        "parameters": list(samples.keys()),
        "samples": np.array([_ for _ in samples.values()]).T.tolist(),
        "injection": None,
        "version": None,
        "kwargs": extra_kwargs,
        "config": config
    }
    return data


class PyCBC(GWSingleAnalysisRead):
    """PESummary wrapper of `pycbc` (https://git.ligo.org/lscsoft/bilby). The
    path_to_results_file argument will be passed directly to
    `pycbc.inference.io.loadfile`. All functions therefore use `pycbc`
    methods and requires `pycbc` to be installed.
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(PyCBC, self).__init__(path_to_results_file, **kwargs)
        self.load(self._grab_data_from_pycbc_file, **kwargs)

    def _grab_data_from_pycbc_file(self, *args, **kwargs):
        """Load the results file using the `pycbc` library
        """
        return read_pycbc(*args, **kwargs)
