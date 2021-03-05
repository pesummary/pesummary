# Licensed under an MIT style license -- see LICENSE.md

import json
import numpy as np
import inspect
from pesummary.utils.dict import load_recursively, paths_to_key

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class PESummaryJsonEncoder(json.JSONEncoder):
    """Personalised JSON encoder for PESummary
    """
    def default(self, obj):
        """Return a json serializable object for 'obj'

        Parameters
        ----------
        obj: object
            object you wish to make json serializable
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if inspect.isfunction(obj):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool, np.bool_, bool)):
            return str(obj)
        elif isinstance(obj, bytes):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def PESummaryJsonDecoder(obj):
    if isinstance(obj, dict):
        if "__array__" in obj.keys() and "content" in obj.keys():
            return obj["content"]
        elif "__complex__" in obj.keys():
            return obj["real"] + obj["imag"] * 1j
    return obj


def read_json(path, path_to_samples=None, decoder=PESummaryJsonDecoder):
    """Grab the parameters and samples in a .json file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    import json
    from pesummary.core.file.formats.base_read import Read

    with open(path, "r") as f:
        data = json.load(f, object_hook=decoder)
    if not path_to_samples:
        try:
            path_to_samples, = paths_to_key("posterior", data)
            path_to_samples = path_to_samples[0]
            path_to_samples += "/posterior"
        except ValueError:
            try:
                path_to_samples, = paths_to_key("posterior_samples", data)
                path_to_samples = path_to_samples[0]
                path_to_samples += "/posterior_samples"
            except ValueError:
                raise ValueError(
                    "Unable to find a 'posterior' or 'posterior_samples' group "
                    "in the file '{}'".format(path_to_samples)
                )
    reduced_data, = load_recursively(path_to_samples, data)
    if "content" in list(reduced_data.keys()):
        reduced_data = reduced_data["content"]
    parameters = list(reduced_data.keys())
    reduced_data = {
        j: list([reduced_data[j]]) if not isinstance(reduced_data[j], list) else
        reduced_data[j] for j in parameters
    }
    _original_parameters = reduced_data.copy().keys()
    _non_numeric = []
    numeric_types = (float, int, np.number)
    for key in _original_parameters:
        if any(np.iscomplex(reduced_data[key])):
            reduced_data[key + "_amp"] = np.abs(reduced_data[key])
            reduced_data[key + "_angle"] = np.angle(reduced_data[key])
            reduced_data[key] = np.real(reduced_data[key])
        elif not all(isinstance(_, numeric_types) for _ in reduced_data[key]):
            _non_numeric.append(key)
        elif all(isinstance(_, (bool, np.bool_)) for _ in reduced_data[key]):
            _non_numeric.append(key)

    parameters = list(reduced_data.keys())
    if len(_non_numeric):
        from pesummary.utils.utils import logger
        logger.info(
            "Removing the parameters: '{}' from the posterior table as they "
            "are non-numeric".format(", ".join(_non_numeric))
        )
    for key in _non_numeric:
        parameters.remove(key)
    samples = [
        [reduced_data[j][i] for j in parameters] for i in
        range(len(reduced_data[parameters[0]]))
    ]
    return parameters, samples


def _write_json(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    indent=4, sort_keys=True, dataset_name="posterior_samples",
    cls=PESummaryJsonEncoder, **kwargs
):
    """Write a set of samples to a json file

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
    indent: int, optional
        The indentation to use in json.dump. Default 4
    sort_keys: Bool, optional
        Whether or not to sort the keys in json.dump. Default True
    dataset_name: str, optional
        name of the dataset to store a set of samples. Default posterior_samples
    cls: class, optional
        Class to use as the JsonEncoder. Default PESumamryJsonEncoder
    """
    from pesummary.utils.utils import check_filename

    default_filename = "pesummary_{}.json"
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=label, filename=filename,
        overwrite=overwrite
    )
    _samples = np.array(samples).T
    data = {
        dataset_name: {
            param: _samples[num] for num, param in enumerate(parameters)
        }
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, cls=cls)


def write_json(
    parameters, samples, outdir="./", label=None, filename=None, overwrite=False,
    indent=4, sort_keys=True, cls=PESummaryJsonEncoder, **kwargs
):
    """Write a set of samples to a json file

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
    indent: int, optional
        The indentation to use in json.dump. Default 4
    sort_keys: Bool, optional
        Whether or not to sort the keys in json.dump. Default True
    cls: class, optional
        Class to use as the JsonEncoder. Default PESumamryJsonEncoder
    """
    from pesummary.io.write import _multi_analysis_write

    _multi_analysis_write(
        _write_json, parameters, samples, outdir=outdir, label=label,
        filename=filename, overwrite=overwrite, indent=indent,
        sort_keys=sort_keys, cls=cls, file_format="json", **kwargs
    )
