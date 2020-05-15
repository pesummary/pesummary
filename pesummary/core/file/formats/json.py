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

import json
import numpy as np
import inspect


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


def read_json(path):
    """Grab the parameters and samples in a .json file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    """
    import json
    from pesummary.core.file.formats.base_read import Read

    with open(path, "r") as f:
        data = json.load(f)
    try:
        path, = Read.paths_to_key("posterior", data)
        path = path[0]
        path += "/posterior"
    except ValueError:
        try:
            path, = Read.paths_to_key("posterior_samples", data)
            path = path[0]
            path += "/posterior_samples"
        except ValueError:
            raise ValueError(
                "Unable to find a 'posterior' or 'posterior_samples' group in the "
                "file '{}'".format(path)
            )
    reduced_data, = Read.load_recursively(path, data)
    if "content" in list(reduced_data.keys()):
        reduced_data = reduced_data["content"]
    parameters = list(reduced_data.keys())

    samples = [
        [
            reduced_data[j][i] if not isinstance(reduced_data[j][i], dict)
            else reduced_data[j][i]["real"] for j in parameters
        ] for i in range(len(reduced_data[parameters[0]]))
    ]
    return parameters, samples


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
    from pesummary.utils.utils import check_filename

    default_filename = "pesummary_{}.json"
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=label, filename=filename,
        overwrite=overwrite
    )
    _samples = np.array(samples).T
    data = {
        "posterior_samples": {
            param: _samples[num] for num, param in enumerate(parameters)
        }
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, cls=cls)
