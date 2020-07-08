#! /usr/bin/env python

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

import pesummary
from pesummary.core.parser import parser
from pesummary.utils.utils import logger
from pesummary.io import read
import numpy as np
import argparse

__doc__ = """This executable is used to compare multiple files"""
COMPARISON_PROPERTIES = [
    "posterior_samples", "config", "priors", "psds"
]
SAME_STRING = "The result files match for entry: '{}'"


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument(
        "-s", "--samples", dest="samples", default=None, nargs='+',
        help="Posterior samples hdf5 file"
    )
    _parser.add_argument(
        "--properties_to_compare", dest="compare", nargs='+',
        default=["posterior_samples"], help=(
            "list of properties you wish to compare between the files. Default "
            "posterior_samples"
        ), choices=COMPARISON_PROPERTIES
    )
    _parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print useful information for debugging purposes"
    )
    return _parser


def _comparison_string(path, values=None, _type=None):
    """Print the comparison string

    Parameters
    ----------
    path: list
        list containing the path to the dataset being compared
    values: list, optional
        list containing the entries which are being compared
    _type: optional
        the type of the values being compared
    """
    _path = "/".join(path)
    if values is None:
        logger.info(
            "'{}' is not in both result files. Unable to compare".format(_path)
        )
    else:
        string = (
            "The result files differ for the following entry: '{}'. ".format(
                _path
            )
        )
        if _type == list:
            try:
                _diff = np.max(np.array(values[0]) - np.array(values[1]))
                string += "The maximum difference is: {}".format(_diff)
            except ValueError as e:
                if "could not be broadcast together" in str(e):
                    string += (
                        "Datasets contain different number of samples: "
                        "{}".format(
                            ", ".join([str(len(values[0])), str(len(values[1]))])
                        )
                    )
        else:
            string += "The entries are: {}".format(", ".join(values))

        logger.info(string)


def _compare(data, path=None):
    """Compare multiple posterior samples

    Parameters
    ----------
    data: list, dict
        data structure which is to be compared
    path: list
        path to the data structure being compared
    """
    if path is None:
        path = []

    if isinstance(data[0], dict):
        for key, value in data[0].items():
            _path = path + [key]
            if not all(key in _dict.keys() for _dict in data):
                _comparison_string(_path)
                continue
            if isinstance(value, dict):
                _data = [_dict[key] for _dict in data]
                _compare(_data, path=_path)
            else:
                _compare_datasets([_data[key] for _data in data], path=_path)
    else:
        _compare_datasets(data, path=path)


def _compare_datasets(data, path=[]):
    """Compare two datasets

    Parameters
    ----------
    data: list, str, int, float
        dataset which you want to compare
    path: list, optional
        path to the dataset being compared
    """
    array_types = (list, pesummary.utils.samples_dict.Array, np.ndarray)
    numeric_types = (float, int, np.number)
    string_types = (str, bytes)

    if isinstance(data[0], array_types):
        try:
            np.testing.assert_almost_equal(data[0], data[1])
            logger.debug(SAME_STRING.format("/".join(path)))
        except AssertionError:
            _comparison_string(path, values=data, _type=list)
    elif isinstance(data[0], numeric_types):
        if not all(i == data[0] for i in data):
            _comparison_string(path, values=data, _type=float)
        else:
            logger.debug(SAME_STRING.format("/".join(path)))
    elif isinstance(data[0], string_types):
        if not all(i == data[0] for i in data):
            _comparison_string(path, values=data, _type=str)
        else:
            logger.debug(SAME_STRING.format("/".join(path)))
    else:
        raise ValueError(
            "Unknown data format. Unable to compare: {}".format(
                ", ".join([str(i) for i in data])
            )
        )


def compare(samples, properties_to_compare=COMPARISON_PROPERTIES):
    """Compare multiple posterior samples

    Parameters
    ----------
    samples: list
        list of files you wish to compare
    properties_to_compare: list, optional
        optional list of properties you wish to compare
    """
    data = [read(path, disable_prior_conversion=True) for path in samples]
    for prop in properties_to_compare:
        if prop.lower() == "posterior_samples":
            prop = "samples_dict"
        _data = [
            getattr(f, prop) if hasattr(f, prop) else False for f in
            data
        ]
        if False in _data:
            logger.warning(
                "Unable to compare the property '{}' because not all files "
                "share this property".format(prop)
            )
            continue
        _compare(_data, path=[prop])


def main(args=None):
    """Top level interface for `summarycompare`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    compare(opts.samples, opts.compare)


if __name__ == "__main__":
    main()
