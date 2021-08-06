#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
import math
import argparse
import json
import h5py
from pathlib import Path

from pesummary.utils.utils import logger, check_file_exists_and_rename
from pesummary.utils.dict import paths_to_key
from pesummary.utils.exceptions import InputError
from pesummary.core.command_line import DelimiterSplitAction
from pesummary.gw.inputs import _GWInput
from pesummary.gw.file.meta_file import _GWMetaFile

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to modify a PESummary metafile from the
command line"""


class _Input(_GWInput):
    """Super class to handle the command line arguments
    """
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        if labels is not None and isinstance(labels, dict):
            self._labels = labels
        elif labels is not None:
            raise InputError(
                "Please provide an existing labels and the label you wish "
                "to replace it with `--labels existing:new`."
            )

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self._kwargs = kwargs
        if kwargs is not None and isinstance(kwargs, dict):
            self._kwargs = kwargs
        elif kwargs is not None:
            raise InputError(
                "Please provide the label, kwarg and value with '--kwargs "
                "label:kwarg:value`"
            )

    @property
    def replace_posterior(self):
        return self._replace_posterior

    @replace_posterior.setter
    def replace_posterior(self, replace_posterior):
        self._replace_posterior = replace_posterior
        if replace_posterior is not None and isinstance(replace_posterior, dict):
            self._replace_posterior = replace_posterior
        elif replace_posterior is not None:
            raise InputError(
                "Please provide the label, posterior and file path with "
                "value with '--replace_posterior "
                "label;posterior:/path/to/posterior.dat where ';' is the chosen "
                "delimiter and provided with '--delimiter ;`"
            )

    @property
    def remove_posterior(self):
        return self._remove_posterior

    @remove_posterior.setter
    def remove_posterior(self, remove_posterior):
        self._remove_posterior = remove_posterior
        if remove_posterior is not None and isinstance(remove_posterior, dict):
            self._remove_posterior = remove_posterior
        elif remove_posterior is not None:
            raise InputError(
                "Please provide the label and posterior with '--remove_posterior "
                "label:posterior`"
            )

    @property
    def store_skymap(self):
        return self._store_skymap

    @store_skymap.setter
    def store_skymap(self, store_skymap):
        self._store_skymap = store_skymap
        if store_skymap is not None and isinstance(store_skymap, dict):
            self._store_skymap = store_skymap
        elif store_skymap is not None:
            raise InputError(
                "Please provide the label and path to skymap with '--store_skymap "
                "label:path/to/skymap.fits`"
            )

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if samples is None:
            raise InputError(
                "Please provide a result file that you wish to modify"
            )
        if len(samples) > 1:
            raise InputError(
                "Only a single result file can be passed"
            )
        samples = samples[0]
        if not self.is_pesummary_metafile(samples):
            raise InputError(
                "Please provide a PESummary metafile to this executable"
            )
        self._samples = samples

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        extension = Path(self.samples).suffix
        if extension == ".h5" or extension == ".hdf5":
            from pesummary.core.file.formats.pesummary import PESummary
            from pandas import DataFrame

            with h5py.File(self.samples, "r") as f:
                data = PESummary._convert_hdf5_to_dict(f)
                for label in data.keys():
                    try:
                        data[label]["posterior_samples"] = DataFrame(
                            data[label]["posterior_samples"]
                        ).to_records(index=False, column_dtypes=np.float)
                    except KeyError:
                        pass
                    except Exception:
                        parameters = data[label]["posterior_samples"]["parameter_names"]
                        if isinstance(parameters[0], bytes):
                            parameters = [
                                parameter.decode("utf-8") for parameter in parameters
                            ]
                        samples = np.array([
                            j for j in data[label]["posterior_samples"]["samples"]
                        ].copy())
                        data[label]["posterior_samples"] = DataFrame.from_dict(
                            {
                                param: samples.T[num] for num, param in
                                enumerate(parameters)
                            }
                        ).to_records(index=False, column_dtypes=np.float)
                self._data = data
        elif extension == ".json":
            with open(self.samples, "r") as f:
                self._data = json.load(f)
        else:
            raise InputError(
                "The extension '{}' is not recognised".format(extension)
            )


class Input(_Input):
    """Class to handle the command line arguments

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options

    Attributes
    ----------
    samples: str
        path to a PESummary meta file that you wish to modify
    labels: dict
        dictionary of labels that you wish to modify. Key is the existing label
        and item is the new label
    """
    def __init__(self, opts, ignore_copy=False):
        logger.info("Command line arguments: %s" % (opts))
        self.opts = opts
        self.existing = None
        self.webdir = self.opts.webdir
        self.samples = self.opts.samples
        self.labels = self.opts.labels
        self.kwargs = self.opts.kwargs
        self.replace_posterior = self.opts.replace_posterior
        self.remove_posterior = self.opts.remove_posterior
        self.store_skymap = self.opts.store_skymap
        self.hdf5 = not self.opts.save_to_json
        self.overwrite = self.opts.overwrite
        self.force_replace = self.opts.force_replace
        self.data = None
        if self.opts.descriptions is not None:
            import copy
            self._labels_copy = copy.deepcopy(self._labels)
            self._labels = self.data.keys()
            self.descriptions = self.opts.descriptions
            self._labels = self._labels_copy
        else:
            self._descriptions = None


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", dest="labels", nargs='+', action=DelimiterSplitAction,
        help=("labels you wish to modify. Syntax: `--labels existing:new` "
              "where ':' is the default delimiter"),
        default=None
    )
    parser.add_argument(
        "--descriptions", nargs="+", action=DelimiterSplitAction, help=(
            "descriptions you wish to modify. Syntax `--descriptions label:desc` "
            "where label is the analysis you wish to change and desc is the "
            "new description"
        ), default=None
    )
    parser.add_argument(
        "-s", "--samples", dest="samples", default=None, nargs='+',
        help="Path to PESummary meta file you wish to modify"
    )
    parser.add_argument(
        "-w", "--webdir", dest="webdir", default="./", metavar="DIR",
        help="Directory to write the output file"
    )
    parser.add_argument(
        "--save_to_json", action="store_true", default=False,
        help="save the modified data in json format"
    )
    parser.add_argument(
        "--delimiter", dest="delimiter", default=":",
        help="Delimiter used to seperate the existing and new quantity"
    )
    parser.add_argument(
        "--kwargs", dest="kwargs", nargs='+', action=DelimiterSplitAction,
        help=("kwargs you wish to modify. Syntax: `--kwargs label/kwarg:item` "
              "where '/' is a delimiter of your choosing (it cannot be ':'), "
              "kwarg is the kwarg name and item is the value of the kwarg"),
        default=None
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help=("Overwrite the supplied PESummary meta file with the modified "
              "version")
    )
    parser.add_argument(
        "--replace_posterior", nargs='+', action=DelimiterSplitAction,
        help=("Replace the posterior for a given label. Syntax: "
              "--replace_posterior label;a:/path/to/posterior.dat where "
              "';' is a delimiter of your choosing (it cannot be '/' or ':'), "
              "a is the posterior you wish to replace and item is a path "
              "to a one column ascii file containing the posterior samples "
              "(/path/to/posterior.dat)"),
        default=None
    )
    parser.add_argument(
        "--remove_posterior", nargs='+', action=DelimiterSplitAction,
        help=("Remove a posterior distribution for a given label. Syntax: "
              "--remove_posterior label:a where a is the posterior you wish to remove"),
        default=None
    )
    parser.add_argument(
        "--store_skymap", nargs='+', action=DelimiterSplitAction,
        help=("Store the contents of a fits file in the metafile. Syntax: "
              "--store_skymap label:path/to/skymap.fits"),
        default=None
    )
    parser.add_argument(
        "--force_replace", action="store_true", default=False,
        help=("Override the ValueError raised if the data is already stored in the "
              "result file")
    )
    return parser


def _check_label(data, label, message, logger_level="warn"):
    """Check that a given label is stored in the data. If it is not stored
    print a warning message

    Parameters
    ----------
    data: dict
        dictionary containing the data
    label: str
        name of the label you wish to check
    message: str
        message you wish to print in logger when the label is not stored
    logger_level: str, optional
        the logger level of the message
    """
    if label not in data.keys():
        getattr(logger, logger_level)(message)
        return False
    return True


def _modify_labels(data, labels=None):
    """Modify the existing labels in the data

    Parameters
    ----------
    data: dict
        dictionary containing the data
    labels: dict
        dictionary of labels showing the existing label, key, and the new
        label, item
    """
    for existing, new in labels.items():
        if existing not in data.keys():
            logger.warning(
                "Unable to find label '{}' in the root of the metafile. "
                "Checking inside the groups".format(existing)
            )
            for key in data.keys():
                if existing in data[key].keys():
                    data[key][new] = data[key].pop(existing)
        else:
            data[new] = data.pop(existing)
    return data


def _modify_descriptions(data, descriptions={}):
    """Modify the existing descriptions in the data

    Parameters
    ----------
    data: dict
        dictionary containing the data
    descriptions: dict
        dictionary of descriptions with label as the key and new description as
        the item
    """
    message = (
        "Unable to find label '{}' in the metafile. Unable to modify "
        "description"
    )
    for label, new_desc in descriptions.items():
        check = _check_label(data, label, message.format(label))
        if check:
            if "description" not in data[label].keys():
                data[label]["description"] = []
            data[label]["description"] = [new_desc]
    return data


def _modify_kwargs(data, kwargs=None):
    """Modify kwargs that are stored in the data

    Parameters
    ----------
    data: dict
        dictionary containing the data
    kwargs: dict
        dictionary of kwargs showing the label as key and kwarg:value as the
        item
    """
    from pesummary.core.file.formats.base_read import Read

    def add_to_meta_data(data, label, string):
        kwarg, value = string.split(":")
        try:
            _group, = paths_to_key(kwarg, data[label]["meta_data"])
            group = _group[0]
        except ValueError:
            group = "other"
        if group == "other" and group not in data[label]["meta_data"].keys():
            data[label]["meta_data"]["other"] = {}
        data[label]["meta_data"][group][kwarg] = value
        return data

    message = "Unable to find label '{}' in the metafile. Unable to modify kwargs"
    for label, item in kwargs.items():
        check = _check_label(data, label, message.format(label))
        if check:
            if isinstance(item, list):
                for _item in item:
                    data = add_to_meta_data(data, label, _item)
            else:
                data = add_to_meta_data(data, label, item)
    return data


def _modify_posterior(data, kwargs=None):
    """Replace a posterior distribution that is stored in the data

    Parameters
    ----------
    data: dict
        dictionary containing the data
    kwargs: dict
        dictionary of kwargs showing the label as key and posterior:path as the
        item
    """
    def _replace_posterior(data, string):
        posterior, path = string.split(":")
        _data = np.genfromtxt(path, usecols=0)
        if math.isnan(_data[0]):
            _data = np.genfromtxt(path, names=True, usecols=0)
            _data = _data[_data.dtype.names[0]]
        if posterior in data[label]["posterior_samples"].dtype.names:
            data[label]["posterior_samples"][posterior] = _data
        else:
            from numpy.lib.recfunctions import append_fields

            data[label]["posterior_samples"] = append_fields(
                data[label]["posterior_samples"], posterior, _data, usemask=False
            )
        return data

    message = "Unable to find label '{}' in the metafile. Unable to modify posterior"
    for label, item in kwargs.items():
        check = _check_label(data, label, message.format(label))
        if check:
            if isinstance(item, list):
                for _item in item:
                    data = _replace_posterior(data, _item)
            else:
                data = _replace_posterior(data, item)
    return data


def _remove_posterior(data, kwargs=None):
    """Remove a posterior distribution that is stored in the data

    Parameters
    ----------
    data: dict
        dictionary containing the data
    kwargs: dict
        dictionary of kwargs showing the label as key and posterior as the item
    """
    def _rmfield(array, *fieldnames_to_remove):
        return array[
            [name for name in array.dtype.names if name not in fieldnames_to_remove]
        ]

    message = "Unable to find label '{}' in the metafile. Unable to remove posterior"
    for label, item in kwargs.items():
        check = _check_label(data, label, message.format(label))
        if check:
            group = "posterior_samples"
            if isinstance(item, list):
                for _item in item:
                    data[label][group] = _rmfield(data[label][group], _item)
            else:
                data[label][group] = _rmfield(data[label][group], item)
    return data


def _store_skymap(data, kwargs=None, replace=False):
    """Store a skymap in the metafile

    Parameters
    ----------
    data: dict
        dictionary containing the data
    kwargs: dict
        dictionary of kwargs showing the label as key and posterior as the item
    replace: dict
        replace a skymap already stored in the result file
    """
    from pesummary.io import read

    message = "Unable to find label '{}' in the metafile. Unable to store skymap"
    for label, path in kwargs.items():
        check = _check_label(data, label, message.format(label))
        if check:
            skymap = read(path, skymap=True)
            if "skymap" not in data[label].keys():
                data[label]["skymap"] = {}
            if "meta_data" not in data[label]["skymap"].keys():
                data[label]["skymap"]["meta_data"] = {}
            if "data" in data[label]["skymap"].keys() and not replace:
                raise ValueError(
                    "Skymap already found in result file for {}. If you wish to replace "
                    "the skymap, add the command line argument '--force_replace".format(
                        label
                    )
                )
            elif "data" in data[label]["skymap"].keys():
                logger.warning("Replacing skymap data for {}".format(label))
            data[label]["skymap"]["data"] = skymap
            for key in skymap.meta_data:
                data[label]["skymap"]["meta_data"][key] = skymap.meta_data[key]
    return data


def modify(data, function, **kwargs):
    """Modify the data according to a given function

    Parameters
    ----------
    data: dict
        dictionary containing the data
    function:
        function you wish to use to modify the data
    kwargs: dict
        dictionary of kwargs for function
    """
    func_map = {
        "labels": _modify_labels,
        "descriptions": _modify_descriptions,
        "kwargs": _modify_kwargs,
        "add_posterior": _modify_posterior,
        "rm_posterior": _remove_posterior,
        "skymap": _store_skymap,
    }
    return func_map[function](data, **kwargs)


def _main(opts):
    """
    """
    args = Input(opts)
    if not args.overwrite:
        meta_file = os.path.join(
            args.webdir, "modified_posterior_samples.{}".format(
                "h5" if args.hdf5 else "json"
            )
        )
        check_file_exists_and_rename(meta_file)
    else:
        meta_file = args.samples
    if args.labels is not None:
        modified_data = modify(args.data, "labels", labels=args.labels)
    if args.descriptions is not None:
        modified_data = modify(
            args.data, "descriptions", descriptions=args.descriptions
        )
    if args.kwargs is not None:
        modified_data = modify(args.data, "kwargs", kwargs=args.kwargs)
    if args.replace_posterior is not None:
        modified_data = modify(args.data, "add_posterior", kwargs=args.replace_posterior)
    if args.remove_posterior is not None:
        modified_data = modify(args.data, "rm_posterior", kwargs=args.remove_posterior)
    if args.store_skymap is not None:
        modified_data = modify(
            args.data, "skymap", kwargs=args.store_skymap, replace=args.force_replace
        )
    logger.info(
        "Saving the modified data to '{}'".format(meta_file)
    )
    if args.hdf5:
        _GWMetaFile.save_to_hdf5(
            modified_data, list(modified_data.keys()), None, meta_file,
            no_convert=True
        )
    else:
        _GWMetaFile.save_to_json(modified_data, meta_file)


def main(args=None):
    """
    """
    parser = command_line()
    opts = parser.parse_args(args=args)
    return _main(opts)


if __name__ == "__main__":
    main()
