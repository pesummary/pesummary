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

import os
from pesummary.gw.file.read import read as GWRead
from pesummary.core.file.read import read as Read
from pesummary.gw.file.meta_file import _GWMetaFile
from pesummary.core.file.meta_file import _MetaFile
from pesummary.utils.utils import make_dir

__doc__ = """This executable is used to combine multiple PESummary metafiles
into a single PESummry metafile"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default="./")
    parser.add_argument("-s", "--samples", dest="samples", nargs='+',
                        help="Path to PESummary metafile",
                        default=None)
    parser.add_argument("--save_to_hdf5", action="store_true",
                        help="save the meta file in hdf5 format", default=False)
    parser.add_argument("--gw", action="store_true",
                        help="run with the gravitational wave pipeline",
                        default=False)
    return parser


def grab_information_from_file(samples, read_function, required_parameters):
    """Grab the information stored in the PESummary metafile

    Parameters
    ----------
    samples: list
        list of file paths
    read_function: function
        function to read in the PESummary metafle. Should be either
        pesummary.gw.file.read.read or pesummary.core.file.read.read
    required_parameters: list
        list of attributes that you would like to return from the
        PESummary metafile
    """
    files = [read_function(i) for i in samples]
    labels = [i.labels for i in files]
    combined = [i for subset in labels for i in subset]
    duplicated = set(
        [x for x in combined if combined.count(x) > 1]
    )
    if len(duplicated) > 0:
        raise Exception(
            "Unable to combine result files because there are duplicated labels"
        )
    full_list = {j: [] for j in required_parameters}
    full_list["labels"] = combined
    for idx, i in enumerate(files):
        mydict = {j: getattr(i, j) for j in required_parameters}
        for j in required_parameters:
            if isinstance(mydict[j], dict):
                if idx == 0:
                    full_list[j] = {}
                full_list[j].update(mydict[j])
            elif isinstance(mydict[j], list):
                if idx == 0:
                    full_list[j] = {}
                for num, k in enumerate(mydict[j]):
                    full_list[j][labels[idx][num]] = mydict[j][num]
            elif mydict[j] is None:
                for k in combined:
                    full_list[j].append(None)
    return full_list


def gw_meta_file(opts):
    """Read in the files with the pesummary.gw.file.read.read function and
    return a pesummary.gw.file.meta_file._GWMetaFile object

    Parameters
    ----------
    opts: argparse.Namespace
        argparse namespace object that contains the passed arguments
    """
    read_function = GWRead
    metafile_function = _GWMetaFile
    required_parameters = [
        "samples_dict", "parameters", "config", "injection_parameters",
        "input_version", "extra_kwargs", "calibration", "psd", "approximant"]

    data = grab_information_from_file(
        opts.samples, read_function, required_parameters)
    if data["psd"] is None or all(i is None for i in data["psd"]):
        data["psd"] = {i: None for i in data["labels"]}
    if data["calibration"] is None or all(i is None for i in data["calibration"]):
        data["calibration"] = {i: None for i in data["labels"]}
    meta_file = _GWMetaFile(data["samples_dict"],
                            data["labels"], data["config"],
                            data["injection_parameters"],
                            data["input_version"], data["extra_kwargs"],
                            calibration=data["calibration"], psd=data["psd"],
                            approximant=data["approximant"], webdir=opts.webdir)
    return meta_file


def core_meta_file(opts):
    """Read in the files with the pesummary.core.file.read.read function and
    return a pesummary.core.file.meta_file._MetaFile object

    Parameters
    ----------
    opts: argparse.Namespace
        argparse namespace object that contains the passed arguments
    """
    read_function = Read
    metafile_function = _MetaFile
    required_parameters = [
        "samples_dict", "parameters", "config", "injection_parameters",
        "input_version", "extra_kwargs"]

    data = grab_information_from_file(
        opts.samples, read_function, required_parameters)

    meta_file = _MetaFile(data["samples_dict"], data["labels"],
                          data["config"], data["injection_parameters"],
                          data["input_version"], data["extra_kwargs"],
                          webdir=opts.webdir)
    return meta_file


def main():
    """Top level interface for `summarycombine_metafiles`
    """
    parser = command_line()
    opts = parser.parse_args()

    if opts.webdir and not os.path.isdir(opts.webdir + "/samples"):
        make_dir(opts.webdir + "/samples")

    if opts.gw:
        meta_file = gw_meta_file(opts)
    else:
        meta_file = core_meta_file(opts)

    meta_file.make_dictionary()
    if not opts.save_to_hdf5:
        meta_file.save_to_json()
    else:
        meta_file.save_to_hdf5()


if __name__ == "__main__":
    main()
