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
from pesummary.gw.inputs import GWPostProcessing
from pesummary.gw.command_line import DictionaryAction
from pesummary.utils.utils import make_dir

import numpy as np

__doc__ = """This executable is used to combine multiple result files into a
single PESummary metafile"""


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
    parser.add_argument("--labels", dest="labels",
                        help="labels used to distinguish runs", nargs='+',
                        default=None)
    parser.add_argument("-c", "--config", dest="config",
                        help=("configuration file associcated with "
                              "each samples file."),
                        nargs='+', default=None)
    parser.add_argument("-i", "--inj_file", dest="inj_file",
                        help="path to injetcion file", nargs='+', default=None)
    parser.add_argument("--save_to_hdf5", action="store_true",
                        help="save the meta file in hdf5 format", default=False)

    gw_group = parser.add_argument_group(
        "Options specific for gravitational wave results files")
    gw_group.add_argument("-a", "--approximant", dest="approximant",
                          help=("waveform approximant used to generate "
                                "samples"), nargs='+', default=None)
    gw_group.add_argument("--psd", dest="psd", action=DictionaryAction,
                          help="psd files used", nargs='+', default=None)
    gw_group.add_argument("--calibration", dest="calibration",
                          help="files for the calibration envelope",
                          nargs="+", action=DictionaryAction, default=None)
    gw_group.add_argument("--trigfile", dest="inj_file",
                          help="xml file containing the trigger values",
                          nargs='+', default=None)
    gw_group.add_argument("--gw", action="store_true",
                          help="run with the gravitational wave pipeline",
                          default=False)
    return parser


def grab_information_from_file(samples, read_function, required_parameters,
                               config_file=None, injection_file=None):
    """Grab the information stored in result files

    Parameters
    ----------
    samples: list
        list of file paths
    read_function: function
        function to read in the result file. Should be either
        pesummary.gw.file.read.read or pesummary.core.file.read.read
    required_parameters: list
        list of attributes that you would like to return from the result file
    """
    full_list = {i: [] for i in required_parameters + ["injection_parameters"]}

    for idx, i in enumerate(samples):
        f = read_function(i)
        if config_file is not None and config_file[idx]:
            f.add_fixed_parameters_from_config_file(config_file[idx])
        if injection_file is not None and injection_file[idx]:
            f.add_injection_parameters_from_file(injection_file[idx])
        f.generate_all_posterior_samples()
        mydict = {j: getattr(f, j) for j in required_parameters}
        for j in required_parameters:
            full_list[j].append(mydict[j])
        if hasattr(f, "injection_parameters"):
            injection = f.injection_parameters
            if injection is not None:
                for i in mydict["parameters"]:
                    if i not in list(injection.keys()):
                        injection[i] = float("nan")
            else:
                injection = {i: j for i, j in zip(
                    mydict["parameters"],
                    [float("nan")] * len(mydict["parameters"]))
                }
        else:
            injection = {i: j for i, j in zip(
                mydict["parameters"],
                [float("nan")] * len(mydict["parameters"]))
            }
        full_list["injection_parameters"].append(injection)
    return full_list


def get_psd_data(psd, number=1):
    """Return the psd data passed from the command line

    Parameters
    ----------
    psd: list/dict
        paths to the psd files
    number: int, optional
        the number of result file passed
    """
    psd_data_list = []
    if isinstance(psd, dict):
        ifos = list(psd.keys())
        psd_labels = [ifos] * number
    else:
        psd_labels = [
            [GWPostProcessing._IFO_from_file_name(i) for i in psd]
        ] * number
        psd = {i: j for i, j in zip(psd_labels[0], psd)}
    for num, i in enumerate(psd_labels):
        psd_data = {}
        for j in i:
            if isinstance(psd[j], list):
                psd_data[j] = np.genfromtxt(psd[j][num], skip_footer=1).tolist()
            else:
                psd_data[j] = np.genfromtxt(psd[j], skip_footer=1).tolist()
        psd_data_list.append(psd_data)
    return psd_data_list


def get_calibration_data(calibration, number=1):
    """Return the calibration data passed from the command line

    Parameters
    ----------
    calibration: list/dict
        paths to the calibration envelopes
    number: int, optional
        the number of result file passed
    """
    calibration_data_list = []
    if isinstance(calibration, dict):
        ifos = list(calibration.keys())
        calibration_labels = [ifos] * number
    else:
        calibration_labels = [
            [GWPostProcessing._IFO_from_file_name(i) for i in calibration]
        ] * number
        calibration = {i: j for i, j in zip(calibration_labels[0], calibration)}
    for num, i in enumerate(calibration_labels):
        calibration_data = {}
        for j in i:
            if isinstance(calibration[j], list):
                calibration_data[j] = np.genfromtxt(calibration[j][num]).tolist()
            else:
                calibration_data[j] = np.genfromtxt(calibration[j]).tolist()
        calibration_data_list.append(calibration_data)
    return calibration_data_list


def gw_meta_file(opts):
    """Read in the files with the pesummary.gw.file.read.read function and
    return a pesummary.gw.file.meta_file._GWMetaFile object

    Parameters
    ----------
    opts: argparse.Namespace
        argparse namespace object that contains the passed arguments
    """
    read_function = GWRead
    required_parameters = [
        "parameters", "samples", "extra_kwargs", "input_version"]
    data = grab_information_from_file(
        opts.samples, read_function, required_parameters,
        config_file=opts.config, injection_file=opts.inj_file)
    data["labels"] = opts.labels

    if opts.psd is None:
        data["psd"] = [None for i in range(len(data["labels"]))]
    else:
        data["psd"] = get_psd_data(opts.psd, number=len(data["labels"]))
    if opts.calibration is None:
        data["calibration"] = [None for i in range(len(data["labels"]))]
    else:
        data["calibration"] = get_calibration_data(
            opts.calibration, number=len(data["labels"])
        )
    if opts.approximant is None:
        data["approximant"] = [None for i in range(len(data["labels"]))]
    else:
        data["approximant"] = opts.approximant
    data["config"] = opts.config

    meta_file = _GWMetaFile(data["parameters"], data["samples"],
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
    required_parameters = [
        "parameters", "samples", "extra_kwargs", "input_version"]
    data = grab_information_from_file(
        opts.samples, read_function, required_parameters,
        config_file=opts.config, injection_file=opts.inj_file)
    data["labels"] = opts.labels
    data["config"] = opts.config

    meta_file = _MetaFile(data["parameters"], data["samples"], data["labels"],
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

    meta_file.generate_meta_file_data()
    if not opts.save_to_hdf5:
        meta_file.save_to_json()
    else:
        meta_file.save_to_hdf5()


if __name__ == "__main__":
    main()
