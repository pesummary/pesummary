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

import argparse
import copy
import collections
import sys
import fnmatch
from pesummary.utils.utils import command_line_arguments
from pesummary.core.command_line import DictionaryAction


def _remnant_command_line_arguments(parser):
    """Add remnant specific command line options to the Argument Parser

    Parameters
    ----------
    parser: object
        OptionParser instance
    """
    remnant_group = parser.add_argument_group(
        "Options specific for calculating the remnant properties\n"
        "-------------------------------------------------------"
    )
    remnant_group.add_argument(
        "--evolve_spins", action="store_true",
        help=(
            "Evolve the spins up to the Schwarzschild ISCO frequency for "
            "remnant fits evaluation"
        ), default=False
    )
    remnant_group.add_argument(
        "--NRSur_fits", nargs="?", dest="NRSur_fits",
        help=(
            "The NRSurrogate you wish to use to calculate the remnant "
            "quantities from your posterior samples. If not passed, the "
            "average NR fits are used"
        ), default=False
    )
    remnant_group.add_argument(
        "--waveform_fits", action="store_true",
        help=(
            "Use the provided approximant (either from command line or stored "
            "in the result file) to calculate the remnant quantities from your "
            "posterior samples. If not passed, the average NR fits are used"
        ), default=False
    )
    return remnant_group


def insert_gwspecific_option_group(parser):
    """Add gravitational wave related options to the optparser object

    Parameters
    ----------
    parser: object
        OptionParser instance.
    """
    gw_group = parser.add_argument_group(
        "\n\n=====================================================\n"
        "Options specific for gravitational wave results files\n"
        "====================================================="
    )

    gw_group.add_argument("-a", "--approximant", dest="approximant",
                          help=("waveform approximant used to generate "
                                "samples"), nargs='+', default=None)
    gw_group.add_argument("--sensitivity", action="store_true",
                          help="generate sky sensitivities for HL, HLV",
                          default=False)
    gw_group.add_argument("--gracedb", dest="gracedb",
                          help="gracedb of the event", default=None)
    gw_group.add_argument("--psd", dest="psd", action=DictionaryAction,
                          help="psd files used", nargs='+', default={})
    gw_group.add_argument("--{}_psd", dest="example_psd",
                          help=("psd files used for a specific label. '{}' "
                                "should be replaced with the label of "
                                "interest. For example "
                                "--IMRPhenomPv3_psd H1:IF0_psd.dat"),
                          default=None, metavar="IFO:PATH_to_PSD.dat")
    gw_group.add_argument("--calibration", dest="calibration",
                          help="files for the calibration envelope",
                          nargs="+", action=DictionaryAction, default={})
    gw_group.add_argument("--{}_calibration", dest="example_calibration",
                          help=("calibration files used for a specific label. "
                                "'{}' should be replaced with the label of "
                                "interest. For example "
                                "--IMRPhenomPv3_calibration H1:IF0_cal.dat"),
                          default=None, metavar="IFO:PATH_to_CAL.txt")
    gw_group.add_argument("--trigfile", dest="inj_file",
                          help="xml file containing the trigger values",
                          nargs='+', default=None)
    gw_group.add_argument("--gwdata", dest="gwdata",
                          help="channels and paths to strain cache files",
                          action=DictionaryAction,
                          metavar="CHANNEL:CACHEFILE or PICKLEFILE",
                          nargs="+", default=None)
    gw_group.add_argument("--multi_threading_for_skymap", action="store_true",
                          help=("use multi-threading to speed up generation of "
                                "ligo.skymap"), default=False)
    gw_group.add_argument("--nsamples_for_skymap", dest="nsamples_for_skymap",
                          help=("The number of samples used to generate the "
                                "ligo.skymap. These samples will be randomly "
                                "drawn from the posterior distributions"),
                          default=None)
    gw_group.add_argument("--f_low", dest="f_low",
                          help=("Low frequency cutoff used to generate the "
                                "samples"),
                          nargs='+', default=None)
    gw_group.add_argument("--f_ref", dest="f_ref",
                          help=("Reference frequency used to generate the "
                                "samples"),
                          nargs='+', default=None)
    gw_group.add_argument("--no_ligo_skymap", action="store_true",
                          help="do not generate a skymap with ligo.skymap",
                          default=False)
    gw_group.add_argument("--gw", action="store_true",
                          help="run with the gravitational wave pipeline",
                          default=False)
    gw_group.add_argument("--public", action="store_true",
                          help="generate public facing summary pages",
                          default=False)
    gw_group.add_argument("--redshift_method", dest="redshift_method",
                          help=("The method to use when estimating the redshift"),
                          choices=["approx", "exact"], default="approx")
    gw_group.add_argument("--cosmology", dest="cosmology",
                          help=("The cosmology to use when calculating "
                                "the redshift"),
                          default="Planck15")
    gw_group.add_argument("--no_conversion", action="store_true",
                          help="Do not generate any conversions",
                          default=False)
    remnant_group = _remnant_command_line_arguments(parser)
    return gw_group


def add_dynamic_argparse(
        existing_namespace, pattern, example="--{}_psd", default={},
        nargs='+', action=DictionaryAction, command_line=None
):
    """Add a dynamic argparse argument and add it to an existing
    argparse.Namespace object

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    pattern: str
        generic pattern for customg argparse. For example '--*_psd'
    example: str, optional
        example string to demonstrate usage
    default: obj, optional
        the default argument for the dynamic argparse object
    nargs: str
    action: argparse.Action
        argparse action to use for the dynamic argparse
    command_line: str, optional
        command line you wish to pass. If None, command line taken from
        sys.argv
    """
    if command_line is None:
        command_line = command_line_arguments()
    commands = fnmatch.filter(command_line, pattern)
    duplicates = [
        item for item, count in collections.Counter(commands).items() if
        count > 1
    ]
    if example in commands:
        commands.remove(example)
    if len(duplicates) > 0:
        raise Exception(
            "'{}' has been repeated. Please give a unique argument".format(
                duplicates[0]
            )
        )
    parser = argparse.ArgumentParser()
    for i in commands:
        parser.add_argument(
            i, help=argparse.SUPPRESS, action=action, nargs=nargs,
            default=default
        )
    args, unknown = parser.parse_known_args(args=command_line)
    existing_namespace.__dict__.update(vars(args))
    return args, unknown


def add_dynamic_PSD_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic PSD argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    command_line: str, optional
        The command line which you are passing. Default None
    """
    return add_dynamic_argparse(
        existing_namespace, "--*_psd", command_line=command_line
    )


def add_dynamic_calibration_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic calibration argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    command_line: str, optional
        The command line which you are passing. Default None
    """
    return add_dynamic_argparse(
        existing_namespace, "--*_calibration", example="--{}_calibration",
        command_line=command_line
    )
