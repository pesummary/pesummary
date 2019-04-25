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


def insert_gwspecific_option_group(parser):
    """Add gravitational wave related options to the optparser object

    Parameters
    ----------
    parser: object
        OptionParser instance.
    """
    gw_group = parser.add_argument_group(
        "Options specific for gravitational wave results files")

    gw_group.add_argument("-a", "--approximant", dest="approximant",
                          help=("waveform approximant used to generate "
                                "samples"), nargs='+', default=None)
    gw_group.add_argument("-c", "--config", dest="config",
                          help=("configuration file associcated with "
                                "each samples file."),
                          nargs='+', default=None)
    gw_group.add_argument("--sensitivity", action="store_true",
                          help="generate sky sensitivities for HL, HLV",
                          default=False)
    gw_group.add_argument("--gracedb", dest="gracedb",
                          help="gracedb of the event", default=None)
    gw_group.add_argument("--psd", dest="psd",
                          help="psd files used", nargs='+', default=None)
    gw_group.add_argument("--calibration", dest="calibration",
                          help="files for the calibration envelope",
                          nargs="+", default=None)
    gw_group.add_argument("--gw", action="store_true",
                          help="run with the gravitational wave pipeline",
                          default=False)
    return gw_group
