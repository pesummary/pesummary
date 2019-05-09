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


class DictionaryAction(argparse.Action):
    """Class to extend the argparse.Action to handle dictionaries as input
    """
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(DictionaryAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=nargs,
            const=const, default=default, type=str, choices=choices,
            required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        bool = [True if ':' in value else False for value in values]
        if all(i is True for i in bool):
            setattr(namespace, self.dest, {})
        elif all(i is False for i in bool):
            setattr(namespace, self.dest, [])
        else:
            raise ValueError("Did not understand input")

        items = getattr(namespace, self.dest)
        items = copy.copy(items)
        for value in values:
            value = value.split(':')
            if len(value) == 2:
                if value[0] in items.keys():
                    if not isinstance(items[value[0]], list):
                        items[value[0]] = [items[value[0]]]
                    items[value[0]].append(value[1])
                else:
                    items[value[0]] = value[1]
            elif len(value) == 1:
                items.append(value[0])
            else:
                raise ValueError("Did not understand input")
        setattr(namespace, self.dest, items)


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
    gw_group.add_argument("--sensitivity", action="store_true",
                          help="generate sky sensitivities for HL, HLV",
                          default=False)
    gw_group.add_argument("--gracedb", dest="gracedb",
                          help="gracedb of the event", default=None)
    gw_group.add_argument("--psd", dest="psd", action=DictionaryAction,
                          help="psd files used", nargs='+', default=None)
    gw_group.add_argument("--calibration", dest="calibration",
                          help="files for the calibration envelope",
                          nargs="+", action=DictionaryAction, default=None)
    gw_group.add_argument("--gw", action="store_true",
                          help="run with the gravitational wave pipeline",
                          default=False)
    return gw_group
