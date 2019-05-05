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


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default=None)
    parser.add_argument("-b", "--baseurl", dest="baseurl",
                        help="make the page at this url", metavar="DIR",
                        default=None)
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Posterior samples hdf5 file", nargs='+',
                        default=None)
    parser.add_argument("-c", "--config", dest="config",
                        help=("configuration file associcated with "
                              "each samples file."),
                        nargs='+', default=None)
    parser.add_argument("--email", action="store",
                        help=("send an e-mail to the given address with a link "
                              "to the finished page."), default=None)
    parser.add_argument("--dump", action="store_true",
                        help="dump all information onto a single html page",
                        default=False)
    parser.add_argument("--add_to_existing", action="store_true",
                        help="add new results to an existing html page",
                        default=False)
    parser.add_argument("-e", "--existing_webdir", dest="existing",
                        help="web directory of existing output", default=None)
    parser.add_argument("-i", "--inj_file", dest="inj_file",
                        help="path to injetcion file", nargs='+', default=None)
    parser.add_argument("--user", dest="user", help=argparse.SUPPRESS,
                        default="albert.einstein")
    parser.add_argument("--labels", dest="labels",
                        help="labels used to distinguish runs", nargs='+',
                        default=None)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print useful information for debugging purposes")
    parser.add_argument("--save_to_hdf5", action="store_true",
                        help="save the meta file in hdf5 format", default=False)
    return parser
