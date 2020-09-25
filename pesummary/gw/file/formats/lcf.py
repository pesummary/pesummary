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

from gwpy.timeseries import TimeSeries
from glue.lal import Cache


def read_lcf(path, channel=None, **kwargs):
    """Grab the data stored in a .lcf file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    channel: str
        channel to use when reading in the data
    """
    if channel is None:
        raise ValueError("Please provider a channel for reading the data")

    with open(path, "r") as f:
        data = Cache.fromfile(f)

    return TimeSeries.read(data, channel, **kwargs)
