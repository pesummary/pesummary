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

try:
    from glue.lal import Cache
    GLUE = True
except ImportError:
    GLUE = False


class StrainFile(object):
    """This class handles the extracting the strain data from the `gwdata`
    input

    Parameters
    ----------
    data: dict
        dictionary of data with keys corresponding to the channel name and
        items corresponding to paths to strain files
    """
    def __init__(self, data):
        self.data = data

    def return_timeseries(self):
        """Return a dictionary of timeseries
        """
        func_map = {"lcf": self._timeseries_from_cache_file}

        timeseries_dict = {}
        for i in list(self.data.keys()):
            extension = self.data[i].split(".")[-1]
            timeseries = func_map[extension](i, self.data[i])
            if "H1" in i:
                ifo = "H1"
            elif "L1" in i:
                ifo = "L1"
            elif "V1" in i:
                ifo = "V1"
            else:
                ifo = i
            timeseries_dict[ifo] = timeseries
        return timeseries_dict

    def _timeseries_from_cache_file(self, channel, cached_file):
        """Return a time series from a cache file

        Parameters
        ----------
        channel: str
            the name of the channel for the cache file
        cached_file: str
            path to the cached file
        """
        if not GLUE:
            raise Exception("lscsoft-glue is required to read from a cached "
                            "file. Please install this package")
        with open(cached_file, "r") as f:
            data = Cache.fromfile(f)
        try:
            strain_data = TimeSeries.read(data, channel)
        except Exception as e:
            raise Exception("Failed to read in the cached file because %s" % (
                            e))
        return strain_data
