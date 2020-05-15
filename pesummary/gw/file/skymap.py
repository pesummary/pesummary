# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
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

import numpy as np
from pesummary.utils.utils import check_file_exists_and_rename


class SkyMap(np.ndarray):
    """Class to handle PSD data

    Parameters
    ----------
    probabilities: np.ndarray
        array of probabilities
    meta_data: dict, optional
        optional meta data associated with the skymap

    Attributes
    ----------
    meta_data: dict
        dictionary containing meta data extracted from the skymap

    Methods
    -------
    plot:
        Generate a ligo.skymap plot based on the probabilities stored
    """
    __slots__ = ["meta_data"]

    def __new__(cls, probabilities, meta_data=None):
        obj = np.asarray(probabilities).view(cls)
        obj.meta_data = meta_data
        return obj

    def __reduce__(self):
        pickled_state = super(SkyMap, self).__reduce__()
        new_state = pickled_state[2] + tuple(
            [getattr(self, i) for i in self.__slots__]
        )
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.meta_data = state[-1]
        super(SkyMap, self).__setstate__(state[0:-1])

    @classmethod
    def from_fits(cls, path, nest=None):
        """Initiate class with the path to a ligo.skymap fits file

        Parameters
        ----------
        path: str
            path to fits file you wish to load
        """
        from ligo.skymap.io.fits import read_sky_map

        skymap, meta = read_sky_map(path, nest=nest)
        return cls(skymap, meta)

    def save_to_file(self, file_name):
        """Save the calibration data to file

        Parameters
        ----------
        file_name: str
            name of the file name that you wish to use
        """
        from ligo.skymap.io.fits import write_sky_map

        check_file_exists_and_rename(file_name)
        kwargs = {}
        if self.meta_data is not None:
            kwargs = self.meta_data
        write_sky_map(file_name, self, **kwargs)

    def plot(self, **kwargs):
        """Generate a plot with ligo.skymap
        """
        from pesummary.gw.plots.plot import _ligo_skymap_plot_from_array

        return _ligo_skymap_plot_from_array(self, **kwargs)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.meta_data = getattr(obj, "meta_data", None)
