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

import os
import numpy as np
from pesummary import conf
from pesummary.utils.utils import logger, check_file_exists_and_rename
from pesummary.utils.dict import Dict


class PSDDict(Dict):
    """Class to handle a dictionary of PSDs

    Parameters
    ----------
    detectors: list
        list of detectors
    data: nd list
        list of psd samples for each detector. First column is frequencies,
        second column is strains

    Attributes
    ----------
    detectors: list
        list of detectors stored in the dictionary

    Methods
    -------
    plot:
        Generate a plot based on the psd samples stored

    Examples
    --------
    >>> from pesummary.gw.file.psd import PSDDict
    >>> detectors = ["H1", "V1"]
    >>> psd_data = [
    ...     [[0.00000e+00, 2.50000e-01],
    ...      [1.25000e-01, 2.50000e-01],
    ...      [2.50000e-01, 2.50000e-01]],
    ...     [[0.00000e+00, 2.50000e-01],
    ...      [1.25000e-01, 2.50000e-01],
    ...      [2.50000e-01, 2.50000e-01]]
    ... ]
    >>> psd_dict = PSDDict(detectors, psd_data)
    >>> psd_data = {
    ...     "H1": [[0.00000e+00, 2.50000e-01],
    ...            [1.25000e-01, 2.50000e-01],
    ...            [2.50000e-01, 2.50000e-01]],
    ...     "V1": [[0.00000e+00, 2.50000e-01],
    ...            [1.25000e-01, 2.50000e-01],
    ...            [2.50000e-01, 2.50000e-01]]
    ... }
    >>> psd_dict = PSDDict(psd_data)
    """
    def __init__(self, *args):
        super(PSDDict, self).__init__(
            *args, value_class=PSD, value_columns=["frequencies", "strains"]
        )

    @property
    def detectors(self):
        return list(self.keys())

    def plot(self, **kwargs):
        """Generate a plot to display the PSD data stored in PSDDict

        Parameters
        ----------
        **kwargs: dict
            all additional kwargs are passed to
            pesummary.gw.plots.plot._psd_plot
        """
        from pesummary.gw.plots.plot import _psd_plot

        _detectors = self.detectors
        frequencies = [self[IFO].frequencies for IFO in _detectors]
        strains = [self[IFO].strains for IFO in _detectors]
        return _psd_plot(frequencies, strains, labels=_detectors, **kwargs)


class PSD(np.ndarray):
    """Class to handle PSD data
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape[1] != 2:
            raise ValueError(
                "Invalid input data. See the docs for instructions"
            )
        return obj

    def save_to_file(self, file_name, comments="#", delimiter=conf.delimiter):
        """Save the calibration data to file

        Parameters
        ----------
        file_name: str
            name of the file name that you wish to use
        comments: str, optional
            String that will be prepended to the header and footer strings, to
            mark them as comments. Default is '#'.
        delimiter: str, optional
            String or character separating columns.
        """
        check_file_exists_and_rename(file_name)
        header = ["Frequency", "Strain"]
        np.savetxt(
            file_name, self, delimiter=delimiter, comments=comments,
            header=delimiter.join(header)
        )

    def __array_finalize__(self, obj):
        if obj is None:
            return
