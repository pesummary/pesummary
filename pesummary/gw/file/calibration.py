# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from pesummary import conf
from pesummary.utils.utils import logger, check_file_exists_and_rename
from pesummary.utils.dict import Dict

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class CalibrationDict(Dict):
    """Class to handle a dictionary of calibration data

    Parameters
    ----------
    detectors: list
        list of detectors
    data: nd list
        list of calibration samples for each detector. Each of the columns
        should represent Frequency, Median Mag, Phase (Rad), -1 Sigma Mag,
        -1 Sigma Phase, +1 Sigma Mag, +1 Sigma Phase

    Attributes
    ----------
    detectors: list
        list of detectors stored in the dictionary

    Methods
    -------
    plot:
        Generate a plot based on the calibration samples stored
    """
    def __init__(self, *args):
        _columns = [
            "frequencies", "magnitude", "phase", "magnitude_lower",
            "phase_lower", "magnitude_upper", "phase_upper"
        ]
        super(CalibrationDict, self).__init__(
            *args, value_class=Calibration, value_columns=_columns
        )

    @property
    def detectors(self):
        return list(self.keys())


class Calibration(np.ndarray):
    """Class to handle Calibration data
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape[1] != 7:
            raise ValueError(
                "Invalid input data. See the docs for instructions"
            )
        return obj

    @classmethod
    def read(cls, path_to_file, IFO=None, **kwargs):
        """Read in a file and initialize the Calibration class

        Parameters
        ----------
        path_to_file: str
            the path to the file you wish to load
        IFO: str, optional
            name of the IFO which relates to the input file
        **kwargs: dict
            all kwargs are passed to the np.genfromtxt method
        """
        try:
            f = np.genfromtxt(path_to_file, **kwargs)
            return cls(f)
        except Exception:
            raise

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
        header = [
            "Frequency", "Median Mag", "Phase (Rad)", "-1 Sigma Mag",
            "-1 Sigma Phase", "+1 Sigma Mag", "+1 Sigma Phase"
        ]
        np.savetxt(
            file_name, self, delimiter=delimiter, comments=comments,
            header=delimiter.join(header)
        )

    def __array_finalize__(self, obj):
        if obj is None:
            return
