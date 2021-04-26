# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from pesummary import conf
from pesummary.utils.utils import logger, check_file_exists_and_rename
from pesummary.utils.dict import Dict

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        obj.delta_f = cls.delta_f(obj)
        obj.f_high = cls.f_high(obj)
        return obj

    @staticmethod
    def delta_f(array):
        return array.T[0][1] - array.T[0][0]

    @staticmethod
    def f_high(array):
        return array.T[0][-1]

    @classmethod
    def read(cls, path_to_file, **kwargs):
        """Read in a file and initialize the PSD class

        Parameters
        ----------
        path_to_file: str
            the path to the file you wish to load
        **kwargs: dict
            all kwargs are passed to the read methods
        """
        from pesummary.core.file.formats.base_read import Read

        mapping = {
            "dat": PSD.read_from_dat,
            "txt": PSD.read_from_dat,
            "xml": PSD.read_from_xml,
        }
        if not os.path.isfile(path_to_file):
            raise FileNotFoundError(
                "The file '{}' does not exist".format(path_to_file)
            )
        extension = Read.extension_from_path(path_to_file)
        if ".xml.gz" in path_to_file:
            return cls(mapping["xml"](path_to_file, **kwargs))
        elif extension not in mapping.keys():
            raise NotImplementedError(
                "Unable to read in a PSD with format '{}'. The allowed formats "
                "are: {}".format(extension, ", ".join(list(mapping.keys())))
            )
        return cls(mapping[extension](path_to_file, **kwargs))

    @staticmethod
    def read_from_dat(path_to_file, IFO=None, **kwargs):
        """Read in a dat file and return a numpy array containing the data

        Parameters
        ----------
        path_to_file: str
            the path to the file you wish to load
        **kwargs: dict
            all kwargs are passed to the numpy.genfromtxt method
        """
        try:
            data = np.genfromtxt(path_to_file, **kwargs)
            return data
        except ValueError:
            data = np.genfromtxt(path_to_file, skip_footer=2, **kwargs)
            return data

    @staticmethod
    def read_from_xml(path_to_file, IFO=None, **kwargs):
        """Read in an xml file and return a numpy array containing the data

        Parameters
        ----------
        path_to_file: str
            the path to the file you wish to load
        IFO: str, optional
            name of the dataset that you wish to load
        **kwargs: dict
            all kwargs are passed to the
            gwpy.frequencyseries.FrequencySeries.read method
        """
        from gwpy.frequencyseries import FrequencySeries

        data = FrequencySeries.read(path_to_file, name=IFO, **kwargs)
        frequencies = np.array(data.frequencies)
        strains = np.array(data)
        return np.vstack([frequencies, strains]).T

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
        self.delta_f = getattr(obj, "delta_f", None)
        self.f_high = getattr(obj, "f_high", None)

    def to_pycbc(
        self, low_freq_cutoff, f_high=None, length=None, delta_f=None,
        f_high_override=False
    ):
        """Convert the PSD object to an interpolated pycbc.types.FrequencySeries

        Parameters
        ----------
        length : int, optional
            Length of the frequency series in samples.
        delta_f : float, optional
            Frequency resolution of the frequency series in Herz.
        low_freq_cutoff : float, optional
            Frequencies below this value are set to zero.
        f_high_override: Bool, optional
            Override the final frequency if it is above the maximum stored.
            Default False
        """
        from pycbc.psd.read import from_numpy_arrays

        if delta_f is None:
            delta_f = self.delta_f
        if f_high is None:
            f_high = self.f_high
        elif f_high > self.f_high:
            msg = (
                "Specified value of final frequency: {} is above the maximum "
                "frequency stored: {}. ".format(f_high, self.f_high)
            )
            if f_high_override:
                msg += "Overwriting the final frequency"
                f_high = self.f_high
            else:
                msg += (
                    "This will result in an interpolation error. Either change "
                    "the final frequency specified or set the 'f_high_override' "
                    "kwarg to True"
                )
            logger.warn(msg)
        if length is None:
            length = int(f_high / delta_f) + 1
        pycbc_psd = from_numpy_arrays(
            self.T[0], self.T[1], length, delta_f, low_freq_cutoff
        )
        return pycbc_psd
