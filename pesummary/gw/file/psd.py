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
    to_pycbc:
        Convert dictionary of PSD objects to a dictionary of
        pycbc.frequencyseries objects objects

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

    @classmethod
    def read(cls, files=None, detectors=None, common_string=None):
        """Initiate PSDDict with a set of PSD files

        Parameters
        ----------
        files: list/dict, optional
            Either a list of files or a dictionary of files to read.
            If a list of files are provided, a list of corresponding
            detectors must also be provided
        common_string: str, optional
            Common string for PSD files. The string must be formattable and
            take one argument which is the detector. For example
            common_string='./{}_psd.dat'. Used if files is not provided
        detectors: list, optional
            List of detectors to use when loading files. Used if files
            if not provided or if files is a list or if common_string is
            provided
        """
        if files is not None:
            if isinstance(files, list) and detectors is not None:
                if len(detectors) != len(files):
                    raise ValueError(
                        "Please provide a detector for each file"
                    )
                files = {det: ff for det, ff in zip(detectors, files)}
            elif isinstance(files, dict):
                pass
            else:
                raise ValueError(
                    "Please provide either a dictionary of files, or a list "
                    "files and a list of detectors for which they correspond."
                )
        elif common_string is not None and detectors is not None:
            files = {det: common_string.format(det) for det in detectors}
        else:
            raise ValueError(
                "Please provide either a list of files to read or "
                "a common string and a list of detectors to load."
            )
        psd = {}
        for key, item in files.items():
            psd[key] = PSD.read(item, IFO=key)
        return PSDDict(psd)

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

    def to_pycbc(self, *args, **kwargs):
        """Transform dictionary to pycbc.frequencyseries objects

        Parameters
        ----------
        *args: tuple
            all args passed to PSD.to_pycbc()
        **kwargs: dict, optional
            all kwargs passed to PSD.to_pycbc()
        """
        psd = {}
        for key, item in self.items():
            psd[key] = item.to_pycbc(*args, **kwargs)
        return PSDDict(psd)

    def interpolate(self, low_freq_cutoff, delta_f):
        """Interpolate a dictionary of PSDs to a new delta_f

        Parameters
        ----------
        low_freq_cutoff: float
            Frequencies below this value are set to zero.
        delta_f : float, optional
            Frequency resolution of the frequency series in Hertz.
        """
        psd = {}
        for key, item in self.items():
            psd[key] = item.interpolate(low_freq_cutoff, delta_f)
        return PSDDict(psd)


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
        obj.frequencies = cls.frequencies(obj)
        return obj

    @property
    def low_frequency(self):
        return self.frequencies[0]

    @staticmethod
    def delta_f(array):
        return array.T[0][1] - array.T[0][0]

    @staticmethod
    def f_high(array):
        return array.T[0][-1]

    @staticmethod
    def frequencies(array):
        return array.T[0]

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
        self.frequencies = getattr(obj, "frequencies", None)

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
            logger.warning(msg)
        if length is None:
            length = int(f_high / delta_f) + 1
        pycbc_psd = from_numpy_arrays(
            self.T[0], self.T[1], length, delta_f, low_freq_cutoff
        )
        return pycbc_psd

    def interpolate(self, low_freq_cutoff, delta_f):
        """Interpolate PSD to a new delta_f

        Parameters
        ----------
        low_freq_cutoff: float
            Frequencies below this value are set to zero.
        delta_f : float, optional
            Frequency resolution of the frequency series in Hertz.
        """
        from pesummary.gw.pycbc import interpolate_psd
        psd = interpolate_psd(self.copy(), low_freq_cutoff, delta_f)
        frequencies, strains = psd.sample_frequencies, psd
        inds = np.where(frequencies >= low_freq_cutoff)
        return PSD(np.vstack([frequencies[inds], strains[inds]]).T)
