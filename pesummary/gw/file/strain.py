# Licensed under an MIT style license -- see LICENSE.md

import pathlib
from gwpy.timeseries import TimeSeries
from pesummary.utils.utils import logger
from pesummary.utils.dict import Dict
from pesummary.utils.decorators import docstring_subfunction

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class StrainDataDict(Dict):
    """Class to store multiple StrainData objects from different IFOs

    Parameters
    ----------
    data: dict
        dict keyed by IFO and values StrainData objects

    Examples
    --------
    >>> from pesummary.gw.file.strain import StrainDataDict
    >>> data = {
    ...     "H1": "./H-H1_LOSC_4_V2-1126257414-4096.gwf",
    ...     "L1": "./L-L1_LOSC_4_V2-1126257414-4096.gwf"
    ... }
    >>> channels = {"H1": "H1:LOSC-STRAIN", "L1": "L1:LOSC-STRAIN"}
    >>> strain = StrainDataDict.read(data, channels=channels)
    """
    def __init__(self, *args):
        super(StrainDataDict, self).__init__(*args, value_class=StrainData)

    @classmethod
    def read(cls, data, channels={}):
        strain_data = {}
        if not len(channels):
            _data = {}
            for key in data.keys():
                if ":" in key:
                    try:
                        IFO, _ = key.split(":")
                        channels[IFO] = key
                        _data[IFO] = data[key]
                        logger.debug(
                            "Found ':' in '{}'. Assuming '{}' is the IFO and "
                            "'{}' is the channel".format(key, IFO, key)
                        )
                    except ValueError:
                        _data[key] = data[key]
            data = _data
        if not all(IFO in channels.keys() for IFO in data.keys()):
            raise ValueError("Please provide a channel for each IFO")
        for IFO in data.keys():
            strain_data[IFO] = StrainData.read(data[IFO], channels[IFO], IFO=IFO)
        return cls(strain_data)

    @property
    def detectors(self):
        return list(self.keys())


class StrainData(TimeSeries):
    """Class to extend the gwpy.timeseries.TimeSeries plotting functions to
    include the pesummary plots

    Parameters
    ----------
    IFO: str, optional
        IFO for which the strain data corresponds too. This is used to determine
        the color on plots. Default 'H1'

    Attributes
    ----------
    gwpy: gwpy.timeseries.TimeSeries
        original gwpy TimeSeries object
    IFO: str
        IFO for which the strain data corresponds too
    strain_dict: dict
        dictionary of strain data

    Methods
    -------
    plot:
        Generate a plot based on the stored data
    """
    def __new__(cls, *args, IFO="H1", **kwargs):
        new = super(StrainData, cls).__new__(cls, *args, **kwargs)
        new.gwpy = TimeSeries(*args, **kwargs)
        new.IFO = IFO
        new.strain_dict = {new.IFO: new}
        return new

    def __array_finalize__(self, obj):
        super(StrainData, self).__array_finalize__(obj)
        try:
            self.gwpy = getattr(obj, 'gwpy')
            self.IFO = getattr(obj, 'IFO')
            self.strain_dict = getattr(obj, 'strain_dict')
        except (TypeError, AttributeError):
            pass

    @classmethod
    def read(cls, *args, IFO="H1", **kwargs):
        from pesummary.gw.file.formats.base_read import GWRead
        if len(args) and isinstance(args[0], str):
            if GWRead.extension_from_path(args[0]) == "pickle":
                try:
                    from pesummary.gw.file.formats.bilby import Bilby
                    obj = Bilby._timeseries_from_bilby_pickle(args[0])
                    return StrainDataDict(obj)
                except Exception as e:
                    pass
            elif GWRead.extension_from_path(args[0]) == "lcf":
                from glue.lal import Cache
                with open(args[0], "r") as f:
                    data = Cache.fromfile(f)
                args[0] = data
        if len(args) and isinstance(args[0], pathlib.PosixPath):
            args = list(args)
            args[0] = str(args[0])
        obj = super(StrainData, cls).read(*args, **kwargs)
        return cls(obj, IFO=IFO)

    @classmethod
    def fetch_open_frame(cls, event, **kwargs):
        """Fetch open frame files for a given event

        Parameters
        ----------
        sampling_rate: int, optional
            sampling rate of strain data you wish to download. Default 16384
        format: str, optional
            format of strain data you wish to download. Default "gwf"
        duration: int, optional
            duration of strain data you wish to download. Default 32
        IFO: str, optional
            detector strain data you wish to download. Default 'L1'
        **kwargs: dict, optional
            all additional kwargs passed to StrainData.read
        """
        from ..fetch import fetch_open_strain
        return fetch_open_strain(event, read_file=True, **kwargs)

    @classmethod
    def fetch_open_data(cls, *args, **kwargs):
        obj = super(StrainData, cls).fetch_open_data(*args, **kwargs)
        return cls(obj, IFO=args[0])

    @property
    def plotting_map(self):
        return {
            "td": self._time_domain_plot,
            "fd": self._frequency_domain_plot,
            "omegascan": self._omega_scan_plot,
            "spectrogram": self._spectrogram_plot
        }

    @property
    def available_plots(self):
        return list(self.plotting_map.keys())

    @docstring_subfunction([
        'pesummary.gw.plots.detchar.spectrogram',
        'pesummary.gw.plots.detchar.omegascan',
        'pesummary.gw.plots.detchar.time_domain_strain_data',
        'pesummary.gw.plots.detchar.frequency_domain_strain_data'
    ])
    def plot(self, *args, type="td", **kwargs):
        """Generate a plot displaying the gravitational wave strain data

        Parameters
        ----------
        *args: tuple
            all arguments are passed to the plotting function
        type: str
            name of the plot you wish to make
        **kwargs: dict
            all additional kwargs are passed to the plotting function
        """
        if type not in self.plotting_map.keys():
            raise NotImplementedError(
                "The {} method is not currently implemented. The allowed "
                "plotting methods are {}".format(
                    type, ", ".join(self.available_plots)
                )
            )
        return self.plotting_map[type](self.strain_dict, *args, **kwargs)

    def _time_domain_plot(self, *args, **kwargs):
        """Plot the strain data in the time domain

        Parameters
        ----------
        *args: tuple
            all args passed to the time_domain_strain_data function
        **kwargs: dict
            all kwargs passed to the time_domain_strain_data function
        """
        from pesummary.gw.plots.detchar import time_domain_strain_data

        return time_domain_strain_data(*args, **kwargs)[self.IFO]

    def _frequency_domain_plot(self, *args, **kwargs):
        """Plot the strain data in the frequency domain

        Parameters
        ----------
        *args: tuple
            all args passed to the frequency_domain_strain_data function
        **kwargs: dict
            all kwargs passed to the frequency_domain_strain_data function
        """
        from pesummary.gw.plots.detchar import frequency_domain_strain_data

        return frequency_domain_strain_data(*args, **kwargs)[self.IFO]

    def _omega_scan_plot(self, *args, **kwargs):
        """Plot an omegascan of the strain data

        Parameters
        ----------
        *args: tuple
            all args passed to the omegascan function
        **kwargs: dict
            all kwargs passed to the omegascan function
        """
        from pesummary.gw.plots.detchar import omegascan

        return omegascan(*args, **kwargs)[self.IFO]

    def _spectrogram_plot(self, *args, **kwargs):
        """Plot the spectrogram of the strain data

        Parameters
        ----------
        *args: tuple
            all args passed to the spectrogram function
        **kwargs: dict
            all kwargs passed to the spectrogram function
        """
        from pesummary.gw.plots.detchar import spectrogram

        return spectrogram(*args, **kwargs)[self.IFO]
