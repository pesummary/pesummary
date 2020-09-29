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

from gwpy.timeseries import TimeSeries
from pesummary.utils.decorators import docstring_subfunction


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
        obj = super(StrainData, cls).read(*args, **kwargs)
        return cls(obj, IFO=IFO)

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
