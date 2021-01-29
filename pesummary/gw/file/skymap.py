# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.utils import check_file_exists_and_rename, Empty
from pesummary import conf
from pesummary.utils.dict import Dict

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class SkyMapDict(Dict):
    """Class to handle a dictionary of skymaps

    Parameters
    ----------
    labels: list
        list of labels for each skymap
    data: nd list
        list of skymap probabilities for each analysis
    **kwargs: dict
        All other kwargs are turned into properties of the class. Key
        is the name of the property

    Attributes
    ----------
    labels: list
        list of labels stored in the dictionary

    Methods
    -------
    plot:
        Generate a plot based on the skymap samples stored

    Examples
    --------
    >>> skymap_1 = SkyMap.from_fits("skymap.fits")
    >>> skymap_2 = SkyMap.from_fits("skymap_2.fits")
    >>> skymap_dict = SkyMapDict(
    ...     ["one", "two"], [skymap_1, skymap_2]
    ... )
    >>> skymap_dict = SkyMapDict(
    ...     {"one": skymap_1, "two": skymap_2}
    ... )
    """
    def __init__(self, *args, **kwargs):
        super(SkyMapDict, self).__init__(*args, value_class=Empty, **kwargs)

    @property
    def labels(self):
        return list(self.keys())

    def plot(self, labels="all", colors=None, show_probability_map=False, **kwargs):
        """Generate a plot to compare the skymaps stored in the SkyMapDict

        Parameters
        ----------
        labels: list, optional
            list of analyses you wish to compare. Default all.
        **kwargs: dict
            all additional kwargs are passed to
            pesummary.gw.plots.plot._
        """
        from pesummary.gw.plots.plot import (
            _ligo_skymap_comparion_plot_from_array
        )

        _labels = self.labels
        if labels != "all" and isinstance(labels, list):
            _labels = []
            for label in labels:
                if label not in self.labels:
                    raise ValueError(
                        "No skymap for '{}' is stored in the dictionary. "
                        "The list of available analyses are: {}".format(
                            label, ", ".join(self.labels)
                        )
                    )
                _labels.append(label)
        skymaps = [self[key] for key in _labels]
        if colors is None:
            colors = list(conf.colorcycle)

        if show_probability_map:
            show_probability_map = _labels.index(show_probability_map)

        return _ligo_skymap_comparion_plot_from_array(
            skymaps, colors, _labels, show_probability_map=show_probability_map,
            **kwargs
        )


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
