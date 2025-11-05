# Licensed under an MIT style license -- see LICENSE.md

from scipy.stats import gaussian_kde
import numpy as np
from pesummary.core.plots.palette import color_palette
from pesummary.core.plots.seaborn import SEABORN
from .kde import _BaseKDE
if SEABORN:
    from seaborn import categorical
    from seaborn import _base
    from seaborn._stats.density import KDE as _DensityKDE
else:
    class _DensityKDE(object):
        """Dummy class for the KDE class to inherit
        """

    class _base(object):
        class HueMapping(object):
            """Dummy class for the HueMapping to inherit
            """

    class categorical(object):
        class _CategoricalPlotter(object):
            """Dummy class for the _CategoricalPlotter to inherit
            """

        def violinplot(*args, **kwargs):
            """Dummy function to call
            """
            raise ValueError("Unable to produce violinplot with 'seaborn'")


__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class DensityKDE(_BaseKDE, _DensityKDE):
    """Extension of the `seaborn._stats.density.KDE` to allow for custom
    kde_kernel

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn._stats.density.KDE` class
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    **kwargs: dict
        all kwargs passed to the `seaborn._stats.density.KDE` class
    """
    def _fit(self, fit_data, orient, **kwargs):
        return super()._fit(fit_data[orient], **kwargs)


class HueMapping(_base.HueMapping):
    ind = {"left": 0, "right": 0, "num": 0}
    _palette_dict = {"left": False, "right": False}
    _lookup_table = {"left": None, "right": None}

    def _lookup_single(self, key):
        # check for different colored left and right violins
        if colorlist is not None and self.palette is None:
            color = colorlist[self.ind["num"]]
            self.ind["num"] += 1
            return color
        if key not in self._palette_dict.keys():
            return super()._lookup_single(key)
        if self._palette_dict[key]:
            color = self._lookup_table[key][self.ind[key]]
        else:
            color = self.lookup_table[key]
        self.ind[key] += 1
        return color

    def categorical_mapping(self, data, palette, order):
        levels, lookup_table = super().categorical_mapping(data, palette, order)
        if isinstance(palette, dict):
            for key in ["left", "right"]:
                if key in palette:
                    if "color:" in palette[key]:
                        _color = palette[key].replace(" ", "").split(":")[1]
                        lookup_table[key] = _color
                    else:
                        self._palette_dict[key] = True
                        self._lookup_table[key] = color_palette(palette[key], n_colors=10)
                        _color = color_palette(palette[key], n_colors=1)[0]
                        lookup_table[key] = _color
        return levels, lookup_table


class _CategoricalPlotter(categorical._CategoricalPlotter):
    def plot_violins(self, *args, **kwargs):
        _kwargs = kwargs.copy()
        kde_kws = _kwargs["kde_kws"]
        kde_kws.update({"kde_kernel": KDE, "kde_kwargs": KDE_kwargs})
        kde_kws.pop("gridsize", None)
        kde_kws.pop("bw_adjust", None)
        _kwargs["kde_kws"] = kde_kws
        return super().plot_violins(*args, **_kwargs)


categorical._CategoricalPlotter = _CategoricalPlotter
categorical.KDE = DensityKDE
_base.HueMapping = HueMapping


def violinplot(
    *args, kde_kernel=gaussian_kde, kde_kwargs={}, inj=None, colors=None,
    **kwargs
):
    """Extension of the seaborn.categorical.violinplot function to allow for
    a custom kde_kernel and associated kwargs.

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn.categorical.violinplot` function
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    inj: float, optional
        injected value. Currently ignored, but kept for backwards compatibility
    colors: list, optional
        list of colors to use for each violin. Default None
    **kwargs: dict
        all kwargs passed to the `seaborn.categorical.violinplot` class
    """
    global KDE
    global KDE_kwargs
    global colorlist
    KDE = kde_kernel
    KDE_kwargs = kde_kwargs
    colorlist = colors
    return categorical.violinplot(*args, **kwargs)


def split_dataframe(
    left, right, labels, left_label="left", right_label="right",
    weights_left=None, weights_right=None
):
    """Generate a pandas DataFrame containing two sets of distributions -- one
    set for the left hand side of the violins, and one set for the right hand
    side of the violins

    Parameters
    ----------
    left: np.ndarray
        array of samples representing the left hand side of the violins
    right: np.ndarray
        array of samples representing the right hand side of the violins
    labels: np.array
        array containing the label associated with each violin
    """
    import pandas

    nviolin = len(left)
    if len(left) != len(right) != len(labels):
        raise ValueError("Please ensure that 'left' == 'right' == 'labels'")
    _left_label = np.array(
        [[left_label] * len(sample) for sample in left], dtype="object"
    )
    _right_label = np.array(
        [[right_label] * len(sample) for sample in right], dtype="object"
    )
    _labels = [
        [label] * (len(left[num]) + len(right[num])) for num, label in
        enumerate(labels)
    ]
    labels = [x for y in _labels for x in y]
    dataframe = [
        x for y in [[i, j] for i, j in zip(left, right)] for x in y
    ]
    dataframe = [x for y in dataframe for x in y]
    sides = [
        x for y in [[i, j] for i, j in zip(_left_label, _right_label)] for x in
        y
    ]
    sides = [x for y in sides for x in y]
    df = pandas.DataFrame(
        data={"data": dataframe, "side": sides, "label": labels}
    )
    if all(kwarg is None for kwarg in [weights_left, weights_right]):
        return df

    left_inds = df["side"][df["side"] == left_label].index
    right_inds = df["side"][df["side"] == right_label].index
    if weights_left is not None and weights_right is None:
        weights_right = [np.ones(len(right[num])) for num in range(nviolin)]
    elif weights_left is None and weights_right is not None:
        weights_left = [np.ones(len(left[num])) for num in range(nviolin)]
    if any(len(kwarg) != nviolin for kwarg in [weights_left, weights_right]):
        raise ValueError("help")

    weights = [
        x for y in [[i, j] for i, j in zip(weights_left, weights_right)] for x in y
    ]
    weights = [x for y in weights for x in y]
    df["weights"] = weights
    return df
