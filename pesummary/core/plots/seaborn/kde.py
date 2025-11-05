# Licensed under an MIT style license -- see LICENSE.md

from scipy import stats
from pesummary.core.plots.seaborn import SEABORN
if SEABORN:
    from seaborn._statistics import KDE as _StatisticsKDE
    from seaborn import distributions
else:
    class _StatisticsKDE(object):
        """Dummy class for the KDE class to inherit
        """

    class distributions(object):
        class _DistributionPlotter(object):
            """Dummy class for the _DistributionPlotter class to inherit
            """

        def kdeplot(*args, **kwargs):
            """Dummy function to call
            """
            raise ValueError("Unable to produce kdeplot with 'seaborn'")


__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>", "Seaborn authors"]


class _BaseKDE(object):
    """Extension of the `seaborn._statistics.KDE` to allow for custom
    kde_kernel

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn._statistics.KDE` class
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    **kwargs: dict
        all kwargs passed to the `seaborn._statistics.KDE` class
    """
    def __init__(self, *args, kde_kernel=stats.gaussian_kde, kde_kwargs={}, **kwargs):
        _kwargs = kwargs.copy()
        for key, item in kwargs.items():
            if key == "bw_method" or key == "bw_adjust":
                setattr(self, key, item)
                _kwargs.pop(key, None)
            elif key not in ["gridsize", "cut", "clip", "cumulative"]:
                kde_kwargs[key] = item
                _kwargs.pop(key, None)
        super().__init__(*args, **_kwargs)
        if kde_kernel is None:
            kde_kernel = stats.gaussian_kde
        self._kde_kernel = kde_kernel
        self._kde_kwargs = kde_kwargs

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = self._kde_kwargs
        fit_kws["bw_method"] = self.bw_method
        if weights is not None:
            fit_kws["weights"] = weights

        kde = self._kde_kernel(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)
        return kde


class StatisticsKDE(_BaseKDE, _StatisticsKDE):
    """Extension of the `seaborn._statistics.KDE` to allow for custom
    kde_kernel

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn._statistics.KDE` class
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    **kwargs: dict
        all kwargs passed to the `seaborn._statistics.KDE` class
    """


class _DistributionPlotter(distributions._DistributionPlotter):
    def _compute_univariate_density(
        self, data_variable, common_norm, common_grid, estimate_kws,
        log_scale, **kwargs
    ):
        estimate_kws.update({"kde_kernel": KDE, "kde_kwargs": KDE_kwargs})
        return super()._compute_univariate_density(
            data_variable, common_norm, common_grid, estimate_kws,
            log_scale, **kwargs
        )


distributions.KDE = StatisticsKDE
distributions._DistributionPlotter = _DistributionPlotter


def kdeplot(*args, kde_kernel=stats.gaussian_kde, kde_kwargs={}, **kwargs):
    """Extension of the seaborn.distributions.kdeplot function to allow for
    a custom kde_kernel and associated kwargs.

    Parameters
    ----------
    *args: tuple
        all args passed to the `seaborn.distributions.kdeplot` function
    kde_kernel: func, optional
        kernel you wish to use to evaluate the KDE. Default
        scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        optional kwargs to be passed to the kde_kernel. Default {}
    **kwargs: dict
        all kwargs passed to the `seaborn.distributions.kdeplot` class
    """
    global KDE
    global KDE_kwargs
    KDE = kde_kernel
    KDE_kwargs = kde_kwargs
    return distributions.kdeplot(*args, **kwargs)
