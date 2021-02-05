# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.decorators import try_latex_plot
from matplotlib.figure import Figure as MatplotlibFigure

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def figure(*args, gca=True, **kwargs):
    """Extension of the matplotlib.pyplot.figure function
    """
    from matplotlib import pyplot

    try:
        _ = kwargs.pop("FigureClass", False)
    except KeyError:
        pass

    kwargs["FigureClass"] = Figure
    fig = pyplot.figure(*args, **kwargs)
    if gca:
        return fig, fig.gca()
    return fig


def subplots(
    nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None,
    gridspec_kw=None, **fig_kw
):
    """Extension of the matplotlib.pyplot.subplots class
    """
    try:
        _ = fig_kw.pop("gca", False)
    except KeyError:
        pass

    fig_kw["gca"] = False
    fig = figure(**fig_kw)
    axs = fig.subplots(
        nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw
    )
    return fig, axs


def _close(self):
    """Extension of the matplotlib.pyplot.close function
    """
    from matplotlib.pyplot import close

    for ax in self.axes[::-1]:
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.cla()
    close(self)


class ExistingFigure(object):
    """An extension of the core matplotlib `~matplotlib.figure.Figure`
    """
    def __new__(self, fig):
        fig.close = lambda: _close(fig)
        return fig


class Figure(MatplotlibFigure):
    """An extension of the core matplotlib `~matplotlib.figure.Figure`
    """
    def __init__(self, *args, **kwargs):
        super(Figure, self).__init__(*args, **kwargs)

    def close(self):
        """Close the plot
        """
        _close(self)

    @try_latex_plot
    def savefig(self, *args, **kwargs):
        return super(Figure, self).savefig(*args, **kwargs)

    @try_latex_plot
    def tight_layout(self, *args, **kwargs):
        return super(Figure, self).tight_layout(*args, **kwargs)
