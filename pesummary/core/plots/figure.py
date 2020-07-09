# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org>
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

from pesummary.utils.decorators import try_latex_plot
from matplotlib.figure import Figure as MatplotlibFigure


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
