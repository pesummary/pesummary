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

import numpy as np
from pesummary.utils.utils import logger, get_matplotlib_backend
import matplotlib
matplotlib.use(get_matplotlib_backend())
import matplotlib.pyplot as plt


def scatter_plot(
    parameters, sample_dict, latex_labels, colors=None, xerr=None, yerr=None
):
    """Produce a plot which shows a population of runs over a certain parameter
    space. If errors are given, then plot error bars.

    Parameters
    ----------
    parameters: list
        names of the parameters that you wish to plot
    sample_dict: dict
        nested dictionary storing the median values for each parameter for each
        run. For example: x = {'one': {'m': 10, 'n': 20}}
    latex_labels: dictionary
        dictionary of latex labels
    colors: list
        list of colors that you wish to use to distinguish the different runs
    xerr: dict
        same structure as sample_dict, but dictionary storing error in x
    yerr: dict
        same structure as sample_dict, but dictionary storing error in y
    """
    fig = plt.figure()
    runs = list(sample_dict.keys())

    xx, yy, xxerr, yyerr = {}, {}, {}, {}
    for analysis in runs:
        if all(i in sample_dict[analysis].keys() for i in parameters):
            xx[analysis] = sample_dict[analysis][parameters[0]]
            yy[analysis] = sample_dict[analysis][parameters[1]]
        else:
            logger.warn(
                "'{}' does not include samples for '{}' and/or '{}'. This "
                "analysis will not be added to the plot".format(
                    analysis, parameters[0], parameters[1]
                )
            )
        if xerr is not None and parameters[0] in xerr[analysis].keys():
            xxerr[analysis] = xerr[analysis][parameters[0]]
        if yerr is not None and parameters[1] in yerr[analysis].keys():
            yyerr[analysis] = yerr[analysis][parameters[1]]

    keys = xx.keys()
    xdata = [xx[key] for key in keys]
    ydata = [yy[key] for key in keys]
    xerrdata = np.array([xxerr[key] if key in xxerr.keys() else [0, 0] for key in keys])
    yerrdata = np.array([yyerr[key] if key in yyerr.keys() else [0, 0] for key in keys])

    if xerr is not None or yerr is not None:
        plt.errorbar(
            xdata, ydata, color=colors, xerr=xerrdata.T, yerr=yerrdata.T, linestyle=" "
        )
    else:
        plt.scatter(xdata, ydata, color=colors)
    plt.xlabel(latex_labels[parameters[0]], fontsize=16)
    plt.ylabel(latex_labels[parameters[1]], fontsize=16)
    plt.tight_layout()
    return fig
