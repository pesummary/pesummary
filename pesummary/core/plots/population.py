# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.core.plots.figure import figure
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
    fig, ax = figure(gca=True)
    runs = list(sample_dict.keys())

    xx, yy, xxerr, yyerr = {}, {}, {}, {}
    for analysis in runs:
        if all(i in sample_dict[analysis].keys() for i in parameters):
            xx[analysis] = sample_dict[analysis][parameters[0]]
            yy[analysis] = sample_dict[analysis][parameters[1]]
        else:
            logger.warning(
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
        ax.errorbar(
            xdata, ydata, color=colors, xerr=xerrdata.T, yerr=yerrdata.T, linestyle=" "
        )
    else:
        ax.scatter(xdata, ydata, color=colors)
    ax.set_xlabel(latex_labels[parameters[0]], fontsize=16)
    ax.set_ylabel(latex_labels[parameters[1]], fontsize=16)
    fig.tight_layout()
    return fig
