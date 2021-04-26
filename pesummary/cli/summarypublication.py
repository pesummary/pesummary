#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import pesummary
from pesummary.gw.file.read import read as GWRead
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.gw.plots import publication as pub
from pesummary.core.plots import population as pop
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.utils.utils import make_dir, logger, _check_latex_install
from pesummary.core.command_line import DictionaryAction
import argparse
import seaborn
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to generate publication quality plots given
result files"""
_check_latex_install()


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default=None)
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Posterior samples hdf5 file", nargs='+',
                        default=None)
    parser.add_argument("--labels", dest="labels",
                        help="labels used to distinguish runs", nargs='+',
                        default=None)
    parser.add_argument("--plot", dest="plot",
                        help=("name of the publication plot you wish to "
                              "produce"), default="2d_contour",
                        choices=[
                            "2d_contour", "violin", "spin_disk",
                            "population_scatter", "population_scatter_error"
                        ])
    parser.add_argument("--parameters", dest="parameters", nargs="+",
                        help=("parameters of the 2d contour plot you wish to "
                              "make"), default=None)
    parser.add_argument("--publication_kwargs",
                        help="Optional kwargs for publication plots",
                        action=DictionaryAction, nargs="+", default={})
    parser.add_argument("--palette", dest="palette",
                        help="Color palette to use to distinguish result files",
                        default="colorblind")
    parser.add_argument("--colors", dest="colors",
                        help="Colors you wish to use to distinguish result files",
                        nargs='+', default=None)
    parser.add_argument("--linestyles", dest="linestyles",
                        help=("Linestyles you wish to use to distinguish result "
                              "files"),
                        nargs='+', default=None)
    parser.add_argument("--levels", dest="levels", default=[0.9], nargs='+',
                        help="Contour levels you wish to plot", type=float)
    return parser


def draw_specific_samples(param, parameters, samples):
    """Return samples for a given parameter

    param: str
        parameter that you wish to get samples for
    parameters: nd list
        list of all parameters stored in the result file
    samples: nd list
        list of samples for each parameter
    """
    ind = [i.index(param) for i in parameters]
    return [[k[ind] for k in l] for l in samples]


def default_2d_contour_plot():
    """Return the default 2d contour plots
    """
    twod_plots = [["mass_ratio", "chi_eff"], ["mass_1", "mass_2"],
                  ["luminosity_distance", "chirp_mass_source"],
                  ["mass_1_source", "mass_2_source"],
                  ["theta_jn", "luminosity_distance"],
                  ["network_optimal_snr", "chirp_mass_source"]]
    return twod_plots


def default_violin_plot():
    """Retrn the default violin plots
    """
    violin_plots = ["chi_eff", "chi_p", "mass_ratio", "luminosity_distance"]
    return violin_plots


def read_samples(result_files):
    """Read and return a list of parameters and samples stored in the result
    files

    Parameters
    ----------
    result_files: list
        list of result files
    """
    parameters = []
    samples = []
    for i in result_files:
        try:
            f = GWRead(i)
            if isinstance(f, pesummary.gw.file.formats.pesummary.PESummary):
                parameters.append(f.parameters[0])
                samples.append(f.samples[0])
            else:
                f.generate_all_posterior_samples()
                parameters.append(f.parameters)
                samples.append(f.samples)
        except Exception:
            logger.warning(
                "Failed to read '{}'. Data will not be added to the "
                "plots".format(i)
            )
            parameters.append([None])
            samples.append([None])
    return parameters, samples


def get_colors_and_linestyles(opts):
    """Return a list of colors and linestyles
    """
    if opts.colors is not None:
        colors = opts.colors
    else:
        colors = seaborn.color_palette(
            palette=opts.palette, n_colors=len(opts.samples)
        ).as_hex()
    if opts.linestyles is not None:
        linestyles = opts.linestyles
        return colors, linestyles
    available_linestyles = ["-", "--", ":", "-."]
    linestyles = ["-"] * len(colors)
    unique_colors = np.unique(colors)
    for color in unique_colors:
        indicies = [num for num, i in enumerate(colors) if i == color]
        for idx, j in enumerate(indicies):
            linestyles[j] = available_linestyles[
                np.mod(idx, len(available_linestyles))
            ]
    return colors, linestyles


def make_2d_contour_plot(opts):
    """Make a 2d contour plot
    """
    if opts.parameters and len(opts.parameters) != 2:
        raise Exception("Please pass 2 variables that you wish to plot")
    if opts.parameters:
        default = [opts.parameters]
    else:
        default = default_2d_contour_plot()
    colors, linestyles = get_colors_and_linestyles(opts)
    parameters, samples = read_samples(opts.samples)
    for i in default:
        if not all(all(k in j for k in i) for j in parameters):
            idxs = [
                num for num, j in enumerate(parameters) if not
                all(k in j for k in i)
            ]
            files = [opts.samples[j] for j in idxs]
            logger.warning(
                "Removing {} from 2d contour plot because the parameters {} are "
                "not in the result file".format(
                    " and ".join(files), " and ".join(i)
                )
            )
            parameters = [j for num, j in enumerate(parameters) if num not in idxs]
            opts.labels = [j for num, j in enumerate(opts.labels) if num not in idxs]
            samples = [j for num, j in enumerate(samples) if num not in idxs]
        ind1 = [j.index(i[0]) for j in parameters]
        ind2 = [j.index(i[1]) for j in parameters]
        samples1 = [[k[ind1[num]] for k in l] for num, l in
                    enumerate(samples)]
        samples2 = [[k[ind2[num]] for k in l] for num, l in
                    enumerate(samples)]
        twod_samples = [[j, k] for j, k in zip(samples1, samples2)]
        gridsize = (
            opts.publication_kwargs["gridsize"] if "gridsize" in
            opts.publication_kwargs.keys() else 100
        )
        fig, ax = pub.twod_contour_plots(
            i, twod_samples, opts.labels, latex_labels, colors=colors,
            linestyles=linestyles, gridsize=gridsize, levels=opts.levels,
            return_ax=True
        )
        current_xlow, current_xhigh = ax.get_xlim()
        current_ylow, current_yhigh = ax.get_ylim()
        keys = opts.publication_kwargs.keys()
        if "xlow" in keys and "xhigh" in keys:
            ax.set_xlim(
                [
                    float(opts.publication_kwargs["xlow"]),
                    float(opts.publication_kwargs["xhigh"])
                ]
            )
        elif "xhigh" in keys:
            ax.set_xlim([current_xlow, float(opts.publication_kwargs["xhigh"])])
        elif "xlow" in keys:
            ax.set_xlim([float(opts.publication_kwargs["xlow"]), current_xhigh])
        if "ylow" in keys and "yhigh" in keys:
            ax.set_ylim(
                [
                    float(opts.publication_kwargs["ylow"]),
                    float(opts.publication_kwargs["yhigh"])
                ]
            )
        elif "yhigh" in keys:
            ax.set_ylim([current_ylow, float(opts.publication_kwargs["yhigh"])])
        elif "ylow" in keys:
            ax.set_ylim([float(opts.publication_kwargs["ylow"]), current_yhigh])
        fig.savefig("%s/2d_contour_plot_%s" % (opts.webdir, "_and_".join(i)))
        fig.close()


def make_violin_plot(opts):
    """
    """
    if opts.parameters and len(opts.parameters) != 1:
        raise Exception("Please pass a single variable that you wish to plot")
    if opts.parameters:
        default = opts.parameters
    else:
        default = default_violin_plot()
    parameters, samples = read_samples(opts.samples)

    for i in default:
        if not all(i in j for j in parameters):
            idxs = [num for num, j in enumerate(parameters) if i not in j]
            files = [opts.samples[j] for j in idxs]
            logger.warning(
                "Removing {} from violin plot because the parameter {} does "
                "not exist in the result file".format(
                    " and ".join(files), i
                )
            )
            parameters = [j for num, j in enumerate(parameters) if num not in idxs]
            opts.labels = [j for num, j in enumerate(opts.labels) if num not in idxs]
            samples = [j for num, j in enumerate(samples) if num not in idxs]
        try:
            ind = [j.index(i) for j in parameters]
            samples = [[k[ind[num]] for k in l] for num, l in
                       enumerate(samples)]
            fig = pub.violin_plots(i, samples, opts.labels, latex_labels)
            fig.savefig("%s/violin_plot_%s.png" % (opts.webdir, i))
            fig.close()
        except Exception as e:
            logger.info(
                "Failed to generate a violin plot for %s because %s" % (i, e)
            )
            continue


def make_spin_disk_plot(opts):
    """Make a spin disk plot
    """
    import seaborn

    colors, linestyles = get_colors_and_linestyles(opts)
    parameters, samples = read_samples(opts.samples)

    required_parameters = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
    for num, i in enumerate(parameters):
        if not all(j in i for j in required_parameters):
            logger.info("Failed to generate spin disk plots for %s because "
                        "%s are not in the result file" % (
                            opts.labels[num],
                            " and ".join(required_parameters)))
            continue
        try:
            ind = [i.index(j) for j in required_parameters]
            spin_samples = [[k[idx] for k in samples[num]] for idx in ind]
            fig = pub.spin_distribution_plots(
                required_parameters, spin_samples, opts.labels[num],
                colors[num])
            fig.savefig("%s/spin_disk_plot_%s.png" % (
                opts.webdir, opts.labels[num]))
            fig.close()
        except Exception as e:
            logger.warning(
                "Failed to generate a spin disk plot for %s because %s" % (
                    opts.labels[num], e
                )
            )
            continue


def make_population_scatter_plot(opts):
    """Make a scatter plot showing a population of runs
    """
    if len(opts.samples) > 1:
        parameters, samples = read_samples(opts.samples)
        plotting_data = {}
        xerr, yerr = None, None
        if "error" in opts.plot:
            xerr, yerr = {}, {}
        for num, label in enumerate(opts.labels):
            if not all(i in parameters[num] for i in opts.parameters):
                logger.warning(
                    "'{}' does not include samples for '{}' and/or '{}'. This "
                    "analysis will not be added to the plot".format(
                        label, opts.parameters[0], opts.parameters[1]
                    )
                )
                continue
            plotting_data[label] = {}
            if xerr is not None:
                xerr[label] = {}
                yerr[label] = {}
            for param in opts.parameters:
                ind = parameters[num].index(param)
                plotting_data[label][param] = np.median(
                    [i[ind] for i in samples[num]]
                )
            if xerr is not None:
                ind = parameters[num].index(opts.parameters[0])
                xerr[label][opts.parameters[0]] = [
                    np.abs(plotting_data[label][opts.parameters[0]] - np.percentile(
                        [i[ind] for i in samples[num]], j
                    )) for j in [5, 95]
                ]
            if yerr is not None:
                ind = parameters[num].index(opts.parameters[1])
                yerr[label][opts.parameters[1]] = [
                    np.abs(plotting_data[label][opts.parameters[1]] - np.percentile(
                        [i[ind] for i in samples[num]], j
                    )) for j in [5, 95]
                ]
        fig = pop.scatter_plot(
            opts.parameters, plotting_data, latex_labels, xerr=xerr, yerr=yerr
        )
        fig.savefig("{}/event_scatter_plot_{}.png".format(
            opts.webdir, "_and_".join(opts.parameters)
        ))
        fig.close()


def main(args=None):
    """Top level interface for `summarypublication`
    """
    latex_labels.update(GWlatex_labels)
    parser = command_line()
    opts = parser.parse_args(args=args)
    make_dir(opts.webdir)
    func_map = {"2d_contour": make_2d_contour_plot,
                "violin": make_violin_plot,
                "spin_disk": make_spin_disk_plot,
                "population_scatter": make_population_scatter_plot,
                "population_scatter_error": make_population_scatter_plot}
    func_map[opts.plot](opts)


if __name__ == "__main__":
    main()
