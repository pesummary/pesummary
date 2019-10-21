#! /usr/bin/env python

# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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

import pesummary
from pesummary.gw.file.read import read as GWRead
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.gw.plots import publication as pub
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.utils.utils import make_dir, logger
from pesummary.gw.command_line import DictionaryAction
import argparse
import matplotlib.pyplot as plt


__doc__ = """This executable is used to generate publication quality plots given
result files"""


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
                        choices=["2d_contour", "violin", "spin_disk"])
    parser.add_argument("--parameters", dest="parameters", nargs="+",
                        help=("parameters of the 2d contour plot you wish to "
                              "make"), default=None)
    parser.add_argument("--plot_kwargs", help="Optional plotting kwargs",
                        action=DictionaryAction, nargs="+", default={})
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
        f = GWRead(i)
        if isinstance(f, pesummary.gw.file.formats.pesummary.PESummary):
            parameters.append(f.parameters[0])
            samples.append(f.samples[0])
        else:
            f.generate_all_posterior_samples()
            parameters.append(f.parameters)
            samples.append(f.samples)
    return parameters, samples


def make_2d_contour_plot(opts):
    """Make a 2d contour plot
    """
    if opts.parameters and len(opts.parameters) != 2:
        raise Exception("Please pass 2 variables that you wish to plot")
    if opts.parameters:
        default = [opts.parameters]
    else:
        default = default_2d_contour_plot()
    parameters, samples = read_samples(opts.samples)
    for i in default:
        if not all(all(k in j for k in i) for j in parameters):
            logger.info("Failed to generate plot because %s are not in both "
                        "result files" % (" and ".join(i)))
            continue
        ind1 = [j.index(i[0]) for j in parameters]
        ind2 = [j.index(i[1]) for j in parameters]
        samples1 = [[k[ind1[num]] for k in l] for num, l in
                    enumerate(samples)]
        samples2 = [[k[ind2[num]] for k in l] for num, l in
                    enumerate(samples)]
        twod_samples = [[j, k] for j, k in zip(samples1, samples2)]
        fig = pub.twod_contour_plots(i, twod_samples, opts.labels, latex_labels)
        current_xlow, current_xhigh = plt.xlim()
        keys = opts.plot_kwargs.keys()
        if "xlow" in keys and "xhigh" in keys:
            plt.xlim([float(opts.plot_kwargs["xlow"]), float(opts.plot_kwargs["xhigh"])])
        elif "xhigh" in keys:
            plt.xlim([current_xlow, float(opts.plot_kwargs["xhigh"])])
        elif "xlow" in keys:
            plt.xlim([float(opts.plot_kwargs["xlow"]), current_xhigh])
        fig.savefig("%s/2d_contour_plot_%s" % (opts.webdir, "_and_".join(i)))
        plt.close()


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
            logger.info("Failed to generate violin plots for %s because "
                        "%s is not in all result files" % (i, i))
            continue
        try:
            ind = [j.index(i) for j in parameters]
            samples = [[k[ind[num]] for k in l] for num, l in
                       enumerate(samples)]
            fig = pub.violin_plots(i, samples, opts.labels, latex_labels)
            fig.savefig("%s/violin_plot_%s.png" % (opts.webdir, i))
            plt.close()
        except Exception:
            logger.info("Failed to generate a violin plot for %s" % (i))
            continue


def make_spin_disk_plot(opts):
    """Make a spin disk plot
    """
    import seaborn

    palette = seaborn.color_palette(
        palette="pastel", n_colors=len(opts.samples))
    parameters, samples = read_samples(opts.samples)

    required_parameters = ["a_1", "a_2", "tilt_1", "tilt_2"]
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
                palette[num])
            fig.savefig("%s/spin_disk_plot_%s.png" % (
                opts.webdir, opts.labels[num]))
            plt.close()
        except Exception as e:
            logger.warn("Failed to generate a spin disk plot for %s because %s" % (
                        opts.labels[num], e))
            continue


def main():
    """Top level interface for `summarypublication`
    """
    latex_labels.update(GWlatex_labels)
    parser = command_line()
    opts = parser.parse_args()
    make_dir(opts.webdir)
    func_map = {"2d_contour": make_2d_contour_plot,
                "violin": make_violin_plot,
                "spin_disk": make_spin_disk_plot}
    func_map[opts.plot](opts)


if __name__ == "__main__":
    main()
