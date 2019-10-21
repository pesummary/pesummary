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

from pesummary.utils.utils import logger, make_dir
from pesummary.core.inputs import PostProcessing
from pesummary.gw.inputs import GWPostProcessing
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.main import _PlotGeneration
from pesummary.gw.command_line import DictionaryAction
from pesummary.gw.file.read import read


class PlotGeneration(object):
    """Wrapper class for _GWPlotGeneration and _CorePlotGeneration

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default", gw=False):
        self.inputs = inputs
        self.colors = colors
        self.gw = gw
        self.generate_plots()

    def generate_plots(self):
        """Generate all plots for all result files passed
        """
        logger.info("Starting to generate plots")
        if self.gw:
            object = _GWPlotGeneration(self.inputs, colors=self.colors)
        else:
            object = _CorePlotGeneration(self.inputs, colors=self.colors)
        object.generate_plots()
        logger.info("Finished generating plots")


class _CorePlotGeneration(PostProcessing):
    """Class to generate all plots associated with the Core module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.core.plots.main import _PlotGeneration

        super(_CorePlotGeneration, self).__init__(inputs, colors)
        self.plotting_object = _PlotGeneration(
            webdir=self.webdir, labels=self.labels,
            samples=self.samples, kde_plot=self.kde_plot,
            existing_labels=self.existing_labels,
            existing_injection_data=self.existing_injection_data,
            existing_samples=self.existing_samples,
            same_parameters=self.same_parameters,
            injection_data=self.injection_data,
            colors=self.colors, custom_plotting=self.custom_plotting,
            add_to_existing=self.add_to_existing, priors=self.priors,
            include_prior=self.include_prior, weights=self.weights
        )

    def generate_plots(self):
        """Generate all plots within the Core module
        """
        self.plotting_object.generate_plots()


class _GWPlotGeneration(GWPostProcessing):
    """Class to generate all plots associated with the GW module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.gw.plots.main import _PlotGeneration

        super(_GWPlotGeneration, self).__init__(inputs, colors)
        self.plotting_object = _PlotGeneration(
            webdir=self.webdir, labels=self.labels,
            samples=self.samples, kde_plot=self.kde_plot,
            existing_labels=self.existing_labels,
            existing_injection_data=self.existing_injection_data,
            existing_samples=self.existing_samples,
            existing_file_kwargs=self.existing_file_kwargs,
            existing_approximant=self.existing_approximant,
            existing_metafile=self.existing_metafile,
            same_parameters=self.same_parameters,
            injection_data=self.injection_data,
            result_files=self.result_files,
            file_kwargs=self.file_kwargs,
            colors=self.colors, custom_plotting=self.custom_plotting,
            add_to_existing=self.add_to_existing, priors=self.priors,
            no_ligo_skymap=self.no_ligo_skymap,
            nsamples_for_skymap=self.nsamples_for_skymap,
            detectors=self.detectors, maxL_samples=self.maxL_samples,
            gwdata=self.gwdata, calibration=self.calibration,
            psd=self.psd, approximant=self.approximant,
            multi_threading_for_skymap=self.multi_threading_for_skymap,
            pepredicates_probs=self.pepredicates_probs,
            include_prior=self.include_prior, publication=self.publication,
            existing_psd=self.existing_psd,
            existing_calibration=self.existing_calibration, weights=self.weights
        )

    def generate_plots(self):
        """Generate all plots within the GW module
        """
        self.plotting_object.generate_plots()


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default="./")
    parser.add_argument("-s", "--samples", dest="samples", nargs='+',
                        help="Path to PESummary metafile", default=None)
    parser.add_argument("--labels", dest="labels",
                        help="labels used to distinguish runs", nargs='+',
                        default=None)
    parser.add_argument("--plot", dest="plot",
                        help=("name of the publication plot you wish to "
                              "produce"), default="2d_contour",
                        choices=["1d_histogram", "sample_evolution",
                                 "autocorrelation", "skymap"])
    parser.add_argument("--parameters", dest="parameters", nargs="+",
                        help=("parameters of the 2d contour plot you wish to "
                              "make"), default=None)
    parser.add_argument("--plot_kwargs", help="Optional plotting kwargs",
                        action=DictionaryAction, nargs="+", default={})
    parser.add_argument("--inj", help="Injected value", default=None)
    parser.add_argument("--kde_plot", action="store_true",
                        help="plot a kde rather than a histogram",
                        default=False)
    parser.add_argument("--burnin", dest="burnin",
                        help="Number of samples to remove as burnin",
                        default=None)
    return parser


def check_inputs(opts):
    """Check that the inputs are compatible with `summaryplots`
    """
    from pesummary.utils.exceptions import InputError

    base = "Please provide {} for each result file"
    if opts.inj is None:
        opts.inj = [float("nan")] * len(opts.samples)
    if opts.labels is None:
        opts.labels = [i for i in range(len(opts.samples))]
    if len(opts.samples) != len(opts.labels):
        raise InputError(base.format("a label"))
    if len(opts.samples) != len(opts.inj):
        raise InputError(base.format("the injected value"))
    if opts.burnin is not None:
        opts.burnin = int(opts.burnin)
    return opts


def read_input_file(path_to_file):
    """Use PESummary to read a result file

    Parameters
    ----------
    path_to_file: str
        path to the results file
    """
    from pesummary.gw.file.read import read

    f = read(path_to_file)
    return f


def oned_histogram_plot(opts):
    """Make a 1d histogram plot
    """
    for num, samples in enumerate(opts.samples):
        data = read(samples)
        for parameter in opts.parameters:
            _PlotGeneration._oned_histogram_plot(
                opts.webdir, opts.labels[num], parameter,
                data.samples_dict[parameter][opts.burnin:],
                latex_labels[parameter], opts.inj[num], kde=opts.kde_plot
            )


def sample_evolution_plot(opts):
    """Make a sample evolution plot
    """
    for num, samples in enumerate(opts.samples):
        data = read(samples)
        for parameter in opts.parameters:
            _PlotGeneration._sample_evolution_plot(
                opts.webdir, opts.labels[num], parameter,
                data.samples_dict[parameter][opts.burnin:],
                latex_labels[parameter], opts.inj[num]
            )


def autocorrelation_plot(opts):
    """Make an autocorrelation plot
    """
    for num, samples in enumerate(opts.samples):
        data = read(samples)
        for parameter in opts.parameters:
            _PlotGeneration._autocorrelation_plot(
                opts.webdir, opts.labels[num], parameter,
                data.samples_dict[parameter][opts.burnin:]
            )


def skymap_plot(opts):
    """Make a skymap plot
    """
    for num, samples in enumerate(opts.samples):
        data = read(samples)
        _PlotGeneration._skymap_plot(
            opts.webdir, data.samples_dict["ra"][opts.burnin:],
            data.samples_dict["dec"][opts.burnin:], opts.labels[num]
        )


def main():
    """The main interface for `summaryplots`
    """
    latex_labels.update(GWlatex_labels)
    parser = command_line()
    opts = parser.parse_args()
    opts = check_inputs(opts)
    make_dir(opts.webdir)
    func_map = {
        "1d_histogram": oned_histogram_plot,
        "sample_evolution": sample_evolution_plot,
        "autocorrelation": autocorrelation_plot,
        "skymap": skymap_plot
    }
    func_map[opts.plot](opts)


if __name__ == "__main__":
    main()
