#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger, make_dir
from pesummary.core.cli.parser import ArgumentParser as _ArgumentParser
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.main import _PlotGeneration
from pesummary.core.cli.actions import DictionaryAction
from pesummary.gw.file.read import read

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        if self.gw and self.inputs.public:
            object = _PublicGWPlotGeneration(self.inputs, colors=self.colors)
            self.ligo_skymap_PID = object.ligo_skymap_PID
        elif self.gw:
            object = _GWPlotGeneration(self.inputs, colors=self.colors)
        else:
            object = _CorePlotGeneration(self.inputs, colors=self.colors)
        object.generate_plots()
        if self.gw:
            self.ligo_skymap_PID = object.ligo_skymap_PID
        logger.info("Finished generating plots")


class _CorePlotGeneration(object):
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
        key_data = inputs.grab_key_data_from_result_files()
        expert_plots = not inputs.disable_expert
        self.plotting_object = _PlotGeneration(
            webdir=inputs.webdir, labels=inputs.labels,
            samples=inputs.samples, kde_plot=inputs.kde_plot,
            existing_labels=inputs.existing_labels,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            same_parameters=inputs.same_parameters,
            injection_data=inputs.injection_data,
            colors=inputs.colors, custom_plotting=inputs.custom_plotting,
            add_to_existing=inputs.add_to_existing, priors=inputs.priors,
            include_prior=inputs.include_prior, weights=inputs.weights,
            disable_comparison=inputs.disable_comparison,
            linestyles=inputs.linestyles,
            disable_interactive=inputs.disable_interactive,
            disable_corner=inputs.disable_corner,
            multi_process=inputs.multi_process, mcmc_samples=inputs.mcmc_samples,
            corner_params=inputs.corner_params, expert_plots=expert_plots,
            checkpoint=inputs.restart_from_checkpoint, key_data=key_data
        )

    def generate_plots(self):
        """Generate all plots within the Core module
        """
        self.plotting_object.generate_plots()


class _GWPlotGeneration(object):
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
        key_data = inputs.grab_key_data_from_result_files()
        expert_plots = not inputs.disable_expert
        self.plotting_object = _PlotGeneration(
            webdir=inputs.webdir, labels=inputs.labels,
            samples=inputs.samples, kde_plot=inputs.kde_plot,
            existing_labels=inputs.existing_labels,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            existing_file_kwargs=inputs.existing_file_kwargs,
            existing_approximant=inputs.existing_approximant,
            existing_metafile=inputs.existing_metafile,
            same_parameters=inputs.same_parameters,
            injection_data=inputs.injection_data,
            result_files=inputs.result_files,
            file_kwargs=inputs.file_kwargs,
            colors=inputs.colors, custom_plotting=inputs.custom_plotting,
            add_to_existing=inputs.add_to_existing, priors=inputs.priors,
            no_ligo_skymap=inputs.no_ligo_skymap,
            nsamples_for_skymap=inputs.nsamples_for_skymap,
            detectors=inputs.detectors, maxL_samples=inputs.maxL_samples,
            gwdata=inputs.gwdata, calibration=inputs.calibration,
            psd=inputs.psd, approximant=inputs.approximant,
            multi_threading_for_skymap=inputs.multi_threading_for_skymap,
            pepredicates_probs=inputs.pepredicates_probs,
            include_prior=inputs.include_prior, publication=inputs.publication,
            existing_psd=inputs.existing_psd,
            existing_calibration=inputs.existing_calibration, weights=inputs.weights,
            linestyles=inputs.linestyles,
            disable_comparison=inputs.disable_comparison,
            disable_interactive=inputs.disable_interactive,
            disable_corner=inputs.disable_corner,
            publication_kwargs=inputs.publication_kwargs,
            multi_process=inputs.multi_process, mcmc_samples=inputs.mcmc_samples,
            skymap=inputs.skymap, existing_skymap=inputs.existing_skymap,
            corner_params=inputs.corner_params,
            preliminary_pages=inputs.preliminary_pages, expert_plots=expert_plots,
            checkpoint=inputs.restart_from_checkpoint, key_data=key_data
        )
        self.ligo_skymap_PID = self.plotting_object.ligo_skymap_PID

    def generate_plots(self):
        """Generate all plots within the GW module
        """
        self.plotting_object.generate_plots()


class _PublicGWPlotGeneration(object):
    """Class to generate all plots associated with the GW module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.gw.plots.public import _PlotGeneration
        expert_plots = not inputs.disable_expert
        self.plotting_object = _PlotGeneration(
            webdir=inputs.webdir, labels=inputs.labels,
            samples=inputs.samples, kde_plot=inputs.kde_plot,
            existing_labels=inputs.existing_labels,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            existing_file_kwargs=inputs.existing_file_kwargs,
            existing_approximant=inputs.existing_approximant,
            existing_metafile=inputs.existing_metafile,
            same_parameters=inputs.same_parameters,
            injection_data=inputs.injection_data,
            result_files=inputs.result_files,
            file_kwargs=inputs.file_kwargs,
            colors=inputs.colors, custom_plotting=inputs.custom_plotting,
            add_to_existing=inputs.add_to_existing, priors=inputs.priors,
            no_ligo_skymap=inputs.no_ligo_skymap,
            nsamples_for_skymap=inputs.nsamples_for_skymap,
            detectors=inputs.detectors, maxL_samples=inputs.maxL_samples,
            gwdata=inputs.gwdata, calibration=inputs.calibration,
            psd=inputs.psd, approximant=inputs.approximant,
            multi_threading_for_skymap=inputs.multi_threading_for_skymap,
            pepredicates_probs=inputs.pepredicates_probs,
            include_prior=inputs.include_prior, publication=inputs.publication,
            existing_psd=inputs.existing_psd,
            existing_calibration=inputs.existing_calibration, weights=inputs.weights,
            linestyles=inputs.linestyles,
            disable_comparison=inputs.disable_comparison,
            disable_interactive=inputs.disable_interactive,
            disable_corner=inputs.disable_corner,
            publication_kwargs=inputs.publication_kwargs,
            multi_process=inputs.multi_process, mcmc_samples=inputs.mcmc_samples,
            skymap=inputs.skymap, existing_skymap=inputs.existing_skymap,
            corner_params=inputs.corner_params,
            preliminary_pages=inputs.preliminary_pages, expert_plots=expert_plots,
            checkpoint=inputs.restart_from_checkpoint
        )
        self.ligo_skymap_PID = self.plotting_object.ligo_skymap_PID

    def generate_plots(self):
        """Generate all plots within the GW module
        """
        self.plotting_object.generate_plots()


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--plot": {
                    "help": "name of the publication plot you wish to produce",
                    "choices": [
                        "1d_histogram", "sample_evolution", "autocorrelation",
                        "skymap"
                    ],
                    "default": "2d_contour"
                },
                "--parameters": {
                    "nargs": "+",
                    "help": "parameters of the 2d contour plot you wish to make",
                },
                "--plot_kwargs": {
                    "help": "Optional plotting kwargs",
                    "default": {},
                    "nargs": "+",
                    "action": DictionaryAction
                },
                "--inj": {
                    "help": "Injected value",
                },
            }
        )
        return options


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


def main(args=None):
    """The main interface for `summaryplots`
    """
    latex_labels.update(GWlatex_labels)
    parser = ArgumentParser(description=__doc__)
    parser.add_known_options_to_parser(
        [
            "--webdir", "--samples", "--labels", "--plot", "--parameters",
            "--kde_plot", "--burnin", "--disable_comparison", "--plot_kwargs",
            "--inj"
        ]
    )
    opts = parser.parse_args(args=args)
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
