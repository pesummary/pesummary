#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import argparse
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import logger
from pesummary.gw.inputs import _GWInput
from pesummary.core.webpage.main import _WebpageGeneration

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is a lightweight version of summarypages. It
allows you to customise which parameters you wish to view rather than plotting
every single parameter in the result file"""


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
    parser.add_argument("--parameters", dest="parameters", nargs='+',
                        help=("list of parameters you wish to include in the "
                              "summarypages"),
                        default=None)
    return parser


class LWInput(_GWInput):
    """
    """
    def __init__(self, opts):
        logger.info("Command line arguments: %s" % (opts))
        self.opts = opts
        self.result_files = self.opts.samples
        self.meta_file = False
        if self.result_files is not None and len(self.result_files) == 1:
            self.meta_file = self.is_pesummary_metafile(self.result_files[0])
        self.existing = None
        self.add_to_existing = False
        self.user = None
        self.webdir = self.opts.webdir
        self.baseurl = None
        self.labels = self.opts.labels
        self.weights = {i: None for i in self.labels}
        self.config = None
        self.injection_file = None
        self.priors = None
        self.samples = self.opts.samples
        self.parameters_to_include = self.opts.parameters
        self.same_parameters = []
        self.publication = False
        self.colors = None
        self.make_directories()
        self.copy_files()

    @property
    def parameters_to_include(self):
        return self._parameters_to_include

    @parameters_to_include.setter
    def parameters_to_include(self, parameters_to_include):
        self._parameters_to_include = parameters_to_include
        if parameters_to_include is None:
            raise InputError(
                "Please provide a list of parameters you wish to plot"
            )
        for num, label in enumerate(self.labels):
            params = [
                i for i in parameters_to_include if i in
                list(self.samples[label].keys())
            ]
            not_included_params = [
                i for i in parameters_to_include if i not in
                list(self.samples[label].keys())
            ]
            if len(not_included_params) != 0:
                logger.warning(
                    "The parameters {} are not in the file {}. They will not "
                    "be included in the final pages".format(
                        ", ".join(not_included_params), self.result_files[num]
                    )
                )
            original = list(self.samples[label].keys())
            for i in original:
                if i not in params:
                    self.samples[label].pop(i)


class LWWebpageGeneration(_WebpageGeneration):
    """
    """
    def __init__(
        self, webdir=None, labels=None, samples=None, colors=None, user=None,
        config=None, same_parameters=None, baseurl=None, file_versions=None
    ):
        super(LWWebpageGeneration, self).__init__(
            webdir=webdir, samples=samples, labels=labels, publication=False,
            user=user, config=config, same_parameters=same_parameters,
            base_url=baseurl, file_versions=file_versions, hdf5=False,
            colors=colors, custom_plotting=False, existing_labels=None,
            existing_config=None, existing_file_version=None,
            existing_injection_data=None, existing_samples=None,
            existing_metafile=None, existing_file_kwargs=None,
            existing_weights=None, add_to_existing=False, notes=None,
            disable_comparison=False
        )

    def generate_webpages(self):
        """Generate all webpages for all result files passed
        """
        self.make_home_pages()
        self.make_1d_histogram_pages()
        if self.make_comparison:
            self.make_comparison_pages()
        if self.make_interactive:
            self.make_interactive_pages()
        self.make_error_page()
        self.make_version_page()
        self.make_logging_page()
        self.generate_specific_javascript()

    def make_navbar_for_result_page(self):
        """Make a navbar for the result page homepage
        """
        links = {
            i: ["1d Histograms", [{"Custom": i}, {"All": i}]] for i in
            self.labels
        }
        for num, label in enumerate(self.labels):
            for j in self.categorize_parameters(self.samples[label].keys()):
                j = [j[0], [{k: label} for k in j[1]]]
                links[label].append(j)

        final_links = {
            i: [
                "home", ["Result Pages", self._result_page_links()], links[i]
            ] for i in self.labels
        }
        if self.make_comparison:
            for label in self.labels:
                final_links[label][1][1] += ["Comparison"]
        if self.make_interactive:
            for label in self.labels:
                final_links[label].append(
                    ["Interactive", [{"Interactive_Corner": label}]]
                )
        return final_links


def make_plots(inputs):
    """Make all available plots

    Parameters
    ----------
    inputs: pesummary.cli.summarypageslw.LWInputs
        Namespace object containing all command line arguments
    """
    from pesummary.core.plots.main import _PlotGeneration

    plotting_object = _PlotGeneration(
        webdir=inputs.webdir, labels=inputs.labels, samples=inputs.samples,
        kde_plot=False, existing_labels=None, existing_injection_data=None,
        existing_samples=None, same_parameters=inputs.same_parameters,
        injection_data=inputs.injection_data, colors=inputs.colors,
        custom_plotting=None, add_to_existing=False, priors=None,
        disable_comparison=False, linestyles=None, disable_interactive=False
    )
    plotting_object.generate_plots()


def make_webpages(inputs):
    """Make the webpages to display the plots

    Parameters
    ----------
    inputs: pesummary.cli.summarypageslw.LWInputs
        Namespace object containing all command line arguments
    """
    from pesummary.core.webpage.main import _WebpageGeneration

    webpage_object = LWWebpageGeneration(
        webdir=inputs.webdir, labels=inputs.labels, samples=inputs.samples,
        colors=inputs.colors, user=inputs.user, config=inputs.config,
        same_parameters=inputs.same_parameters, baseurl=inputs.baseurl,
        file_versions=inputs.file_version
    )
    webpage_object.generate_webpages()


def main(args=None):
    """The main interface to `summarypageslw`
    """
    parser = command_line()
    opts = parser.parse_args(args=args)
    inputs = LWInput(opts)
    make_plots(inputs)
    make_webpages(inputs)


if __name__ == "__main__":
    main()
