#! /usr/bin/env python

# Copyright (C) 2020  Aditya Vijaykumar <aditya.vijaykumar@ligo.org>
#                     Charlie Hoy <charlie.hoy@ligo.org>
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

import os
import socket
import pesummary
from pesummary.core.webpage.main import _WebpageGeneration
from pesummary.gw.conversions.tgr import imrct_deviation_parameters_from_final_mass_final_spin
from pesummary.io import read
from pesummary.gw.pepredicates import PEPredicates
from pesummary.gw.p_astro import PAstro
from pesummary.utils.utils import make_dir, logger, guess_url
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.utils.exceptions import InputError
import argparse


__doc__ = """This executable is used to generate a txt file containing the
source classification probailities"""

TESTS = ["imrct"]


def command_line():
    """Generate an Argument Parser object to control the command line options"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--webdir", dest="webdir", help="make page and plots in DIR", metavar="DIR", required=True, default=None
    )
    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        help="What test do you want to run? Currently only supports `imrct`",
        metavar="DIR",
        default=None,
        choices=TESTS,
    )
    parser.add_argument("-s", "--samples", dest="samples", help="Posterior samples hdf5 file", nargs="+", default=None)
    parser.add_argument("--labels", dest="labels", help="labels used to distinguish runs", nargs="+", default=None)
    parser.add_argument(
        "--evolve_spins",
        dest="evolve_spins",
        help="Evolve spins while calculating remnant quantities",
        action="store_true",
    )
    parser.add_argument(
        "--cutoff_frequency",
        dest="cutoff_frequency",
        help="Cutoff Frequency for IMRCT. Overrides any cutoff frequency present in the supplied files.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--links_to_pe_pages",
        dest="links_to_pe_pages",
        help="Links to PE pages separated by space.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--disable_pe_page_generation", action="store_true", help="Disable PE page generation", default=False
    )
    parser.add_argument(
        "--pe_page_options",
        dest="pe_page_options",
        help=(
            "Additional options to pass to 'summarypages' when generating PE "
            "webpages. All options should be wrapped in quotation marks like, "
            "--pe_page_options='--no_ligo_skymap --nsamples 1000 --psd...'. "
            "See 'summarypages --help' for details. These options are added to "
            "base executable: 'summarypages --webdir {} --samples {} --labels "
            "{} --gw'"
        ),
        type=str,
        default="",
    )
    parser.add_argument(
        "--make_diagnostic_plots", dest="make_diagnostic_plots", help="Make extra diagnostic plots", action="store_true"
    )
    return parser


def generate_imrct_deviation_parameters(samples, evolve_spins=True, **kwargs):
    """Generate deviation parameter pdfs for the IMR Consistency Test

    Parameters
    ----------
    inspiral_samples_file: filename
        Path to inspiral samples file
    postinspiral_samples_file: filename
        Path to post-inspiral samples file
    kwargs: dict, optional
        Keywords to be passed to imrct_deviation_parameters_from_final_mass_final_spin

    Returns
    -------
    deviations: ProbabilityDict2D
    """
    import time

    t0 = time.time()
    logger.info("Calculating IMRCT deviation parameters and GR Quantile")
    evolve_spins_string = "evolved"
    if not evolve_spins:
        evolve_spins_string = "non_" + evolve_spins_string

    samples_string = "final_{}_" + evolve_spins_string
    imrct_deviations = imrct_deviation_parameters_from_final_mass_final_spin(
        samples["inspiral"][samples_string.format("mass")],
        samples["inspiral"][samples_string.format("spin")],
        samples["postinspiral"][samples_string.format("mass")],
        samples["postinspiral"][samples_string.format("spin")],
        **kwargs,
    )
    gr_quantile = (
        imrct_deviations["final_mass_final_spin_deviations"].minimum_encompassing_contour_level(0.0, 0.0) * 100
    )
    t1 = time.time()
    data = kwargs.copy()
    data["evolve_spin"] = evolve_spins
    data["Time (seconds)"] = round(t1 - t0, 2)
    data["GR Quantile (%)"] = round(gr_quantile[0], 2)

    logger.info(
        "Calculation Finished in {} seconds. GR Quantile is {} %.".format(
            data["Time (seconds)"], data["GR Quantile (%)"]
        )
    )

    return imrct_deviations, data


def make_imrct_plots(imrct_deviations, samples, webdir=None, evolve_spins=False, make_diagnostic_plots=False):
    """Save the plots generated by PEPredicates

    Parameters
    ----------
    imrct_deviations: ProbabilityDict2D
        Output of imrct_deviation_parameters_from_final_mass_final_spin
    webdir: str
        path to save the files
    labels: list
        lisy of strings to identify each result file
    plot_type: str
        The plot type that you wish to make
    probs: dict
        Dictionary of classification probabilities
    """
    if webdir is None:
        webdir = "./"

    evolve_spins_string = "evolved"
    if not evolve_spins:
        evolve_spins_string = "non_" + evolve_spins_string

    samples_string = "final_{}_" + evolve_spins_string
    plotdir = os.path.join(webdir, "plots/")
    make_dir(plotdir)
    base_string = plotdir + "imrct_{}.png"
    logger.info("Creating IMRCT deviations triangle plot")

    plot_kwargs = dict(
        grid=True,
        smooth=4,
        type="triangle",
        cmap="YlOrBr",
        fontsize=dict(label=20),
        levels=[0.68, 0.95],
        level_kwargs=dict(colors=["k", "k"]),
        xlabel=r"$\Delta M_{\mathrm{f}} / \bar{M_{\mathrm{f}}}$",
        ylabel=r"$\Delta a_{\mathrm{f}} / \bar{a_{\mathrm{f}}}$",
        fig_kwargs=dict(wspace=0.2, hspace=0.2),
    )
    fig, _, ax_2d, _ = imrct_deviations.plot(
        "final_mass_final_spin_deviations",
        **plot_kwargs,
    )
    ax_2d.plot(0, 0, "k+", ms=12, mew=2)

    fig.savefig(base_string.format("deviations_triangle_plot"))
    fig.close()
    logger.info("Finished creating IMRCT deviations triangle plot.")

    if make_diagnostic_plots:
        logger.info("Creating diagnostic plots")
        plot_kwargs = dict(
            grid=True,
            smooth=4,
            type="triangle",
            fill_alpha=0.2,
            labels=["inspiral", "postinspiral"],
            fontsize=dict(label=20, legend=14),
            fig_kwargs=dict(wspace=0.2, hspace=0.2),
        )
        parameters_to_plot = [
            [samples_string.format("mass"), samples_string.format("spin")],
            ["mass_1", "mass_2"],
            ["a_1", "a_2"],
        ]

        for parameters in parameters_to_plot:
            fig, _, _, _ = samples.plot(
                parameters,
                **plot_kwargs,
            )
            save_string = "{}_{}".format(parameters[0], parameters[1])
            fig.savefig(base_string.format(save_string))
            fig.close()
        logger.info("Finished creating diagnostic plots.")


def generate_pe_pages(webdir, result_files, labels, additional_options=""):
    """Launch a subprocess to generate PE pages using `summarypages`

    Parameters
    ----------
    webdir: str
        directory to store the output pages
    result_files: list
        list of paths to result files
    labels: list
        list of labels to use for each result file
    additional_options: str, optional
        additional options to add to the summarypages executable
    """
    from .summarytest import launch
    from subprocess import PIPE

    logger.info("Creating PE summarypages")
    base_command_line = "summarypages --webdir {} --samples {} --labels {} --gw ".format(
        webdir, " ".join(result_files), " ".join(labels)
    )
    base_command_line += additional_options
    process = launch(base_command_line, check_call=False, out=PIPE, err=PIPE)
    return base_command_line, process


def check_on_pe_page_generate(command_line, process):
    """Check on the status of the PE page subprocess

    Parameters
    ----------
    command_line: str
        command line used to generate the PE page
    process:

    """
    _running = True
    while _running:
        if process.poll() is not None:
            _running = False
            if process.returncode != 0:
                msg = "The PE job: {} failed. The error is: {}"
                _output, _error = process.communicate()
                raise ValueError(msg.format(command_line, _error.decode("utf-8")))
    logger.info("PE page generation complete")


class TGRWebpageGeneration(_WebpageGeneration):
    """Class to handle webpage generation displaying the outcome of various
    TGR tests

    Parameters
    ----------
    webdir: str
        the web directory of the run
    path_to_results_file: str
        the path to the lalinference h5 file
    *args: tuple
        all args passed to the _WebpageGeneration class
    **kwargs: dict
        all kwargs passed to the _WebpageGeneration class
    """

    def __init__(
        self,
        webdir,
        path_to_results_file,
        *args,
        test="all",
        test_key_data={},
        open_files=None,
        links_to_pe_pages=None,
        input_file_summary={},
        **kwargs
    ):
        self.test = test
        if self.test.lower() == "all":
            labels = TESTS
        else:
            labels = [self.test]

        _labels = labels
        self.links_to_pe_pages = links_to_pe_pages
        if open_files is not None:
            _labels = list(open_files.keys())
        super(TGRWebpageGeneration, self).__init__(
            *args, webdir=webdir, labels=_labels, user=os.environ["USER"], samples=open_files, **kwargs
        )
        self.labels = labels
        self.file_versions = {label: "No version information found" for label in self.labels}
        self.test_key_data = test_key_data
        self.open_files = open_files
        self.input_file_summary = input_file_summary
        if not len(self.test_key_data):
            self.test_key_data = {_test: {} for _test in self.labels}
        if not len(self.input_file_summary) and self.open_files is None:
            self.input_file_summary = {_test: {} for _test in self.labels}
        elif not len(self.input_file_summary):
            self.input_file_summary = self.key_data
        self.copy_css_and_js_scripts()

    def generate_webpages(self, make_diagnostic_plots=False):
        """Generate all webpages for all tests"""
        self.make_home_pages()
        if self.test == "imrct" or self.test == "all":
            self.make_imrct_pages(make_diagnostic_plots=make_diagnostic_plots)
        self.make_version_page()
        self.make_logging_page()
        self.make_about_page()
        self.generate_specific_javascript()

    def make_navbar_for_result_page(self):
        links = self.make_navbar_for_homepage()
        # must have the 'external:' syntax
        # dummy_link = "external:https://www.google.com"
        if self.links_to_pe_pages is not None:
            link_format = "external:{}"
            links.insert(
                2,
                [
                    "PE Pages",
                    [
                        {self.labels[0]: link_format.format(self.links_to_pe_pages[0])},
                        {self.labels[1]: link_format.format(self.links_to_pe_pages[1])},
                    ],
                ],
            )
        return links

    def make_navbar_for_comparison_page(self):
        return

    def _test_links(self):
        """Return the navbar structure for the Test tab"""
        if self.test == "all":
            return TESTS
        return [self.test]

    def make_navbar_for_homepage(self):
        """Make a navbar for the homepage"""
        links = ["home", ["Tests", self._test_links()], "Logging", "Version"]
        return links

    def _make_home_pages(self, pages):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = self.setup_page("home", self.navbar["home"])
        html_file.make_banner("Tests of General Relativity", key="content", content=" ")
        image_contents = [["plots/imrct_deviations_triangle_plot.png"]]
        html_file = self.make_modal_carousel(html_file, image_contents=image_contents, unique_id=True)
        _banner_desc = "Below we show summary statistics associated with each test of GR"
        html_file.make_banner(
            approximant="Summary Statistics", key="content", content=_banner_desc, _style="font-size: 26px;"
        )
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        html_file.make_container(style=_style)
        html_file.make_div(4, _class=_class, _style=None)
        base_label = self.labels[0]
        total_keys = list(self.test_key_data[base_label].keys())
        if len(self.labels) > 1:
            for _label in self.labels[1:]:
                total_keys += [key for key in self.test_key_data[_label].keys() if key not in total_keys]
        table_contents = [
            [i] + [self.test_key_data[i][key] if key in self.test_key_data[i].keys() else "-" for key in total_keys]
            for i in self.labels
        ]
        _headings = [" "] + total_keys
        html_file.make_table(
            headings=_headings, format="table-hover", heading_span=1, contents=table_contents, accordian=False
        )
        html_file.end_div(4)
        html_file.end_container()
        html_file.export("{}.csv".format("summary_of_tests_of_GR.csv"))
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        for key, value in self.input_file_summary.items():
            html_file.make_banner(
                approximant="{} Summary Table".format(key), key="summary_table", _style="font-size: 26px;"
            )
            html_file.make_container(style=_style)
            html_file.make_div(4, _class=_class, _style=None)
            headings = [" "] + self.key_data_headings.copy()
            contents = []
            for j in value.keys():
                row = []
                row.append(j)
                row += self.key_data_table[key][j]
                contents.append(row)
            html_file.make_table(
                headings=headings,
                contents=contents,
                heading_span=1,
                accordian=False,
                format="table-hover header-fixed",
                sticky_header=True,
            )
            html_file.end_div(4)
            html_file.end_container()
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_imrct_pages(self, make_diagnostic_plots=False):
        """Make the IMR consistency test pages"""
        pages = ["imrct"]
        self.create_blank_html_pages(pages)
        html_file = self.setup_page("imrct", self.navbar["result_page"], title="IMR Consistency Test")
        html_file.make_banner(approximant="IMR Consistency Test", key="content", content=" ")
        desc = "Below we show the executive plots for the IMR consistency test"
        html_file.make_banner(approximant="Executive plots", key="content", content=desc, _style="font-size: 26px;")
        path = self.image_path["other"]
        base_string = path + "imrct_{}.png"
        image_contents = [[base_string.format("deviations_triangle_plot")]]
        captions = [
            [
                (
                    "This triangle plot shows the 2D and marginalized 1D "
                    "posterior distributions for the fractional parameters "
                    "fractional_final_mass and fractional_final_spin. The "
                    "prediction from General Relativity is shown"
                ),
            ]
        ]
        html_file = self.make_modal_carousel(
            html_file,
            image_contents,
            captions=captions,
            cli=[[" "]],
            unique_id=True,
            extra_div=True,
            autoscale=False,
        )

        if make_diagnostic_plots:
            desc = "Below we show additional plots generated for the IMR consistency " "test"
            html_file.make_banner(
                approximant="Additional plots", key="content", content=desc, _style="font-size: 26px;"
            )
            image_contents = [
                [
                    base_string.format("final_mass_non_evolved_final_spin_non_evolved"),
                    base_string.format("mass_1_mass_2"),
                    base_string.format("a_1_a_2"),
                ],
            ]
            _base = "2D posterior distribution for {} estimated from the inspiral and post-inspiral parts of the signal"
            captions = [
                [
                    _base.format("final_mass and final_spin"),
                    _base.format("mass_1 and mass_2"),
                    _base.format("a_1 and a_2"),
                ],
            ]
            cli = [[" ", " ", " "]]
            html_file = self.make_modal_carousel(
                html_file, image_contents, captions=captions, cli=cli, unique_id=True, autoscale=True
            )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()


def main(args=None):
    """Top level interface for `summarytgr`"""
    from pesummary.gw.parser import parser

    CHECKUP = False
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    make_dir(opts.webdir)
    base_url = guess_url(os.path.abspath(opts.webdir), socket.getfqdn(), os.environ["USER"])
    evolve_spins_string = "evolved"
    if not opts.evolve_spins:
        evolve_spins = opts.evolve_spins
        evolve_spins_string = "non_" + evolve_spins_string
    else:
        evolve_spins = "ISCO"

    open_files_paths = {_label: read(path) for _label, path in zip(opts.labels, opts.samples)}
    open_files = MultiAnalysisSamplesDict({_label: open_files_paths[_label].samples_dict for _label in opts.labels})
    test_key_data = {}
    if opts.test == "imrct":
        if sorted(opts.labels) != ["inspiral", "postinspiral"]:
            raise ValueError(
                "The IMRCT test requires an inspiral and postinspiral result "
                "file. Please indicate which file is the inspiral and which "
                "is postinspiral by providing these exact labels to the "
                "summarytgr executable"
            )
        test_key_data["imrct"] = {}
        for key, sample in open_files.items():
            if "final_mass_{}".format(evolve_spins_string) not in sample.keys():
                logger.info("Remnant properties not in samples, trying to generate them")
                returned_extra_kwargs = sample.generate_all_posterior_samples(
                    evolve_spins=evolve_spins, return_kwargs=True
                )
                converted_keys = sample.keys()
                if "final_mass_{}".format(evolve_spins_string) not in converted_keys:
                    raise KeyError("Remnant properties not in samples and cannot be generated")
                else:
                    logger.info("Remnant properties generated.")
                    for fit in ["final_mass_NR_fits", "final_spin_NR_fits"]:
                        test_key_data["imrct"]["{} {}".format(key, fit)] = returned_extra_kwargs["meta_data"][fit]

        imrct_deviations, data = generate_imrct_deviation_parameters(open_files, evolve_spins=opts.evolve_spins)
        make_imrct_plots(
            imrct_deviations,
            open_files,
            webdir=opts.webdir,
            evolve_spins=opts.evolve_spins,
            make_diagnostic_plots=opts.make_diagnostic_plots,
        )

        if opts.cutoff_frequency is not None:
            data["Cutoff frequency"] = opts.cutoff_frequency
        else:
            logger.info("No Cutoff Frequency supplied on command line. Checking in the samples files.")
            try:
                if float(open_files["inspiral"].config["maximum-frequency"]) == float(
                    open_files["postinspiral"].config["minimum-frequency"]
                ):
                    data["Cutoff Frequency"] = float(open_files["inspiral"].config["maximum-frequency"])
                else:
                    logger.warning("The minimum and maximum frequencies for inspiral and postinspiral do not match!")
                    data["Cutoff Frequency"] = None
            except (AttributeError, KeyError):
                logger.info("No Cutoff Frequency information in supplied samples file. Setting to None.")
                data["Cutoff Frequency"] = None

        test_key_data["imrct"].update(data)

        if not len(opts.links_to_pe_pages):
            try:
                links_to_pe_pages = [open_files_paths[_label].history["webpage_url"] for _label in opts.labels]
            except (AttributeError, KeyError, TypeError):
                links_to_pe_pages = []
        else:
            links_to_pe_pages = opts.links_to_pe_pages

        if not len(links_to_pe_pages) and not opts.disable_pe_page_generation:
            CHECKUP = True
            _webdir = os.path.join(opts.webdir, "pe_pages")
            _command_line, process = generate_pe_pages(_webdir, opts.samples, opts.labels, opts.pe_page_options)
            links_to_pe_pages = ["../pe_pages/html/{0}_{0}.html".format(label) for label in opts.labels]

    logger.info("Creating webpages for IMRCT")
    webpage = TGRWebpageGeneration(
        opts.webdir,
        opts.samples,
        test=opts.test,
        open_files=open_files,
        links_to_pe_pages=links_to_pe_pages,
        test_key_data=test_key_data,
    )
    webpage.generate_webpages(make_diagnostic_plots=opts.make_diagnostic_plots)
    msg = "Complete. Webpages can be viewed at the following url {}.".format(base_url + "/home.html")
    if CHECKUP:
        msg += "The PE webpages are still being generated. These links will not " "work currently"
    logger.info(msg)
    if CHECKUP:
        check_on_pe_page_generate(command_line, process)


if __name__ == "__main__":
    main()
