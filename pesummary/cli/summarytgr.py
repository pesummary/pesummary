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

import ast
import os
import socket
from itertools import cycle
import pesummary
from pesummary import conf
from pesummary.gw.parser import parser as gw_parser
from pesummary.gw.webpage.tgr import TGRWebpageGeneration
from pesummary.gw.file.meta_file import TGRMetaFile
from pesummary.gw.conversions.tgr import (
    imrct_deviation_parameters_from_final_mass_final_spin,
)
from pesummary.io import read
from pesummary.utils.utils import make_dir, logger, guess_url
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
import argparse


__doc__ = """This executable is used to generate a txt file containing the
source classification probailities"""

TESTS = ["imrct"]
DEFAULT_PLOT_KWARGS = dict(
    grid=True,
    smooth=4,
    type="triangle",
    fontsize=dict(label=20, legend=14),
    fig_kwargs=dict(wspace=0.2, hspace=0.2),
)


def command_line():
    """Generate an Argument Parser object to control the command line options"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w",
        "--webdir",
        dest="webdir",
        help="make page and plots in DIR",
        metavar="DIR",
        required=True,
        default=None,
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
    parser.add_argument(
        "-s",
        "--samples",
        dest="samples",
        help="Posterior samples hdf5 file",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        help="labels used to distinguish runs",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-a",
        "--approximant",
        dest="approximant",
        help="Approximant used for the runs",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--evolve_spins",
        dest="evolve_spins",
        help="Evolve spins while calculating remnant quantities",
        action="store_true",
    )
    parser.add_argument(
        "--cutoff_frequency",
        dest="cutoff_frequency",
        help="Cutoff Frequency for IMRCT. Overrides any cutoff frequency "
        "present in the supplied files. "
        "If only one number is supplied, the inspiral maximum frequency "
        "and the postinspiral maximum frequency are set to the same number. "
        "If a list of length 2 is supplied, this assumes that the "
        "one corresponding to the inspiral label is the maximum frequency "
        "for inspiral and that corresponding to the postinspiral label is the"
        "minimum frequency for postinspiral_samples_file",
        type=float,
        nargs="+",
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
        "--disable_pe_page_generation",
        action="store_true",
        help="Disable PE page generation",
        default=False,
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
        "--make_diagnostic_plots",
        dest="make_diagnostic_plots",
        help="Make extra diagnostic plots",
        action="store_true",
    )
    return parser


def generate_imrct_deviation_parameters(
    samples,
    evolve_spins=True,
    inspiral_string="inspiral",
    postinspiral_string="postinspiral",
    **kwargs
):
    """
    Generate deviation parameter pdfs for the IMR Consistency Test

    Parameters
    ----------
    samples: MultiAnalysisSamplesDict
        Dictionary containing inspiral and postinspiral samples
    evolve_spins: bool
        If `True`, evolve spins. Default: True.
    inspiral_string: string
        Identifier for the inspiral samples
    postinspiral_string: string
        Identifier for the post-inspiral samples
    kwargs: dict, optional
        Keywords to be passed to imrct_deviation_parameters_from_final_mass_final_spin

    Returns
    -------
    imrct_deviations: ProbabilityDict2d
        2d pdf of the IMRCT deviation parameters
    data: dict
        Metadata
    """
    import time

    t0 = time.time()
    logger.info("Calculating IMRCT deviation parameters and GR Quantile")
    evolve_spins_string = "evolved"
    if not evolve_spins:
        evolve_spins_string = "non_" + evolve_spins_string

    samples_string = "final_{}_" + evolve_spins_string
    imrct_deviations = imrct_deviation_parameters_from_final_mass_final_spin(
        samples[inspiral_string][samples_string.format("mass")],
        samples[inspiral_string][samples_string.format("spin")],
        samples[postinspiral_string][samples_string.format("mass")],
        samples[postinspiral_string][samples_string.format("spin")],
        **kwargs,
    )
    gr_quantile = (
        imrct_deviations[
            "final_mass_final_spin_deviations"
        ].minimum_encompassing_contour_level(0.0, 0.0)
        * 100
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


def make_imrct_plots(
    imrct_deviations,
    samples,
    webdir="./",
    evolve_spins=False,
    make_diagnostic_plots=False,
    plot_label=None,
    inspiral_string="inspiral",
    postinspiral_string="postinspiral",
    cmap="YlOrBr",
    levels=[0.68, 0.95],
    level_kwargs={"colors": ["k", "k"]},
    xlabel=r"$\Delta M_{\mathrm{f}} / \bar{M_{\mathrm{f}}}$",
    ylabel=r"$\Delta a_{\mathrm{f}} / \bar{a_{\mathrm{f}}}$",
    save=True,
    return_fig=False,
    _default_plot_kwargs=DEFAULT_PLOT_KWARGS,
    **plot_kwargs
):
    """
    Create and save the plots related to IMRCT

    Parameters
    ----------
    imrct_deviations: ProbabilityDict2D
        Output of imrct_deviation_parameters_from_final_mass_final_spin
    samples: MultiAnalysisSamplesDict
        Dictionary containing inspiral and postinspiral samples
    webdir: str
        path to save the files
    """
    logger.info("Starting to generate plots")
    evolve_spins_string = "evolved"
    if not evolve_spins:
        evolve_spins_string = "non_" + evolve_spins_string

    samples_string = "final_{}_" + evolve_spins_string
    plotdir = os.path.join(webdir, "plots")
    make_dir(plotdir)
    if plot_label is not None:
        base_string = os.path.join(plotdir, "%s_imrct_{}.png" % (plot_label))
    else:
        base_string = os.path.join(plotdir, "imrct_{}.png")
    logger.debug("Creating IMRCT deviations triangle plot")
    _plot_kwargs = _default_plot_kwargs.copy()
    _plot_kwargs.update(
        {
            "cmap": cmap,
            "levels": levels,
            "level_kwargs": level_kwargs,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }
    )
    _plot_kwargs.update(plot_kwargs)
    fig, _ax1, ax_2d, _ax3 = imrct_deviations.plot(
        "final_mass_final_spin_deviations",
        **_plot_kwargs,
    )
    ax_2d.plot(0, 0, "k+", ms=12, mew=2)
    if save:
        fig.savefig(base_string.format("deviations_triangle_plot"), bbox_inches="tight")
        fig.close()
    if return_fig:
        return [fig, _ax1, ax_2d, _ax3]
    logger.debug("Finished creating IMRCT deviations triangle plot.")

    if make_diagnostic_plots:
        logger.info("Creating diagnostic plots")
        plot_kwargs = _default_plot_kwargs.copy()
        plot_kwargs.update(
            {"fill_alpha": 0.2, "labels": [inspiral_string, postinspiral_string]}
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
        logger.debug("Finished creating diagnostic plots")
    logger.info("Finished generating plots")


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
    import time

    _running = True
    while _running:
        if process.poll() is not None:
            _running = False
            if process.returncode != 0:
                msg = "The PE job: {} failed. The error is: {}"
                _output, _error = process.communicate()
                raise ValueError(msg.format(command_line, _error.decode("utf-8")))
        time.sleep(20)
    logger.info("PE page generation complete")


def add_dynamic_kwargs_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic calibration argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    command_line: str, optional
        The command line which you are passing. Default None
    """
    from pesummary.gw.command_line import add_dynamic_argparse

    return add_dynamic_argparse(
        existing_namespace, "--*_kwargs", example="--{}_kwargs", command_line=command_line
    )


class parser(gw_parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """

    def __init__(self, existing_parser=None):
        super(parser, self).__init__(existing_parser=existing_parser)

    @property
    def dynamic_argparse(self):
        return [add_dynamic_kwargs_to_namespace]


def main(args=None):
    """Top level interface for `summarytgr`"""
    CHECKUP = False
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    make_dir(opts.webdir)
    base_url = guess_url(
        os.path.abspath(opts.webdir), socket.getfqdn(), os.environ["USER"]
    )
    evolve_spins_string = "evolved"
    if not opts.evolve_spins:
        evolve_spins = opts.evolve_spins
        evolve_spins_string = "non_" + evolve_spins_string
    else:
        evolve_spins = "ISCO"

    open_files_paths = {
        _label: read(path) for _label, path in zip(opts.labels, opts.samples)
    }
    open_files = MultiAnalysisSamplesDict(
        {_label: open_files_paths[_label].samples_dict for _label in opts.labels}
    )
    test_key_data = {}
    if opts.test == "imrct":
        test_kwargs = opts.imrct_kwargs.copy()
        for key, value in test_kwargs.items():
            try:
                test_kwargs[key] = ast.literal_eval(value)
            except ValueError:
                pass
        if len(opts.labels) % 2 != 0:
            raise ValueError(
                "The IMRCT test requires 2 results files for each analysis. "
            )
        elif len(opts.labels) > 2:
            cond = all(
                ":inspiral" in label or ":postinspiral" in label for label in opts.labels
            )
            if not cond:
                raise ValueError(
                    "To compare 2 or more analyses, please provide labels as "
                    "'{}:inspiral' and '{}:postinspiral' where {} indicates "
                    "the analysis label"
                )
            else:
                analysis_label = [
                    label.split(":inspiral")[0]
                    for label in opts.labels
                    if ":inspiral" in label and "post" not in label
                ]
                if len(analysis_label) != len(opts.samples) / 2:
                    raise ValueError(
                        "When comparing more than 2 analyses, labels must "
                        "be of the form '{}:inspiral' and '{}:postinspiral'."
                    )
                logger.info(
                    "Using the labels: {} to distinguish analyses".format(
                        ", ".join(analysis_label)
                    )
                )
        elif sorted(opts.labels) != ["inspiral", "postinspiral"]:
            raise ValueError(
                "The IMRCT test requires an inspiral and postinspiral result "
                "file. Please indicate which file is the inspiral and which "
                "is postinspiral by providing these exact labels to the "
                "summarytgr executable"
            )
        else:
            analysis_label = ["primary"]
        for _arg in ["cutoff_frequency", "approximant", "links_to_pe_pages"]:
            _attr = getattr(opts, _arg)
            if _attr is not None and len(_attr) and len(_attr) != len(opts.labels):
                raise ValueError("Please provide a {} for each file".format(_arg))
        test_key_data["imrct"] = {}
        fits_data = {}
        for key, sample in open_files.items():
            if "final_mass_{}".format(evolve_spins_string) not in sample.keys():
                logger.info("Remnant properties not in samples, trying to generate them")
                returned_extra_kwargs = sample.generate_all_posterior_samples(
                    evolve_spins=evolve_spins, return_kwargs=True
                )
                converted_keys = sample.keys()
                if "final_mass_{}".format(evolve_spins_string) not in converted_keys:
                    raise KeyError(
                        "Remnant properties not in samples and cannot be generated"
                    )
                else:
                    logger.info("Remnant properties generated.")
                    for fit in ["final_mass_NR_fits", "final_spin_NR_fits"]:
                        fits_data["{} {}".format(key, fit)] = returned_extra_kwargs[
                            "meta_data"
                        ][fit]

        inspiral_keys = [
            key for key in open_files.keys() if "inspiral" in key and "post" not in key
        ]
        postinspiral_keys = [
            key.replace("inspiral", "postinspiral") for key in inspiral_keys
        ]
        _imrct_deviations = []
        cmap_cycle = cycle(conf.cmapcycle)
        for num, _inspiral in enumerate(inspiral_keys):
            _postinspiral = postinspiral_keys[num]
            _samples = open_files[[_inspiral, _postinspiral]]
            imrct_deviations, data = generate_imrct_deviation_parameters(
                _samples,
                evolve_spins=opts.evolve_spins,
                inspiral_string=_inspiral,
                postinspiral_string=_postinspiral,
                **test_kwargs,
            )
            data.update(fits_data)
            _imrct_deviations.append(imrct_deviations)
            _legend_kwargs = {}
            if len(analysis_label) > 1:
                _legend_kwargs = {
                    "legend": True,
                    "label": analysis_label[num],
                    "legend_kwargs": {"frameon": True},
                }
            make_imrct_plots(
                imrct_deviations,
                _samples,
                plot_label=analysis_label[num],
                webdir=opts.webdir,
                cmap=next(cmap_cycle),
                evolve_spins=opts.evolve_spins,
                make_diagnostic_plots=opts.make_diagnostic_plots,
                inspiral_string=_inspiral,
                postinspiral_string=_postinspiral,
                **_legend_kwargs,
            )
            frequency_dict = dict()
            approximant_dict = dict()
            zipped = zip(
                [opts.cutoff_frequency, opts.approximant],
                [frequency_dict, approximant_dict],
            )
            _inspiral_string = inspiral_keys[num]
            _postinspiral_string = postinspiral_keys[num]
            for _list, _dict in zipped:
                if _list is not None and len(_list) == len(opts.labels):
                    inspiral_ind = opts.labels.index(_inspiral_string)
                    postinspiral_ind = opts.labels.index(_postinspiral_string)
                    _dict["inspiral"] = _list[inspiral_ind]
                    _dict["postinspiral"] = _list[postinspiral_ind]
                elif _list is not None:
                    raise ValueError(
                        "Please provide a 'cutoff_frequency' and 'approximant' "
                        "for each file"
                    )
                else:
                    try:
                        if _list == opts.cutoff_frequency:
                            _dict["inspiral"] = float(
                                open_files[_inspiral_string].config["maximum-frequency"]
                            )
                            _dict["postinspiral"] = float(
                                open_files[_postinspiral_string].config[
                                    "minimum-frequency"
                                ]
                            )
                        elif _list == opts.approximant:
                            _dict["inspiral"] = float(
                                open_files[_inspiral_string].config["config"]
                            )
                            _dict["postinspiral"] = float(
                                open_files[_postinspiral_string].config["config"]
                            )
                    except (AttributeError, KeyError):
                        _dict["inspiral"] = None
                        _dict["postinspiral"] = None

            data["inspiral maximum frequency (Hz)"] = frequency_dict["inspiral"]
            data["postinspiral mininum frequency (Hz)"] = frequency_dict["postinspiral"]

            for key in ["inspiral", "postinspiral"]:
                data["{} approximant".format(key)] = approximant_dict[key]

            desired_metadata_order = [
                "GR Quantile (%)",
                "inspiral maximum frequency (Hz)",
                "postinspiral mininum frequency (Hz)",
                "inspiral final_mass_NR_fits",
                "postinspiral final_mass_NR_fits",
                "inspiral final_spin_NR_fits",
                "postinspiral final_spin_NR_fits",
                "inspiral approximant",
                "postinspiral approximant",
                "evolve_spin",
                "N_bins",
                "Time (seconds)",
            ]

            desired_metadata = {}

            for key in desired_metadata_order:
                try:
                    desired_metadata[key] = data[key]
                except KeyError:
                    continue

            test_key_data["imrct"][analysis_label[num]] = desired_metadata

        if len(inspiral_keys) > 1:
            fig = None
            colors = cycle(conf.colorcycle)
            for num, _samples in enumerate(_imrct_deviations):
                save = False
                if num == len(_imrct_deviations) - 1:
                    save = True
                fig = make_imrct_plots(
                    _samples,
                    {},
                    webdir=opts.webdir,
                    evolve_spins=opts.evolve_spins,
                    make_diagnostic_plots=False,
                    plot_label="combined",
                    cmap="off",
                    levels=[0.95],
                    level_kwargs={"colors": [next(colors)]},
                    existing_figure=fig,
                    save=save,
                    return_fig=True,
                    legend=True,
                    label=analysis_label[num],
                    legend_kwargs={"frameon": True},
                )
        if not len(opts.links_to_pe_pages):
            try:
                links_to_pe_pages = [
                    open_files_paths[_label].history["webpage_url"]
                    for _label in opts.labels
                ]
            except (AttributeError, KeyError, TypeError):
                links_to_pe_pages = []
        else:
            links_to_pe_pages = opts.links_to_pe_pages

        if not len(links_to_pe_pages) and not opts.disable_pe_page_generation:
            CHECKUP = True
            _webdir = os.path.join(opts.webdir, "pe_pages")
            _command_line, process = generate_pe_pages(
                _webdir, opts.samples, opts.labels, opts.pe_page_options
            )
            links_to_pe_pages = [
                "../pe_pages/html/{0}_{0}.html".format(label) for label in opts.labels
            ]
    samplesdir = os.path.join(opts.webdir, "samples")
    make_dir(samplesdir)
    TGRMetaFile(
        open_files,
        analysis_label,
        webdir=opts.webdir,
        imrct_data={
            label: _imrct_deviations[num] for num, label in enumerate(analysis_label)
        },
        file_kwargs=test_key_data,
    )
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
    msg = "Complete. Webpages can be viewed at the following url {}.".format(
        base_url + "/home.html"
    )
    if CHECKUP:
        msg += (
            "The PE webpages are still being generated. These links will not "
            "work currently"
        )
    logger.info(msg)
    if CHECKUP:
        check_on_pe_page_generate(command_line, process)


if __name__ == "__main__":
    main()
