#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
import argparse
from itertools import cycle
from pesummary import conf
from pesummary.gw.parser import TGRparser
from pesummary.gw.webpage.tgr import TGRWebpageGeneration
from pesummary.gw.file.meta_file import TGRMetaFile
from pesummary.gw.plots.tgr import make_and_save_imrct_plots
from pesummary.gw.conversions.tgr import generate_imrct_deviation_parameters
from pesummary.io import read
from pesummary.utils.utils import logger

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Aditya Vijaykumar <aditya.vijaykumar@ligo.org>",
]
__doc__ = """This executable is used to post-process and generate webpages to
display results from analyses which test the General Theory of Relativity"""
TESTS = ["imrct"]


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
    )
    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        help="What test do you want to run? Currently only supports {}".format(
            ", ".join(TESTS)
        ),
        required=True,
        choices=TESTS,
    )
    parser.add_argument(
        "--{test}_kwargs",
        dest="example_test_kwargs",
        help=(
            "Kwargs you wish to use when postprocessing the results. Kwargs "
            "should be provided as a dictionary 'kwarg:value'. For example "
            "`--imrct_kwargs N_bins:201 multi_process:4` would pass the kwargs "
            "N_bins=201, multi_process=4 to the IMRCT function. The test name "
            "'{test}' should match the test provided with the --test flag"
        ),
        default=None,
    )
    parser.add_argument(
        "-s",
        "--samples",
        dest="samples",
        help=(
            "Path to posterior samples file(s). See documentation for allowed "
            "formats. If path is on a remote server, add username and "
            "servername in the form {username}@{servername}:{path}"
        ),
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        help=(
            "Labels used to distinguish runs. The label format is dependent "
            "on the TGR test you wish to use. For the IMRCT test, labels "
            "need to be inspiral and postinspiral if analysing a single event "
            "or {label1}:inspiral,{label1}:postinspiral,{label2}:inspiral,"
            "{label2}:postinspiral,... if analysing two or more events (where "
            "label1/label2 is a unique string to distinguish files from a "
            "single event). If a metafile is provided, labels need to be "
            "{inspiral_label}:inspiral {postinspiral_label}:postinspiral "
            "where inspiral_label and postinspiral_label are the "
            "pesummary labels for the inspiral and postinspiral analyses "
            "respectively."
        ),
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
        "--f_low", dest="f_low", help=(
            "Low frequency cutoff used to generate the samples. Only used when "
            "evolving spins"
        ), nargs='+', default=None
    )
    parser.add_argument(
        "--cutoff_frequency",
        dest="cutoff_frequency",
        help="Cutoff Frequency for IMRCT. Overrides any cutoff frequency "
        "present in the supplied files. "
        "The supplied cutoff frequency will only be used as metadata and "
        "does not affect the cutoff frequency used in the analysis. "
        "If only one number is supplied, the inspiral maximum frequency "
        "and the postinspiral maximum frequency are set to the same number. "
        "If a list of length 2 is supplied, this assumes that the "
        "one corresponding to the inspiral label is the maximum frequency "
        "for the inspiral and that corresponding to the postinspiral label is "
        "the minimum frequency for the postinspiral",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--links_to_pe_pages",
        dest="links_to_pe_pages",
        help="URLs for PE results pages separated by spaces.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--disable_pe_page_generation",
        action="store_true",
        help=(
            "Disable PE page generation for the input samples. This option is "
            "only relevant if no URLs for PE results pages are provided using "
            "--links_to_pe_pages."
        ),
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

    logger.info("Creating PE summarypages")
    base_command_line = (
        "summarypages --webdir {} --samples {} --labels {} --gw ".format(
            webdir, " ".join(result_files), " ".join(labels)
        )
    )
    base_command_line += additional_options
    launch(base_command_line, check_call=True)
    logger.info("PE summarypages created")
    return


def imrct(opts):
    """Postprocess the IMR consistency test results

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options

    Returns
    -------
    args: pesummary.gw.inputs.IMRCTInput
        IMRCTInput object containing the command line arguments
    data: list
        a list of length 3 containing a dictionary of key data associated with
        the IMR consistency test results, a list containing links to PE pages
        displaying the posterior samples for the inspiral and postinspiral
        analyses and a list containing the IMRCT deviation PDFs for each
        analysis
    """
    from pesummary.gw.inputs import IMRCTInput

    args = IMRCTInput(opts)
    test_key_data = {}
    fits_data = {}
    _imrct_deviations = []
    cmap_cycle = cycle(conf.cmapcycle)
    evolved = np.ones_like(args.inspiral_keys, dtype=bool)
    for num, _inspiral in enumerate(args.inspiral_keys):
        _postinspiral = args.postinspiral_keys[num]
        _samples = args.samples[[_inspiral, _postinspiral]]
        args.imrct_kwargs.update({
            prop: {
                _key: (
                    getattr(args, prop)[args.labels.index(_key)] if
                    getattr(args, prop) is not None else None
                ) for _key in [_inspiral, _postinspiral]
            } for prop in ["approximant", "f_low"]
        })
        imrct_deviations, data, _evolved, samples = generate_imrct_deviation_parameters(
            _samples,
            evolve_spins_forward=opts.evolve_spins,
            inspiral_string=_inspiral,
            postinspiral_string=_postinspiral,
            return_samples_used=True,
            **args.imrct_kwargs,
        )
        for key, value in samples.items():
            args.samples[key] = value
        evolved[num] = _evolved
        data.update(fits_data)
        _imrct_deviations.append(imrct_deviations)
        _legend_kwargs = {}
        if len(args.analysis_label) > 1:
            _legend_kwargs = {
                "legend": True,
                "label": args.analysis_label[num],
                "legend_kwargs": {"frameon": True},
            }
        logger.info("Starting to generate plots")
        make_and_save_imrct_plots(
            imrct_deviations,
            samples=_samples,
            plot_label=args.analysis_label[num],
            webdir=args.webdir,
            cmap=next(cmap_cycle),
            evolve_spins=evolved[num],
            make_diagnostic_plots=opts.make_diagnostic_plots,
            inspiral_string=_inspiral,
            postinspiral_string=_postinspiral,
            **_legend_kwargs,
        )
        logger.info("Finished generating plots")
        _keys = [
            "inspiral maximum frequency (Hz)",
            "postinspiral minimum frequency (Hz)",
            "inspiral approximant",
            "postinspiral approximant",
            "inspiral remnant fits",
            "postinspiral remnant fits"
        ]
        for key in _keys:
            if "remnant" not in key:
                data[key] = args.meta_data[args.analysis_label[num]][key]
            else:
                _meta_data = args.meta_data[args.analysis_label[num]][key]
                _fits = [
                    "final_mass_NR_fits", "final_spin_NR_fits"
                ]
                prefix = key.split(" remnant fits")[0]
                for _fit in _fits:
                    if _meta_data is not None and _fit in _meta_data.keys():
                        data["{} {}".format(prefix, _fit)] = _meta_data[_fit]

        desired_metadata_order = [
            "GR Quantile (%)",
            "inspiral maximum frequency (Hz)",
            "postinspiral minimum frequency (Hz)",
            "inspiral final_mass_NR_fits",
            "postinspiral final_mass_NR_fits",
            "inspiral final_spin_NR_fits",
            "postinspiral final_spin_NR_fits",
            "inspiral approximant",
            "postinspiral approximant",
            "evolve_spins",
            "N_bins",
            "Time (seconds)",
        ]

        desired_metadata = {}
        for key in desired_metadata_order:
            try:
                desired_metadata[key] = data[key]
            except KeyError:
                continue
        test_key_data[args.analysis_label[num]] = desired_metadata
    if len(args.inspiral_keys) > 1:
        fig = None
        colors = cycle(conf.colorcycle)
        for num, _samples in enumerate(_imrct_deviations):
            save = False
            if num == len(_imrct_deviations) - 1:
                save = True
            fig = make_and_save_imrct_plots(
                _samples,
                samples={},
                webdir=args.webdir,
                evolve_spins=evolved[num],
                make_diagnostic_plots=False,
                plot_label="combined",
                cmap="off",
                levels=[0.95],
                level_kwargs={"colors": [next(colors)]},
                existing_figure=fig,
                save=save,
                return_fig=True,
                legend=True,
                label=args.analysis_label[num],
                legend_kwargs={"frameon": True},
            )
    if not len(args.links_to_pe_pages):
        try:
            links_to_pe_pages = [
                args.samples_paths[_label].history["webpage_url"]
                for _label in args.labels
            ]
        except (AttributeError, KeyError, TypeError):
            links_to_pe_pages = []
    else:
        links_to_pe_pages = args.links_to_pe_pages

    if not len(links_to_pe_pages) and not opts.disable_pe_page_generation:
        links_to_pe_pages = [
            "../pe_pages/html/{0}_{0}.html".format(label) for label in
            args.labels
        ]
    return args, [test_key_data, links_to_pe_pages, _imrct_deviations]


def main(args=None):
    """Top level interface for `summarytgr`"""
    _parser = TGRparser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    test_key_data = {}
    if opts.test == "imrct":
        args, _data = imrct(opts)
        test_key_data["imrct"], links_to_pe_pages, _imrct_deviations = _data
    samplesdir = os.path.join(args.webdir, "samples")
    TGRMetaFile(
        args.samples,
        args.analysis_label,
        webdir=opts.webdir,
        imrct_data={
            label: _imrct_deviations[num] for num, label in enumerate(
                args.analysis_label
            )
        },
        file_kwargs=test_key_data,
    )
    test_key_data_for_webpage = test_key_data
    for key in test_key_data_for_webpage["imrct"].keys():
        test_key_data_for_webpage["imrct"][key]["GR Quantile (%)"] = round(
            test_key_data_for_webpage["imrct"][key]["GR Quantile (%)"], 2
        )
    logger.info("Creating webpages for IMRCT")
    webpage = TGRWebpageGeneration(
        args.webdir,
        args.result_files,
        test=opts.test,
        open_files=args.samples,
        links_to_pe_pages=links_to_pe_pages,
        test_key_data=test_key_data_for_webpage,
    )
    webpage.generate_webpages(make_diagnostic_plots=opts.make_diagnostic_plots)
    msg = "Complete. Webpages can be viewed at the following url {}.".format(
        args.baseurl + "/home.html"
    )
    if not opts.disable_pe_page_generation:
        msg += (
            " The PE webpages are about to be generated. These links will not "
            "work currently"
        )
        logger.info(msg)
        _webdir = os.path.join(args.webdir, "pe_pages")
        _ = generate_pe_pages(
            _webdir, args.result_files, args.labels, opts.pe_page_options
        )
    else:
        logger.info(msg)


if __name__ == "__main__":
    main()
