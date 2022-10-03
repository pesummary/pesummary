#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np
from itertools import cycle
from pesummary import conf
from pesummary.gw.cli.parser import TGRArgumentParser
from pesummary.gw.webpage.tgr import TGRWebpageGeneration
from pesummary.gw.file.meta_file import TGRMetaFile
from pesummary.gw.plots.tgr import make_and_save_imrct_plots
from pesummary.gw.conversions.tgr import generate_imrct_deviation_parameters
from pesummary.utils.utils import logger

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Aditya Vijaykumar <aditya.vijaykumar@ligo.org>",
]
__doc__ = """This executable is used to post-process and generate webpages to
display results from analyses which test the General Theory of Relativity"""


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
    args: pesummary.gw.cli.inputs.IMRCTInput
        IMRCTInput object containing the command line arguments
    data: list
        a list of length 3 containing a dictionary of key data associated with
        the IMR consistency test results, a list containing links to PE pages
        displaying the posterior samples for the inspiral and postinspiral
        analyses and a list containing the IMRCT deviation PDFs for each
        analysis
    """
    from pesummary.gw.cli.inputs import IMRCTInput

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
            evolve_spins_forward=opts.evolve_spins_forwards,
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
    parser = TGRArgumentParser(description=__doc__)
    parser.add_all_known_options_to_parser()
    opts, unknown = parser.parse_known_args(args=args)
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
