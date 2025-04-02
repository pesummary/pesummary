#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import pesummary
from pesummary.core.cli.inputs import _Input
from pesummary.gw.file.read import read as GWRead
from pesummary.gw.classification import EMBright, PAstro
from pesummary.utils.utils import make_dir, logger
from pesummary.utils.exceptions import InputError
from pesummary.core.cli.parser import ArgumentParser as _ArgumentParser

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to generate a txt file containing the
source classification probailities"""


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--plot": {
                    "choices": ["bar"],
                    "default": "bar",
                    "help": "Name of the plot you wish to make",
                },
                "--pastro_category_file": {
                    "default": None,
                    "help": (
                        "path to yml file containing summary data for each "
                        "category (BBH, BNS, NSBH). This includes e.g. rates, "
                        "mass bounds etc. This is used when computing PAstro"
                    )
                },
                "--terrestrial_probability": {
                    "default": None,
                    "help": (
                        "Terrestrial probability for the candidate you are "
                        "analysing. This is used when computing PAstro"
                    ),
                },
                "--catch_terrestrial_probability_error": {
                    "default": False,
                    "action": "store_true",
                    "help": (
                        "Catch the ValueError raised when no terrestrial "
                        "probability is provided when computing PAstro"
                    ),
                    "key": "gw",
                },
            },
        )
        return options


def generate_probabilities(
    result_files, classification_file, terrestrial_probability,
    catch_terrestrial_probability_error
):
    """Generate the classification probabilities

    Parameters
    ----------
    result_files: list
        list of result files
    """
    classifications = []
    _func = "classification"
    _kwargs = {}

    for num, i in enumerate(result_files):
        mydict = {}
        if not _Input.is_pesummary_metafile(i):
            mydict = getattr(
                EMBright, "{}_from_file".format(_func)
            )(i, **_kwargs)
            em_bright = getattr(PAstro, "{}_from_file".format(_func))(
                i, category_data=classification_file,
                terrestrial_probability=terrestrial_probability,
                catch_terrestrial_probability_error=catch_terrestrial_probability_error,
                **_kwargs
            )
        else:
            f = GWRead(i)
            label = f.labels[0]
            mydict = getattr(
                 EMBright(f.samples_dict[label]), _func
            )(**_kwargs)
            em_bright = getattr(
                PAstro(
                    f.samples_dict[label],
                    category_data=classification_file,
                    terrestrial_probability=terrestrial_probability,
                    catch_terrestrial_probability_error=catch_terrestrial_probability_error
                ), _func
            )(**_kwargs)
        mydict.update(em_bright)
        classifications.append(mydict)
    return classifications


def save_classifications(savedir, classifications, labels):
    """Read and return a list of parameters and samples stored in the result
    files

    Parameters
    ----------
    result_files: list
        list of result files
    classifications: dict
        dictionary of classification probabilities
    """
    import os
    import json

    base_path = os.path.join(savedir, "{}_pe_classification.json")
    for num, i in enumerate(classifications):
        for prior in i.keys():
            with open(base_path.format(labels[num]), "w") as f:
                json.dump(i, f)


def make_plots(
    result_files, webdir=None, labels=None, plot_type="bar",
    probs=None
):
    """Save the plots generated by EMBright

    Parameters
    ----------
    result_files: list
        list of result files
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

    for num, i in enumerate(result_files):
        if labels is None:
            label = num
        else:
            label = labels[num]
        f = GWRead(i)
        if not isinstance(f, pesummary.gw.file.formats.pesummary.PESummary):
            f.generate_all_posterior_samples()
        if plot_type == "bar":
            from pesummary.gw.plots.plot import _classification_plot
            fig = _classification_plot(probs[num])
            fig.savefig(
                os.path.join(
                    webdir,
                    "{}_pastro_bar.png".format(label)
                )
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")


def main(args=None):
    """Top level interface for `summarypublication`
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_known_options_to_parser(
        [
            "--webdir", "--samples", "--labels", "--plot",
            "--pastro_category_file", "--terrestrial_probability",
            "--catch_terrestrial_probability_error"
        ]
    )
    opts, _ = parser.parse_known_args(args=args)
    if opts.webdir:
        make_dir(opts.webdir)
    else:
        logger.warning(
            "No webdir given so plots will not be generated and "
            "classifications will be shown in stdout rather than saved to file"
        )
    classifications = generate_probabilities(
        opts.samples, opts.pastro_category_file, opts.terrestrial_probability,
        opts.catch_terrestrial_probability_error
    )
    if opts.labels is None:
        opts.labels = []
        for i in opts.samples:
            f = GWRead(i)
            if hasattr(f, "labels"):
                opts.labels.append(f.labels[0])
            else:
                raise InputError("Please provide a label for each result file")
    if opts.webdir:
        save_classifications(opts.webdir, classifications, opts.labels)
    else:
        print(classifications)
        return
    if opts.plot == "bar":
        probs = classifications
    else:
        probs = None
    make_plots(
        opts.samples, webdir=opts.webdir, labels=opts.labels,
        probs=probs
    )


if __name__ == "__main__":
    main()
