#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os
import copy
import pesummary
from pesummary.core.parser import parser
from pesummary.core.plots.main import _PlotGeneration
from pesummary.core.webpage.main import _WebpageGeneration
from pesummary.utils.utils import logger
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary import conf
from pesummary.io import read
import numpy as np
import argparse

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to compare multiple files"""
COMPARISON_PROPERTIES = [
    "posterior_samples", "config", "priors", "psds"
]
SAME_STRING = "The result files match for entry: '{}'"


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument(
        "-s", "--samples", dest="samples", default=None, nargs='+',
        help="Posterior samples hdf5 file"
    )
    _parser.add_argument(
        "--properties_to_compare", dest="compare", nargs='+',
        default=["posterior_samples"], help=(
            "list of properties you wish to compare between the files. Default "
            "posterior_samples"
        ), choices=COMPARISON_PROPERTIES
    )
    _parser.add_argument(
        "-w", "--webdir", dest="webdir", default=None, metavar="DIR",
        help="make page and plots in DIR."
    )
    _parser.add_argument(
        "--generate_comparison_page", action="store_true", default=False,
        help="Generate a comparison page to compare contents"
    )
    _parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print useful information for debugging purposes"
    )
    return _parser


def _comparison_string(path, values=None, _type=None):
    """Print the comparison string

    Parameters
    ----------
    path: list
        list containing the path to the dataset being compared
    values: list, optional
        list containing the entries which are being compared
    _type: optional
        the type of the values being compared
    """
    _path = "/".join(path)
    if values is None:
        string = "'{}' is not in both result files. Unable to compare".format(
            _path
        )
        logger.info(string)
    else:
        string = (
            "The result files differ for the following entry: '{}'. ".format(
                _path
            )
        )
        if _type == list:
            try:
                _diff = np.max(np.array(values[0]) - np.array(values[1]))
                string += "The maximum difference is: {}".format(_diff)
            except ValueError as e:
                if "could not be broadcast together" in str(e):
                    string += (
                        "Datasets contain different number of samples: "
                        "{}".format(
                            ", ".join([str(len(values[0])), str(len(values[1]))])
                        )
                    )
        else:
            string += "The entries are: {}".format(", ".join(values))

        logger.info(string)
    return string


def _compare(data, path=None):
    """Compare multiple posterior samples

    Parameters
    ----------
    data: list, dict
        data structure which is to be compared
    path: list
        path to the data structure being compared
    """
    if path is None:
        path = []

    string = ""
    if isinstance(data[0], dict):
        for key, value in data[0].items():
            _path = path + [key]
            if not all(key in _dict.keys() for _dict in data):
                string += "{}\n".format(_comparison_string(_path))
                continue
            if isinstance(value, dict):
                _data = [_dict[key] for _dict in data]
                string += "{}\n".format(_compare(_data, path=_path))
            else:
                string += "{}\n".format(
                    _compare_datasets([_data[key] for _data in data], path=_path)
                )
    else:
        string += "{}\n".format(_compare_datasets(data, path=path))
    return string


def _compare_datasets(data, path=[]):
    """Compare two datasets

    Parameters
    ----------
    data: list, str, int, float
        dataset which you want to compare
    path: list, optional
        path to the dataset being compared
    """
    array_types = (list, pesummary.utils.samples_dict.Array, np.ndarray)
    numeric_types = (float, int, np.number)
    string_types = (str, bytes)

    string = SAME_STRING.format("/".join(path))
    if isinstance(data[0], array_types):
        try:
            np.testing.assert_almost_equal(data[0], data[1])
            logger.debug(string)
        except AssertionError:
            string = _comparison_string(path, values=data, _type=list)
    elif isinstance(data[0], numeric_types):
        if not all(i == data[0] for i in data):
            string = _comparison_string(path, values=data, _type=float)
        else:
            logger.debug(string)
    elif isinstance(data[0], string_types):
        if not all(i == data[0] for i in data):
            string = _comparison_string(path, values=data, _type=str)
        else:
            logger.debug(string)
    else:
        raise ValueError(
            "Unknown data format. Unable to compare: {}".format(
                ", ".join([str(i) for i in data])
            )
        )
    return string


def compare(samples, properties_to_compare=COMPARISON_PROPERTIES):
    """Compare multiple posterior samples

    Parameters
    ----------
    samples: list
        list of files you wish to compare
    properties_to_compare: list, optional
        optional list of properties you wish to compare
    """
    data = [read(path, disable_prior_conversion=True) for path in samples]
    string = ""
    for prop in properties_to_compare:
        if prop.lower() == "posterior_samples":
            prop = "samples_dict"
        _data = [
            getattr(f, prop) if hasattr(f, prop) else False for f in
            data
        ]
        if False in _data:
            logger.warning(
                "Unable to compare the property '{}' because not all files "
                "share this property".format(prop)
            )
            continue
        string += "{}\n\n".format(_compare(_data, path=[prop]))
    return string


class ComparisonPlots(_PlotGeneration):
    """Class to handle the generation of comparison plots
    """
    def __init__(self, webdir, samples, *args, **kwargs):
        logger.info("Starting to generate comparison plots")
        parameters = [list(samples[key]) for key in samples.keys()]
        params = list(set.intersection(*[set(l) for l in parameters]))
        linestyles = ["-"] * len(samples.keys())
        colors = list(conf.colorcycle)
        super(ComparisonPlots, self).__init__(
            *args, webdir=webdir, labels=list(samples.keys()), samples=samples,
            same_parameters=params, injection_data={
                label: {param: float("nan") for param in params} for label
                in list(samples.keys())
            }, linestyles=linestyles, colors=colors, **kwargs
        )

    def generate_plots(self):
        """Generate all plots for all result files
        """
        self._generate_comparison_plots()


class ComparisonWebpage(_WebpageGeneration):
    """Class to handle the generation of comparison plots
    """
    def __init__(self, webdir, samples, *args, comparison_string="", **kwargs):
        logger.info("Starting to generate comparison pages")
        parameters = [list(samples[key]) for key in samples.keys()]
        params = list(set.intersection(*[set(l) for l in parameters]))
        self.comparison_string = comparison_string
        super(ComparisonWebpage, self).__init__(
            *args, webdir=webdir, labels=list(samples.keys()), samples=samples,
            user=os.environ["USER"], same_parameters=params, **kwargs
        )
        self.copy_css_and_js_scripts()

    def generate_webpages(self):
        """Generate all webpages
        """
        self.make_home_pages()
        self.make_comparison_string_pages()
        self.make_comparison_pages()
        self.make_version_page()
        self.make_about_page()
        self.make_logging_page()
        try:
            self.generate_specific_javascript()
        except Exception:
            pass

    def make_navbar_for_homepage(self):
        """Make a navbar for the homepage
        """
        return ["home", "Logging", "Version"]

    def make_navbar_for_comparison_page(self):
        """Make a navbar for the comparison homepage
        """
        links = ["1d Histograms", ["Custom", "All"]]
        for i in self.categorize_parameters(self.same_parameters):
            links.append(i)
        final_links = ["home", "Output", links]
        return final_links

    def make_comparison_string_pages(self):
        """
        """
        self.create_blank_html_pages(["Comparison_Output"], stylesheets=["Output"])
        html_file = self.setup_page(
            "Comparison_Output", self.navbar["comparison"],
            approximant="Comparison", title="Summarycompare output"
        )
        html_file.make_div(indent=2, _class='banner', _style=None)
        html_file.add_content("Summarycompare output")
        html_file.end_div()
        html_file.make_div(indent=2, _class='paragraph')
        html_file.add_content(
            "Below we show the output from summarycompare"
        )
        html_file.end_div()
        html_file.make_container()
        styles = html_file.make_code_block(
            language="bash", contents=self.comparison_string
        )
        html_file.end_container()
        with open("{}/css/Output.css".format(self.webdir), "w") as f:
            f.write(styles)
        html_file.make_footer(user=self.user, rundir=self.webdir,)
        html_file.close()

    def _make_home_pages(self, *args, **kwargs):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = self.setup_page("home", self.navbar["home"])
        html_file.add_content("<script>")
        html_file.add_content(
            "window.location.href = './html/Comparison.html'"
        )
        html_file.add_content("</script>")
        html_file.close()


def main(args=None):
    """Top level interface for `summarycompare`
    """
    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    string = compare(opts.samples, opts.compare)
    if opts.generate_comparison_page:
        open_files = [read(ff).samples_dict for ff in opts.samples]
        samples = {}
        for num, _samples in enumerate(open_files):
            if isinstance(_samples, MultiAnalysisSamplesDict):
                samples.update(
                    {
                        "{}_file_{}".format(ff, num): val for ff, val in
                        _samples.items()
                    }
                )
            else:
                samples.update({"file_{}".format(num): _samples})
        plots = ComparisonPlots(opts.webdir, samples)
        plots.generate_plots()
        webpage = ComparisonWebpage(
            opts.webdir, samples, comparison_string=string
        )
        webpage.generate_webpages()


if __name__ == "__main__":
    main()
