# Licensed under an MIT style license -- see LICENSE.md

import os
from pesummary.core.webpage.main import _WebpageGeneration

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Aditya Vijaykumar <aditya.vijaykumar@ligo.org>",
]
TESTS = ["imrct"]


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
        links_to_pe_pages=[],
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
            *args,
            webdir=webdir,
            labels=_labels,
            user=os.environ["USER"],
            samples=open_files,
            **kwargs
        )
        self.labels = labels
        self.file_versions = {
            label: "No version information found" for label in self.labels
        }
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
        if len(self.links_to_pe_pages):
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
        html_file = self.make_modal_carousel(
            html_file, image_contents=image_contents, unique_id=True
        )
        _banner_desc = (
            "Below we show summary statistics associated with each test of GR"
        )
        html_file.make_banner(
            approximant="Summary Statistics",
            key="content",
            content=_banner_desc,
            _style="font-size: 26px;",
        )
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        html_file.make_container(style=_style)
        html_file.make_div(4, _class=_class, _style=None)
        base_label = self.labels[0]
        total_keys = list(self.test_key_data[base_label].keys())
        if len(self.labels) > 1:
            for _label in self.labels[1:]:
                total_keys += [
                    key
                    for key in self.test_key_data[_label].keys()
                    if key not in total_keys
                ]
        table_contents = [
            [i]
            + [
                self.test_key_data[i][key]
                if key in self.test_key_data[i].keys()
                else "-"
                for key in total_keys
            ]
            for i in self.labels
        ]
        _headings = [" "] + total_keys
        html_file.make_table(
            headings=_headings,
            format="table-hover",
            heading_span=1,
            contents=table_contents,
            accordian=False,
        )
        html_file.end_div(4)
        html_file.end_container()
        html_file.export("{}.csv".format("summary_of_tests_of_GR.csv"))
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        for key, value in self.input_file_summary.items():
            html_file.make_banner(
                approximant="{} Summary Table".format(key),
                key="summary_table",
                _style="font-size: 26px;",
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
        html_file = self.setup_page(
            "imrct", self.navbar["result_page"], title="IMR Consistency Test"
        )
        html_file.make_banner(
            approximant="IMR Consistency Test", key="content", content=" "
        )
        desc = "Below we show the executive plots for the IMR consistency test"
        html_file.make_banner(
            approximant="Executive plots",
            key="content",
            content=desc,
            _style="font-size: 26px;",
        )
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
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        html_file.make_container(style=_style)
        html_file.make_div(4, _class=_class, _style=None)
        _data = self.test_key_data["imrct"]
        table_contents = [list(_data.values())]
        html_file.make_table(
            headings=list(_data.keys()),
            format="table-hover",
            heading_span=1,
            contents=table_contents,
            accordian=False,
        )
        html_file.end_div(4)
        html_file.end_container()
        html_file.export("{}.csv".format("summary_of_IMRCT_test.csv"))
        if make_diagnostic_plots:
            desc = (
                "Below we show additional plots generated for the IMR consistency "
                "test"
            )
            html_file.make_banner(
                approximant="Additional plots",
                key="content",
                content=desc,
                _style="font-size: 26px;",
            )
            image_contents = [
                [
                    base_string.format("final_mass_non_evolved_final_spin_non_evolved"),
                    base_string.format("mass_1_mass_2"),
                    base_string.format("a_1_a_2"),
                ],
            ]
            _base = (
                "2D posterior distribution for {} estimated from the inspiral "
                "and post-inspiral parts of the signal"
            )
            captions = [
                [
                    _base.format("final_mass and final_spin"),
                    _base.format("mass_1 and mass_2"),
                    _base.format("a_1 and a_2"),
                ],
            ]
            cli = [[" ", " ", " "]]
            html_file = self.make_modal_carousel(
                html_file,
                image_contents,
                captions=captions,
                cli=cli,
                unique_id=True,
                autoscale=True,
            )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()
