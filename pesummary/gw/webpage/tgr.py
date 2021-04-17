# Licensed under an MIT style license -- see LICENSE.md

import os
import glob
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

    @property
    def _metafile(self):
        return "tgr_samples.h5"

    def generate_webpages(self, make_diagnostic_plots=False):
        """Generate all webpages for all tests"""
        self.make_home_pages()
        if self.test == "imrct" or self.test == "all":
            self.make_imrct_pages(
                make_diagnostic_plots=make_diagnostic_plots
            )
        self.make_version_page()
        self.make_logging_page()
        self.make_about_page()
        self.make_downloads_page()
        self.generate_specific_javascript()

    def make_navbar_for_result_page(self):
        links = self.make_navbar_for_homepage()
        if self.test == "imrct" and len(self.links_to_pe_pages):
            link_format = "external:{}"
            if len(self.links_to_pe_pages) > 2:
                analysis_label = [
                    label.split(":inspiral")[0] for label in self.samples.keys()
                    if "inspiral" in label and "postinspiral" not in label
                ]
                _links = ["PE Pages"]
                for label in analysis_label:
                    inspiral_ind = self.labels.index(
                        "{}:inspiral".format(label)
                    )
                    postinspiral_ind = self.labels.index(
                        "{}:postinspiral".format(label)
                    )
                    _links.append(
                        [
                            label, [
                                {
                                    "inspiral": link_format.format(
                                        self.links_to_pe_pages[inspiral_ind]
                                    )
                                },
                                {
                                    "postinspiral": link_format.format(
                                        self.links_to_pe_pages[postinspiral_ind]
                                    )
                                }
                            ]
                        ]
                    )
                links.insert(2, _links)
            else:
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
        html_file.make_banner(
            "Tests of General Relativity", custom=" ", key="custom"
        )
        imrct_plots = glob.glob(
            "{}/plots/combined_imrct_deviations_triangle_plot.png".format(
                self.webdir
            )
        )
        if len(imrct_plots):
            image_contents = [
                [_plot.replace(self.webdir, ".") for _plot in imrct_plots]
            ]
        else:
            image_contents = [["plots/primary_imrct_deviations_triangle_plot.png"]]
        html_file = self.make_modal_carousel(
            html_file, image_contents=image_contents, unique_id=True
        )
        _banner_desc = (
            "Below we show summary statistics associated with each test of GR"
        )
        html_file.make_banner(
            approximant="Summary Statistics",
            key="custom",
            custom=_banner_desc,
            _style="font-size: 26px;",
        )
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        _include = {
            "imrct": ["GR Quantile (%)"]
        }
        for label in self.labels:
            html_file.make_container(style=_style)
            html_file.make_div(4, _class=_class, _style=None)
            analyses = list(self.test_key_data[label].keys())
            table_contents = [
                [i]
                + [
                    self.test_key_data[label][i][key]
                    if key in self.test_key_data[label][i].keys()
                    else "-"
                    for key in _include[label]
                ] for i in analyses
            ]
            _headings = [label] + _include[label]
            html_file.make_table(
                headings=_headings,
                format="table-hover",
                heading_span=1,
                contents=table_contents,
                accordian=False,
            )
            html_file.end_div(4)
            html_file.end_container()
            html_file.export("{}.csv".format("summary_of_{}.csv".format(label)))
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_imrct_pages(self, make_diagnostic_plots=False):
        """Make the IMR consistency test pages"""
        analysis_label = [
            label.split(":inspiral")[0] for label in self.samples.keys() if
            "inspiral" in label and "postinspiral" not in label
        ]
        if analysis_label == ["inspiral"]:
            analysis_label = ["primary"]
        pages = ["imrct"]
        self.create_blank_html_pages(pages)
        html_file = self.setup_page(
            "imrct", self.navbar["result_page"], title="IMR Consistency Test"
        )
        html_file.make_banner(
            approximant="IMR Consistency Test", key="custom", custom=" "
        )
        desc = "Below we show the executive plots for the IMR consistency test"
        html_file.make_banner(
            approximant="Executive plots",
            key="custom",
            custom=desc,
            _style="font-size: 26px;",
        )
        path = self.image_path["other"]
        base_string = path + "{}_imrct_{}.png"
        image_contents = [
            path + base_string.format(analysis_label[num], "deviations_triangle_plot")
            for num in range(len(analysis_label))
        ]
        image_contents = [
            image_contents[i:2 + i] for i in range(0, len(image_contents), 2)
        ]
        captions = [
            (
                "This triangle plot shows the 2D and marginalized 1D "
                "posterior distributions for the fractional parameters "
                "fractional_final_mass and fractional_final_spin for the "
                "{} analysis. The prediction from General Relativity is "
                "shown as a plus."
            ).format(analysis_label[num]) for num in range(len(analysis_label))
        ]
        captions = [captions[i:2 + i] for i in range(0, len(captions), 2)]
        html_file = self.make_modal_carousel(
            html_file,
            image_contents,
            captions=captions,
            cli=None,
            unique_id=True,
            extra_div=True,
            autoscale=False,
        )
        _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
        _class = "row justify-content-center"
        html_file.make_container(style=_style)
        html_file.make_div(4, _class=_class, _style=None)
        _data = self.test_key_data["imrct"]
        headings = [" "] + list(_data[analysis_label[0]].keys())
        table_contents = [
            [_label] + list(_data[_label].values()) for _label in analysis_label
        ]
        html_file.make_table(
            headings=[" "] + list(_data[analysis_label[0]].keys()),
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
                key="custom",
                custom=desc,
                _style="font-size: 26px;",
            )
            image_contents = [
                [
                    base_string.format(_label, "final_mass_non_evolved_final_spin_non_evolved"),
                    base_string.format(_label, "mass_1_mass_2"),
                    base_string.format(_label, "a_1_a_2"),
                ] for _label in analysis_label
            ]
            _base = (
                "2D posterior distribution for {} estimated from the inspiral "
                "and post-inspiral parts of the signal for analysis: {}"
            )
            captions = [
                [
                    _base.format("final_mass and final_spin", _label),
                    _base.format("mass_1 and mass_2", _label),
                    _base.format("a_1 and a_2", _label),
                ] for _label in analysis_label
            ]
            cli = None
            html_file = self.make_modal_carousel(
                html_file,
                image_contents,
                captions=captions,
                cli=cli,
                extra_div=True,
                unique_id=True,
                autoscale=True,
            )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def _make_downloads_page(self, pages):
        """Make a page with links to files which can be downloaded

        Parameters
        ----------
        pages: list
            list of pages you wish to create
        """
        return super(TGRWebpageGeneration, self)._make_downloads_page(
            pages, fix_bottom=True
        )

    def _make_entry_in_downloads_table(self, *args, **kwargs):
        """Make a label specific entry into the downloads table. Given that
        we do not want to have label specific entries in the downloads table
        this function simply returns None to overwrite the inherited
        pesummary.core.webpage.main._WebpageGeneration function
        """
        return
