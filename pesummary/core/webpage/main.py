# Licensed under an MIT style license -- see LICENSE.md

import os
import sys
import uuid
from glob import glob
from operator import itemgetter
from pathlib import Path
import shutil

from scipy import stats
import numpy as np
import math

import pesummary
from pesummary import conf, __version_string__
from pesummary.utils.utils import (
    logger, LOG_FILE, jensen_shannon_divergence, safe_round, make_dir
)
from pesummary.core.webpage import webpage

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class PlotCaption(object):
    """Class to handle the generation of a plot caption

    Parameters
    ----------
    plot: str
        name of the plot you wish to generate a caption for
    """
    def __new__(cls, plot="1d_histogram"):
        super(PlotCaption, cls).__init__(cls)
        obj = cls.caption(plot)
        return obj

    def __init__(self, plot="1d_histogram"):
        self.plot = plot

    @staticmethod
    def caption(plot):
        if hasattr(conf, "caption_{}".format(plot)):
            return getattr(conf, "caption_{}".format(plot))
        return "No caption found for {}".format(plot)


class _WebpageGeneration(object):
    """Super class to handle the webpage generation for a given set of result
    files

    Parameters
    ----------
    savedir: str
        the directory to store the plots
    webdir: str
        the web directory of the run
    samples: dict
        dictionary of posterior samples stored in the result files
    labels: list
        list of labels used to distinguish the result files
    publication: Bool
        Bool to determine the generation of a publication webpage
    user: str
        the user that submitted the job
    config: list
        list of configuration files used for each result file
    same_parameters: list
        list of paramerers that are common in all result files
    base_url: str
        the url corresponding to the web directory
    file_versions: dict
        dictionary of file versions for each result file
    hdf5: Bool
        Bool to determine if the metafile has been saved as hdf5 or not
    colors: list
        colors that you wish to use to distinguish different result files
    custom_plotting:
    existing_labels: list
        list of labels stored in an existing metafile
    existing_config: list
        list of configuration files stored in the existing metafile
    existing_file_version: dict
        dictionary of file versions stored in the existing metafile
    existing_injection_data: dict
        dictionary of injection data stored in an existing metafile
    existing_samples: dict
        dictionary of posterior samples stored in an existing metafile
    existing_metafile: str
        path to the existing metafile
    existing_file_kwargs: dict
        dictionary of file kwargs stored in an existing metafile
    add_to_existing: Bool
        Bool to determine if you wish to add to an existing webpage
    notes: str
        notes that you wish to put on the webpages
    disable_comparison: bool
        Whether to make comparison pages
    package_information: dict
        dictionary of package information
    mcmc_samples: Bool
        Whether or not mcmc samples have been passed
    """
    def __init__(
        self, webdir=None, samples=None, labels=None, publication=None,
        user=None, config=None, same_parameters=None, base_url=None,
        file_versions=None, hdf5=None, colors=None, custom_plotting=None,
        existing_labels=None, existing_config=None, existing_file_version=None,
        existing_injection_data=None, existing_samples=None,
        existing_metafile=None, existing_file_kwargs=None,
        existing_weights=None, add_to_existing=False, notes=None,
        disable_comparison=False, disable_interactive=False,
        package_information={"packages": [], "manager": "pypi"},
        mcmc_samples=False, external_hdf5_links=False, key_data=None,
        existing_plot=None, disable_expert=False, analytic_priors=None,
    ):
        self.webdir = webdir
        make_dir(self.webdir)
        make_dir(os.path.join(self.webdir, "html"))
        make_dir(os.path.join(self.webdir, "css"))
        self.samples = samples
        self.labels = labels
        self.publication = publication
        self.user = user
        self.config = config
        self.same_parameters = same_parameters
        self.base_url = base_url
        self.file_versions = file_versions
        if self.file_versions is None:
            self.file_versions = {
                label: "No version information found" for label in self.labels
            }
        self.hdf5 = hdf5
        self.colors = colors
        if self.colors is None:
            self.colors = list(conf.colorcycle)
        self.custom_plotting = custom_plotting
        self.existing_labels = existing_labels
        self.existing_config = existing_config
        self.existing_file_version = existing_file_version
        self.existing_samples = existing_samples
        self.existing_metafile = existing_metafile
        self.existing_file_kwargs = existing_file_kwargs
        self.add_to_existing = add_to_existing
        self.analytic_priors = analytic_priors
        if self.analytic_priors is None:
            self.analytic_priors = {label: None for label in self.samples.keys()}
        self.key_data = key_data
        _label = self.labels[0]
        if self.samples is not None:
            if key_data is None:
                self.key_data = {
                    label: _samples.key_data for label, _samples in
                    self.samples.items()
                }
            self.key_data_headings = sorted(
                list(self.key_data[_label][list(self.samples[_label].keys())[0]])
            )
            self.key_data_table = {
                label: {
                    param: [
                        safe_round(self.key_data[label][param][key], 3) for key
                        in self.key_data_headings
                    ] for param in self.samples[label].keys()
                } for label in self.labels
            }
        self.notes = notes
        self.make_interactive = not disable_interactive
        self.package_information = package_information
        self.mcmc_samples = mcmc_samples
        self.external_hdf5_links = external_hdf5_links
        self.existing_plot = existing_plot
        self.expert_plots = not disable_expert
        self.make_comparison = (
            not disable_comparison and self._total_number_of_labels > 1
        )
        self.preliminary_pages = {label: False for label in self.labels}
        self.all_pages_preliminary = False
        self._additional_1d_pages = {label: [] for label in self.labels}
        if self.additional_1d_pages is not None:
            for j, _parameters in self.additional_1d_pages.items():
                for i in self.labels:
                    if all(
                            param in self.samples[i].keys() for param in
                            _parameters
                    ):
                        self._additional_1d_pages[i].append(j)
        self.categories = self.default_categories()
        self.popular_options = self.default_popular_options()
        self.navbar = {
            "home": self.make_navbar_for_homepage(),
            "result_page": self.make_navbar_for_result_page(),
            "comparison": self.make_navbar_for_comparison_page()
        }
        self.image_path = {
            "home": os.path.join(".", "plots", ""),
            "other": os.path.join("..", "plots", "")
        }
        self.results_path = {
            "home": "./samples/", "other": "../samples/"
        }
        self.config_path = {
            "home": "./config/", "other": "../config/"
        }
        if self.make_comparison:
            try:
                self.comparison_stats = self.generate_comparison_statistics()
            except Exception as e:
                self.comparison_stats = None
                logger.info(
                    "Failed to generate comparison statistics because {}. As a "
                    "result they will not be added to the webpages".format(e)
                )

    @property
    def _metafile(self):
        return (
            "posterior_samples.json" if not self.hdf5 else
            "posterior_samples.h5"
        )

    @property
    def _total_number_of_labels(self):
        _number_of_labels = 0
        for item in [self.labels, self.existing_labels]:
            if isinstance(item, list):
                _number_of_labels += len(item)
        return _number_of_labels

    def copy_css_and_js_scripts(self):
        """Copy css and js scripts from the package to the web directory
        """
        import pkg_resources
        import shutil
        files_to_copy = []
        path = pkg_resources.resource_filename("pesummary", "core")
        scripts = glob(os.path.join(path, "js", "*.js"))
        for i in scripts:
            files_to_copy.append(
                [i, os.path.join(self.webdir, "js", os.path.basename(i))]
            )
        scripts = glob(os.path.join(path, "css", "*.css"))
        for i in scripts:
            files_to_copy.append(
                [i, os.path.join(self.webdir, "css", os.path.basename(i))]
            )
        for _dir in ["js", "css"]:
            try:
                os.mkdir(os.path.join(self.webdir, _dir))
            except FileExistsError:
                pass
        for ff in files_to_copy:
            shutil.copy(ff[0], ff[1])

    def make_modal_carousel(
        self, html_file, image_contents, unique_id=False, **kwargs
    ):
        """Make a modal carousel for a table of images

        Parameters
        ----------
        html_file: pesummary.core.webpage.webpage.page
            open html file
        image_contents: list
            list of images to place in a table
        unique_id: Bool, optional
            if True, assign a unique identifer to the modal
        **kwargs: dict, optional
            all additional kwargs passed to `make_table_of_images` function
        """
        if unique_id:
            unique_id = '{}'.format(uuid.uuid4().hex.upper()[:6])
        else:
            unique_id = None
        html_file.make_table_of_images(
            contents=image_contents, unique_id=unique_id, **kwargs
        )
        images = [y for x in image_contents for y in x]
        html_file.make_modal_carousel(images=images, unique_id=unique_id)
        return html_file

    def generate_comparison_statistics(self):
        """Generate comparison statistics for all parameters that are common to
        all result files
        """
        data = {
            i: self._generate_comparison_statistics(
                i, [self.samples[j][i] for j in self.labels]
            ) for i in self.same_parameters
        }
        return data

    def _generate_comparison_statistics(self, param, samples):
        """Generate comparison statistics for a set of samples

        Parameters
        ----------
        samples: list
            list of samples for each result file
        """
        from scipy.stats import gaussian_kde
        from pesummary.utils.utils import kolmogorov_smirnov_test

        rows = range(len(samples))
        columns = range(len(samples))
        ks = [
            [
                kolmogorov_smirnov_test([samples[i], samples[j]]) for i in
                rows
            ] for j in columns
        ]
        js = [
            [
                self._jensen_shannon_divergence(param, [samples[i], samples[j]])
                for i in rows
            ] for j in columns
        ]
        return [ks, js]

    def _jensen_shannon_divergence(self, param, samples, **kwargs):
        """Return the Jensen Shannon divergence between two sets of samples

        Parameters
        ----------
        param: str
            The parameter that the samples belong to
        samples: list
            2d list containing the samples you wish to calculate the Jensen
            Shannon divergence between
        """
        return jensen_shannon_divergence([samples[0], samples[1]], **kwargs)

    @staticmethod
    def get_executable(executable):
        """Return the path to an executable

        Parameters
        ----------
        executable: str
            the name of the executable you wish to find
        """
        return shutil.which(
            executable,
            path=os.pathsep.join((
                os.getenv("PATH", ""),
                str(Path(sys.executable).parent),
            )),
        )

    def _result_page_links(self):
        """Return the navbar structure for the Result Page tab.
        """
        return [{i: i} for i in self.labels]

    def make_navbar_for_homepage(self):
        """Make a navbar for the homepage
        """
        links = [
            "home", ["Result Pages", self._result_page_links()], "Logging",
            "Version"
        ]
        if self.make_comparison:
            links[1][1] += ["Comparison"]
        if self.publication:
            links.insert(2, "Publication")
        if self.notes is not None:
            links.append("Notes")
        return links

    def make_navbar_for_result_page(self):
        """Make a navbar for the result page homepage
        """
        links = {
            i: ["1d Histograms", [{"Custom": i}, {"All": i}]] for i in
            self.labels
        }
        for num, label in enumerate(self.labels):
            _params = list(self.samples[label].keys())
            if len(self._additional_1d_pages[label]):
                _params += self._additional_1d_pages[label]
            for j in self.categorize_parameters(_params):
                j = [j[0], [{k: label} for k in j[1]]]
                links[label].append(j)

        final_links = {
            i: [
                "home", ["Result Pages", self._result_page_links()],
                {"Corner": i}, {"Config": i}, links[i]
            ] for i in self.labels
        }
        if self.make_comparison:
            for label in self.labels:
                final_links[label][1][1] += ["Comparison"]
        _dummy_label = self.labels[0]
        if len(final_links[_dummy_label][1][1]) > 1:
            for label in self.labels:
                _options = [{l: "switch"} for l in self.labels if l != label]
                if self.make_comparison:
                    _options.append({"Comparison": "switch"})
                final_links[label].append(["Switch", _options])
        if self.make_interactive:
            for label in self.labels:
                final_links[label].append(
                    ["Interactive", [{"Interactive_Corner": label}]]
                )
        if self.existing_plot is not None:
            for _label in self.labels:
                if _label in self.existing_plot.keys():
                    final_links[_label].append(
                        {"Additional": _label}
                    )
        return final_links

    def make_navbar_for_comparison_page(self):
        """Make a navbar for the comparison homepage
        """
        if self.same_parameters is not None:
            links = ["1d Histograms", ["Custom", "All"]]
            for i in self.categorize_parameters(self.same_parameters):
                links.append(i)
            final_links = [
                "home", ["Result Pages", self._result_page_links()], links
            ]
            final_links[1][1] += ["Comparison"]
            final_links.append(
                ["Switch", [{l: "switch"} for l in self.labels]]
            )
            if self.make_interactive:
                final_links.append(
                    ["Interactive", ["Interactive_Ridgeline"]]
                )
            return final_links
        return None

    def categorize_parameters(
        self, parameters, starting_letter=True, heading_all=True
    ):
        """Categorize the parameters into common headings

        Parameters
        ----------
        parameters: list
            list of parameters that you would like to sort
        """
        params = []
        for heading, category in self.categories.items():
            if any(
                any(i[0] in j for j in category["accept"]) for i in parameters
            ):
                cond = self._condition(
                    category["accept"], category["reject"],
                    starting_letter=starting_letter
                )
                part = self._partition(cond, parameters)
                if heading_all and len(part):
                    part = ["{}_all".format(heading)] + part
                params.append([heading, part])
        used_headings = [i[0] for i in params]
        other_index = \
            used_headings.index("others") if "others" in used_headings else None
        other_params = []
        for pp in parameters:
            if not any(pp in j[1] for j in params):
                if other_index is not None:
                    params[other_index][1].append(pp)
                else:
                    other_params.append(pp)
        if other_index is None:
            params.append(["others", other_params])
        return params

    def _condition(self, true, false, starting_letter=False):
        """Setup a condition

        Parameters
        ----------
        true: list
            list of strings that you would like to include
        false: list
            list of strings that you would like to neglect
        """
        def _starting_letter(param, condition):
            if condition:
                return param[0]
            return param
        if len(true) != 0 and len(false) == 0:
            condition = lambda j: True if any(
                i in _starting_letter(j, starting_letter) for i in true
            ) else False
        elif len(true) == 0 and len(false) != 0:
            condition = lambda j: True if any(
                i not in _starting_letter(j, starting_letter) for i in false
            ) else False
        elif len(true) and len(false) != 0:
            condition = lambda j: True if any(
                i in _starting_letter(j, starting_letter) and all(
                    k not in _starting_letter(j, starting_letter) for k in false
                ) for i in true
            ) else False
        return condition

    def _partition(self, condition, array):
        """Filter the list according to a condition

        Parameters
        ----------
        condition: func
            lambda function containing the condition that you want to use to
            filter the array
        array: list
            List of parameters that you would like to filter
        """
        return sorted(list(filter(condition, array)))

    def generate_webpages(self):
        """Generate all webpages for all result files passed
        """
        if self.add_to_existing:
            self.add_existing_data()
        self.make_home_pages()
        self.make_1d_histogram_pages()
        self.make_corner_pages()
        self.make_config_pages()
        if self.make_comparison:
            self.make_comparison_pages()
        if self.make_interactive:
            self.make_interactive_pages()
        if self.existing_plot is not None:
            self.make_additional_plots_pages()
        self.make_error_page()
        self.make_version_page()
        self.make_logging_page()
        if self.notes is not None:
            self.make_notes_page()
        self.make_downloads_page()
        self.make_about_page()
        try:
            self.generate_specific_javascript()
        except Exception:
            pass

    def create_blank_html_pages(self, pages, stylesheets=[]):
        """Create blank html pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        webpage.make_html(
            web_dir=self.webdir, pages=pages, stylesheets=stylesheets
        )

    def setup_page(
        self, html_page, links, label=None, title=None, approximant=None,
        background_colour=None, histogram_download=False, toggle=False
    ):
        """Set up each webpage with a header and navigation bar.

        Parameters
        ----------
        html_page: str
            String containing the html page that you would like to set up
        links: list
            List containing the navbar structure that you would like to include
        label: str, optional
            The label that prepends your webpage name
        title: str, optional
            String that you would like to include in your header
        approximant: str, optional
            The approximant that you would like associated with your html_page
        background_colour: str, optional
            String containing the background colour of your header
        histogram_download: bool, optional
            If true, a download link for the each histogram is displayed in
            the navbar
        """
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page=html_page,
            label=label
        )
        _preliminary_keys = self.preliminary_pages.keys()
        if self.all_pages_preliminary:
            html_file.make_watermark()
        elif approximant is not None and approximant in _preliminary_keys:
            if self.preliminary_pages[approximant]:
                html_file.make_watermark()

        html_file.make_header(approximant=approximant)
        if html_page == "home" or html_page == "home.html":
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour,
                hdf5=self.hdf5, toggle=toggle
            )
        elif histogram_download:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["other"],
                toggle=toggle, histogram_download=os.path.join(
                    "..", "samples", "dat", label, "{}_{}_samples.dat".format(
                        label, html_page
                    )
                ), background_color=background_colour, hdf5=self.hdf5
            )
        else:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour, hdf5=self.hdf5,
                toggle=toggle
            )
        return html_file

    def make_home_pages(self):
        """Wrapper function for _make_home_pages()
        """
        pages = ["{}_{}".format(i, i) for i in self.labels]
        pages.append("home")
        self.create_blank_html_pages(pages)
        self._make_home_pages(pages)

    def _make_home_pages(
        self, pages, title=None, banner="Summary", make_home=True,
        make_result=True, return_html=False
    ):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        if make_home:
            html_file = self.setup_page("home", self.navbar["home"], title=title)
            html_file.make_banner(approximant=banner, key="Summary")
            if return_html:
                return html_file
        if not make_result:
            return

        for num, i in enumerate(self.labels):
            html_file = self.setup_page(
                i, self.navbar["result_page"][i], i, approximant=i,
                title="{} Summary page".format(i),
                background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key=i)
            images, cli, captions = self.default_images_for_result_page(i)
            if len(images):
                html_file = self.make_modal_carousel(
                    html_file, images, cli=cli, unique_id=True,
                    captions=captions, autoscale=True
                )

            if self.custom_plotting:
                custom_plots = glob(
                    "{}/plots/{}_custom_plotting_*".format(self.webdir, i)
                )
                path = self.image_path["other"]
                for num, i in enumerate(custom_plots):
                    custom_plots[num] = path + i.split("/")[-1]
                image_contents = [
                    custom_plots[i:4 + i] for i in range(
                        0, len(custom_plots), 4
                    )
                ]
                html_file = self.make_modal_carousel(
                    html_file, image_contents, unique_id=True
                )

            html_file.make_banner(
                approximant="Summary Table", key="summary_table",
                _style="font-size: 26px;"
            )
            _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
            _class = "row justify-content-center"
            html_file.make_container(style=_style)
            html_file.make_div(4, _class=_class, _style=None)

            key_data = self.key_data
            contents = []
            headings = [" "] + self.key_data_headings.copy()
            _injection = False
            if "injected" in headings:
                _injection = not all(
                    math.isnan(_data["injected"]) for _data in
                    self.key_data[i].values()
                )
            if _injection:
                headings.append("injected")
            for j in self.samples[i].keys():
                row = []
                row.append(j)
                row += self.key_data_table[i][j]
                if _injection:
                    row.append(safe_round(self.key_data[i][j]["injected"], 3))
                contents.append(row)

            html_file.make_table(
                headings=headings, contents=contents, heading_span=1,
                accordian=False, format="table-hover header-fixed",
                sticky_header=True
            )
            html_file.end_div(4)
            html_file.end_container()
            html_file.export(
                "summary_information_{}.csv".format(i)
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_1d_histogram_pages(self):
        """Wrapper function for _make_1d_histogram pages
        """
        pages = [
            "{}_{}_{}".format(i, i, j) for i in self.labels for j in
            self.samples[i].keys()
        ]
        pages += ["{}_{}_Custom".format(i, i) for i in self.labels]
        pages += ["{}_{}_All".format(i, i) for i in self.labels]
        for i in self.labels:
            if len(self._additional_1d_pages[i]):
                pages += [
                    "{}_{}_{}".format(i, i, j) for j in
                    self._additional_1d_pages[i]
                ]
        pages += [
            "{}_{}_{}_all".format(i, i, j[0]) for i in self.labels for j in
            self.categorize_parameters(self.samples[i].keys()) if len(j[1])
        ]
        self.create_blank_html_pages(pages)
        self._make_1d_histogram_pages(pages)

    def _make_1d_histogram_pages(self, pages):
        """Make the 1d histogram pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        for num, i in enumerate(self.labels):
            if len(self._additional_1d_pages[i]):
                for j in self._additional_1d_pages[i]:
                    _parameters = self.additional_1d_pages[j]
                    html_file = self.setup_page(
                        "{}_{}".format(i, j), self.navbar["result_page"][i],
                        i, title="{} Posterior PDFs describing {}".format(i, j),
                        approximant=i, background_colour=self.colors[num],
                        histogram_download=False, toggle=self.expert_plots
                    )
                    html_file.make_banner(approximant=i, key=i)
                    path = self.image_path["other"]
                    _plots = [
                        path + "{}_1d_posterior_{}.png".format(i, param) for
                        param in _parameters
                    ]
                    contents = [
                        _plots[i:2 + i] for i in range(0, len(_plots), 2)
                    ]
                    html_file.make_table_of_images(
                        contents=contents, code="changeimage",
                        mcmc_samples=self.mcmc_samples, autoscale=True
                    )
                    key_data = self.key_data
                    contents = []
                    headings = [" "] + self.key_data_headings.copy()
                    _injection = False
                    rows = []
                    for param in _parameters:
                        _row = [param]
                        _row += self.key_data_table[i][param]
                        rows.append(_row)
                    _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
                    _class = "row justify-content-center"
                    html_file.make_container(style=_style)
                    html_file.make_div(4, _class=_class, _style=None)
                    html_file.make_table(
                        headings=headings, contents=rows, heading_span=1,
                        accordian=False, format="table-hover"
                    )
                    html_file.end_div(4)
                    html_file.end_container()
                    html_file.export("summary_information_{}.csv".format(i))
                    html_file.make_footer(user=self.user, rundir=self.webdir)
                    html_file.close()
            for j in self.samples[i].keys():
                html_file = self.setup_page(
                    "{}_{}".format(i, j), self.navbar["result_page"][i],
                    i, title="{} Posterior PDF for {}".format(i, j),
                    approximant=i, background_colour=self.colors[num],
                    histogram_download=False, toggle=self.expert_plots
                )
                if j.description != "Unknown parameter description":
                    _custom = (
                        "The figures below show the plots for {}: {}"
                    )
                    html_file.make_banner(
                        approximant=i, key="custom",
                        custom=_custom.format(j, j.description)
                    )
                else:
                    html_file.make_banner(approximant=i, key=i)
                path = self.image_path["other"]
                contents = [
                    [path + "{}_1d_posterior_{}.png".format(i, j)],
                    [
                        path + "{}_sample_evolution_{}.png".format(i, j),
                        path + "{}_autocorrelation_{}.png".format(i, j)
                    ]
                ]
                captions = [
                    [PlotCaption("1d_histogram").format(j)],
                    [
                        PlotCaption("sample_evolution").format(j),
                        PlotCaption("autocorrelation").format(j)
                    ]
                ]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage",
                    captions=captions, mcmc_samples=self.mcmc_samples
                )
                contents = [
                    [path + "{}_2d_contour_{}_log_likelihood.png".format(i, j)],
                    [
                        path + "{}_sample_evolution_{}_{}_colored.png".format(
                            i, j, "log_likelihood"
                        ), path + "{}_1d_posterior_{}_bootstrap.png".format(i, j)
                    ]
                ]
                captions = [
                    [PlotCaption("2d_contour").format(j, "log_likelihood")],
                    [
                        PlotCaption("sample_evolution_colored").format(
                            j, "log_likelihood"
                        ),
                        PlotCaption("1d_histogram_bootstrap").format(100, j, 1000)
                    ],
                ]
                if self.expert_plots:
                    html_file.make_table_of_images(
                        contents=contents, rows=1, columns=2, code="changeimage",
                        captions=captions, mcmc_samples=self.mcmc_samples,
                        display='none', container_id='expert_div',
                        close_container=False
                    )
                    _additional = self.add_to_expert_pages(path, i)
                    if _additional is not None and j in _additional.keys():
                        html_file.make_table_of_images(
                            contents=_additional[j], code="changeimage",
                            mcmc_samples=self.mcmc_samples,
                            autoscale=True, display='none',
                            add_to_open_container=True,
                        )
                    html_file.end_div()
                html_file.export(
                    "", csv=False, json=False, shell=False, margin_bottom="1em",
                    histogram_dat=os.path.join(
                        self.results_path["other"], i, "{}_{}.dat".format(i, j)
                    )
                )
                key_data = self.key_data
                contents = []
                headings = self.key_data_headings.copy()
                _injection = False
                if "injected" in headings:
                    _injection = not math.isnan(self.key_data[i][j]["injected"])
                row = self.key_data_table[i][j]
                if _injection:
                    headings.append("injected")
                    row.append(safe_round(self.key_data[i][j]["injected"], 3))
                _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
                _class = "row justify-content-center"
                html_file.make_container(style=_style)
                html_file.make_div(4, _class=_class, _style=None)
                html_file.make_table(
                    headings=headings, contents=[row], heading_span=1,
                    accordian=False, format="table-hover"
                )
                html_file.end_div(4)
                html_file.end_container()
                html_file.export(
                    "summary_information_{}.csv".format(i)
                )
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()
            html_file = self.setup_page(
                "{}_Custom".format(i), self.navbar["result_page"][i],
                i, title="{} Posteriors for multiple".format(i),
                approximant=i, background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key=i)
            ordered_parameters = self.categorize_parameters(
                self.samples[i].keys()
            )
            ordered_parameters = [i for j in ordered_parameters for i in j[1]]
            popular_options = self.popular_options
            html_file.make_search_bar(
                sidebar=[i for i in self.samples[i].keys()],
                popular_options=popular_options + [{
                    "all": ", ".join(ordered_parameters)
                }],
                label=self.labels[num], code="combines"
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
            html_file = self.setup_page(
                "{}_All".format(i), self.navbar["result_page"][i],
                i, title="All posteriors for {}".format(i),
                approximant=i, background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key=i)
            for j in self.samples[i].keys():
                html_file.make_banner(
                    approximant=j, _style="font-size: 26px;"
                )
                contents = [
                    [path + "{}_1d_posterior_{}.png".format(i, j)],
                    [
                        path + "{}_sample_evolution_{}.png".format(i, j),
                        path + "{}_autocorrelation_{}.png".format(i, j)
                    ]
                ]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage")
            html_file.close()
            for j in self.categorize_parameters(self.samples[i].keys()):
                if not len(j[1]):
                    continue
                html_file = self.setup_page(
                    "{}_{}_all".format(i, j[0]), self.navbar["result_page"][i],
                    i, title="All posteriors for describing {}".format(j[0]),
                    approximant=i, background_colour=self.colors[num]
                )
                for k in j[1][1:]:
                    html_file.make_banner(
                        approximant=k, _style="font-size: 26px;"
                    )
                    contents = [
                        [path + "{}_1d_posterior_{}.png".format(i, k)],
                        [
                            path + "{}_sample_evolution_{}.png".format(i, k),
                            path + "{}_autocorrelation_{}.png".format(i, k)
                        ]
                    ]
                    html_file.make_table_of_images(
                        contents=contents, rows=1, columns=2, code="changeimage"
                    )
                html_file.make_banner(
                    approximant="Summary Table", key="summary_table",
                    _style="font-size: 26px;"
                )
                _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
                _class = "row justify-content-center"
                html_file.make_container(style=_style)
                html_file.make_div(4, _class=_class, _style=None)
                headings = [" "] + self.key_data_headings.copy()
                contents = []
                for k in j[1][1:]:
                    row = [k]
                    row += self.key_data_table[i][k]
                    contents.append(row)
                html_file.make_table(
                    headings=headings, contents=contents, heading_span=1,
                    accordian=False, format="table-hover",
                    sticky_header=True
                )
                html_file.end_div(4)
                html_file.end_container()
                html_file.export("{}_summary_{}.csv".format(j[0], i))
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()

    def make_additional_plots_pages(self):
        """Wrapper function for _make_additional_plots_pages
        """
        pages = [
            "{}_{}_Additional".format(i, i) for i in self.labels if i in
            self.existing_plot.keys()
        ]
        self.create_blank_html_pages(pages)
        self._make_additional_plots_pages(pages)

    def _make_additional_plots_pages(self, pages):
        """Make the additional plots pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        from PIL import Image
        for num, i in enumerate(self.labels):
            if i not in self.existing_plot.keys():
                continue
            html_file = self.setup_page(
                "{}_Additional".format(i), self.navbar["result_page"][i], i,
                title="Additional plots for {}".format(i), approximant=i,
                background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key="additional")
            if isinstance(self.existing_plot[i], list):
                images = sorted(
                    self.existing_plot[i],
                    key=lambda _path: Image.open(_path).size[1]
                )
                tol = 2.
                # if all images are of similar height, then grid them uniformly, else
                # grid by heights
                cond = all(
                    Image.open(_path).size[1] / tol < Image.open(images[0]).size[1]
                    for _path in images
                )
                if cond:
                    image_contents = [
                        [
                            self.image_path["other"] + Path(_path).name for
                            _path in images[j:4 + j]
                        ] for j in range(0, len(self.existing_plot[i]), 4)
                    ]
                else:
                    heights = [Image.open(_path).size[1] for _path in images]
                    image_contents = [
                        [self.image_path["other"] + Path(images[0]).name]
                    ]
                    row = 0
                    for num, _height in enumerate(heights[1:]):
                        if _height / tol < heights[num]:
                            image_contents[row].append(
                                self.image_path["other"] + Path(images[num + 1]).name
                            )
                        else:
                            row += 1
                            image_contents.append(
                                [self.image_path["other"] + Path(images[num + 1]).name]
                            )
                margin_left = None
            else:
                image_contents = [
                    [self.image_path["other"] + Path(self.existing_plot[i]).name]
                ]
                margin_left = "-250px"
            html_file = self.make_modal_carousel(
                html_file, image_contents, unique_id=True, autoscale=True,
                margin_left=margin_left
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_corner_pages(self):
        """Wrapper function for _make_corner_pages
        """
        pages = ["{}_{}_Corner".format(i, i) for i in self.labels]
        self.create_blank_html_pages(pages)
        self._make_corner_pages(pages)

    def _make_corner_pages(self, pages):
        """Make the corner pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        for num, i in enumerate(self.labels):
            html_file = self.setup_page(
                "{}_Corner".format(i), self.navbar["result_page"][i], i,
                title="{} Corner Plots".format(i), approximant=i,
                background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key="corner")
            popular_options = self.popular_options
            if len(self.default_corner_params()):
                params = self.default_corner_params()
                included_parameters = [
                    i for i in list(self.samples[i].keys()) if i in params
                ]
                popular_options += [{
                    "all": ", ".join(included_parameters)
                }]
            else:
                included_params = self.samples[i].keys()
            html_file.make_search_bar(
                sidebar=self.samples[i].keys(),
                popular_options=popular_options, label=i
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_config_pages(self):
        """Wrapper function for _make_config_pages
        """
        pages = ["{}_{}_Config".format(i, i) for i in self.labels]
        self.create_blank_html_pages(pages, stylesheets=pages)
        self._make_config_pages(pages)

    def _make_config_pages(self, pages):
        """Make the config pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        for num, i in enumerate(self.labels):
            html_file = self.setup_page(
                "{}_Config".format(i), self.navbar["result_page"][i], i,
                title="{} Configuration".format(i), approximant=i,
                background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key="config")
            if self.config and num < len(self.config) and self.config[num]:
                with open(self.config[num], 'r') as f:
                    contents = f.read()
                html_file.make_container()
                styles = html_file.make_code_block(
                    language='ini', contents=contents
                )
                html_file.end_container()
                with open(
                    "{0:s}/css/{1:s}_{2:s}_Config.css".format(
                        self.webdir, i, i
                    ), "w"
                ) as f:
                    f.write(styles)
                _fix = False
            else:
                html_file.add_content(
                    "<div class='row justify-content-center'; "
                    "style='font-family: Arial-body; font-size: 14px'>"
                    "<p style='margin-top:2.5em'> No configuration file was "
                    "provided </p></div>"
                )
                _fix = True
            if i in self.analytic_priors.keys():
                if self.analytic_priors[i] is not None:
                    html_file.make_div(indent=2, _class='paragraph')
                    html_file.add_content(
                        "Below is the prior file for %s" % (i)
                    )
                    html_file.end_div()
                    html_file.make_container()
                    styles = html_file.make_code_block(
                        language='ini', contents=self.analytic_priors[i]
                    )
                    html_file.end_container()
                    _fix = False
            html_file.make_footer(
                user=self.user, rundir=self.webdir, fix_bottom=_fix
            )
            html_file.close()

    def make_comparison_pages(self):
        """Wrapper function for _make_comparison_pages
        """
        pages = ["Comparison_{}".format(i) for i in self.same_parameters]
        pages += ["Comparison_Custom"]
        pages += ["Comparison_All"]
        pages += ["Comparison"]
        pages += [
            "Comparison_{}_all".format(j[0]) for j in
            self.categorize_parameters(self.same_parameters) if len(j[1])
        ]
        self.create_blank_html_pages(pages)
        self._make_comparison_pages(pages)

    def _make_comparison_pages(self, pages):
        """Make pages to compare all result files

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = self.setup_page(
            "Comparison", self.navbar["comparison"], approximant="Comparison",
            title="Comparison Summary Page"
        )
        html_file.make_banner(approximant="Comparison", key="Comparison")
        path = self.image_path["other"]
        if len(self.default_comparison_homepage_plots()):
            contents = self.default_comparison_homepage_plots()
            html_file = self.make_modal_carousel(
                html_file, contents, unique_id=True
            )
        if self.custom_plotting:
            from glob import glob

            custom_plots = glob(
                os.path.join(
                    self.webdir, "plots", "combined_custom_plotting_*"
                )
            )
            for num, i in enumerate(custom_plots):
                custom_plots[num] = path + i.split("/")[-1]
            image_contents = [
                custom_plots[i:4 + i] for i in range(0, len(custom_plots), 4)
            ]
            html_file = self.make_modal_carousel(
                html_file, image_contents, unique_id=True
            )
        path = self.image_path["other"]

        if self.comparison_stats is not None:
            for _num, _key in enumerate(["KS_test", "JS_test"]):
                if _key == "KS_test":
                    html_file.make_banner(
                        approximant="KS test", key="ks_test",
                        _style="font-size: 26px;"
                    )
                else:
                    html_file.make_banner(
                        approximant="JS test", key="js_test",
                        _style="font-size: 26px;"
                    )
                _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
                _class = "row justify-content-center"
                html_file.make_container(style=_style)
                html_file.make_div(4, _class=_class, _style=None)

                rows = range(len(self.labels))
                table_contents = {
                    i: [
                        [self.labels[j]] + self.comparison_stats[i][_num][j] for
                        j in rows
                    ] for i in self.same_parameters
                }
                _headings = [" "] + self.labels
                html_file.make_table(
                    headings=_headings, format="table-hover scroll-table",
                    heading_span=1, contents=table_contents, accordian=False,
                    scroll_table=True
                )
                html_file.end_div(4)
                html_file.end_container()
                html_file.export("{}_total.csv".format(_key))
        html_file.make_footer(user=self.user, rundir=self.webdir)

        for num, i in enumerate(self.same_parameters):
            html_file = self.setup_page(
                "Comparison_{}".format(i), self.navbar["comparison"],
                title="Comparison PDF for {}".format(i),
                approximant="Comparison"
            )
            html_file.make_banner(approximant="Comparison", key="Comparison")
            path = self.image_path["other"]
            contents = [
                [path + "combined_1d_posterior_{}.png".format(i)],
                [
                    path + "combined_cdf_{}.png".format(i),
                    path + "combined_boxplot_{}.png".format(i)
                ]
            ]
            html_file.make_table_of_images(
                contents=contents, rows=1, columns=2, code="changeimage"
            )
            html_file.make_banner(
                approximant="Summary Table", key="summary_table",
                _style="font-size: 26px;"
            )
            _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
            _class = "row justify-content-center"
            html_file.make_container(style=_style)
            html_file.make_div(4, _class=_class, _style=None)
            headings = [" "] + self.key_data_headings.copy()
            contents = []
            for label in self.labels:
                row = [label]
                row += self.key_data_table[label][i]
                contents.append(row)
            html_file.make_table(
                headings=headings, contents=contents, heading_span=1,
                accordian=False, format="table-hover",
                sticky_header=True
            )
            html_file.end_div(4)
            html_file.end_container()
            html_file.export("comparison_summary_{}.csv".format(i))

            if self.comparison_stats is not None:
                for _num, _key in enumerate(["KS_test", "JS_test"]):
                    if _key == "KS_test":
                        html_file.make_banner(
                            approximant="KS test", key="ks_test",
                            _style="font-size: 26px;"
                        )
                    else:
                        html_file.make_banner(
                            approximant="JS test", key="js_test",
                            _style="font-size: 26px;"
                        )
                    _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
                    _class = "row justify-content-center"
                    html_file.make_container(style=_style)
                    html_file.make_div(4, _class=_class, _style=None)

                    table_contents = [
                        [self.labels[j]] + self.comparison_stats[i][_num][j]
                        for j in rows
                    ]
                    _headings = [" "] + self.labels
                    html_file.make_table(
                        headings=_headings, format="table-hover",
                        heading_span=1, contents=table_contents, accordian=False
                    )
                    html_file.end_div(4)
                    html_file.end_container()
                    html_file.export("{}_{}.csv".format(_key, i))

            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        html_file = self.setup_page(
            "Comparison_Custom", self.navbar["comparison"],
            approximant="Comparison", title="Comparison Posteriors for multiple"
        )
        html_file.make_search_bar(
            sidebar=self.same_parameters, label="None", code="combines",
            popular_options=[
                {"all": ", ".join(self.same_parameters)}
            ]
        )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()
        html_file = self.setup_page(
            "Comparison_All", self.navbar["comparison"],
            title="All posteriors for Comparison", approximant="Comparison"
        )
        html_file.make_banner(approximant="Comparison", key="Comparison")
        for j in self.same_parameters:
            html_file.make_banner(
                approximant=j, _style="font-size: 26px;"
            )
            contents = [
                [path + "combined_1d_posterior_{}.png".format(j)],
                [
                    path + "combined_cdf_{}.png".format(j),
                    path + "combined_boxplot_{}.png".format(j)
                ]
            ]
            html_file.make_table_of_images(
                contents=contents, rows=1, columns=2, code="changeimage")
        html_file.close()
        for j in self.categorize_parameters(self.same_parameters):
            if not len(j[1]):
                continue
            html_file = self.setup_page(
                "Comparison_{}_all".format(j[0]), self.navbar["comparison"],
                title="All posteriors for describing {}".format(j[0]),
                approximant="Comparison"
            )
            for k in j[1][1:]:
                html_file.make_banner(
                    approximant=k, _style="font-size: 26px;"
                )
                contents = [
                    [path + "combined_1d_posterior_{}.png".format(k)],
                    [
                        path + "combined_cdf_{}.png".format(k),
                        path + "combined_boxplot_{}.png".format(k)
                    ]
                ]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage"
                )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_interactive_pages(self):
        """Wrapper function for _make_interactive_pages
        """
        pages = ["{}_{}_Interactive_Corner".format(i, i) for i in self.labels]
        if self.make_comparison:
            pages += ["Comparison_Interactive_Ridgeline"]
        self.create_blank_html_pages(pages)
        savedir = os.path.join(self.webdir, "plots")
        html_files = glob(os.path.join(savedir, "*interactive*.html"))
        html_files += glob(os.path.join(savedir, "corner", "*interactive*.html"))
        self._make_interactive_pages(pages, html_files)

    def _make_interactive_pages(self, pages, html_files):
        """Make a page that shows all interactive plots

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        for num, i in enumerate(self.labels):
            html_file = self.setup_page(
                "{}_Interactive_Corner".format(i),
                self.navbar["result_page"][i], i,
                title="{} Interactive Corner Plots".format(i), approximant=i,
                background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key="interactive_corner")
            html_file.make_container()
            corner_files = [
                figure for figure in html_files if "/corner/" in figure
                and i in figure
            ]
            for plot in corner_files:
                with open(plot, "r") as f:
                    data = f.read()
                    html_file.add_content(data)
            html_file.end_container()
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        if not self.make_comparison:
            return
        html_file = self.setup_page(
            "Comparison_Interactive_Ridgeline", self.navbar["comparison"],
            approximant="Comparison", title="Interactive Ridgeline Plots"
        )
        html_file.make_banner(
            approximant="Comparison", key="interactive_ridgeline"
        )
        posterior_files = [
            figure for figure in html_files if "ridgeline" in figure
        ]
        for plot in posterior_files:
            with open(plot, "r") as f:
                data = f.read()
                parameter = \
                    plot.split("interactive_ridgeline_")[1].split(".html")[0]
                html_file.make_banner(
                    approximant=parameter, _style="font-size: 26px;"
                )
                html_file.make_container()
                html_file.add_content(data)
                html_file.end_container()
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_error_page(self):
        """Wrapper function for _make_error_page
        """
        pages = ["error"]
        self.create_blank_html_pages(pages)
        self._make_error_page(pages)

    def _make_error_page(self, pages):
        """Make a page that is shown when something goes wrong

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        webpage.make_html(web_dir=self.webdir, pages=pages)
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="error"
        )
        html_file.make_div(
            _class="jumbotron text-center",
            _style="background-color: #D3D3D3; margin-bottom:0"
        )
        html_file.make_div(
            _class="container",
            _style="margin-top:1em; background-color: #D3D3D3; width: 45em"
        )
        html_file.add_content(
            "<h1 style='color:white; font-size: 8em'>404</h1>"
        )
        html_file.add_content(
            "<h2 style='color:white;'> Something went wrong... </h2>"
        )
        html_file.end_div()
        html_file.end_div()
        html_file.close()

    def make_version_page(self):
        """Wrapper function for _make_version_page
        """
        pages = ["Version"]
        self.create_blank_html_pages(pages, stylesheets=pages)
        self._make_version_page(pages)

    def _make_version_page(self, pages):
        """Make a page to display the version information

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        from pesummary._version_helper import (
            PackageInformation, install_path
        )

        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="Version"
        )
        html_file = self.setup_page(
            "Version", self.navbar["home"], title="Version Information"
        )
        html_file.make_banner(approximant="Version", key="Version")
        contents = __version_string__
        contents += install_path(return_string=True)
        for i in self.labels:
            contents = (
                "# {} version information\n\n{}_version={}\n\n".format(
                    i, i, self.file_versions[i]
                )
            ) + contents
        html_file.make_container()
        styles = html_file.make_code_block(language='shell', contents=contents)
        with open('{0:s}/css/Version.css'.format(self.webdir), 'w') as f:
            f.write(styles)
        html_file.end_container()
        packages = self.package_information["packages"]
        style = "margin-top:{}; margin-bottom:{};"
        if len(packages):
            html_file.make_table(
                headings=[x.title().replace('_', ' ') for x in packages.dtype.names],
                contents=[[pp.decode("utf-8") for pp in pkg] for pkg in packages],
                accordian=False, style=style.format("1em", "1em")
            )
            if self.package_information["manager"] == "conda":
                html_file.export(
                    "environment.yml", margin_top="1em", csv=False,
                    conda=True
                )
            else:
                html_file.export(
                    "requirements.txt", margin_top="1em", csv=False,
                    requirements=True
                )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_logging_page(self):
        """Wrapper function for _make_logging_page
        """
        pages = ["Logging"]
        self.create_blank_html_pages(pages, stylesheets=pages)
        self._make_logging_page(pages)

    def _make_logging_page(self, pages):
        """Make a page to display the logging output from PESummary

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="Logging"
        )
        html_file = self.setup_page(
            "Logging", self.navbar["home"], title="Logger Information"
        )
        html_file.make_banner(approximant="Logging", key="Logging")
        path = pesummary.__file__[:-12]
        log_file = LOG_FILE
        if not os.path.isfile(log_file):
            log_file = ".tmp/pesummary/no_log_information.log"
            with open(log_file, "w") as f:
                f.writelines(["No log information stored"])

        with open(log_file, 'r') as f:
            contents = f.read()
        html_file.make_container()
        styles = html_file.make_code_block(language='shell', contents=contents)
        with open('{0:s}/css/Logging.css'.format(self.webdir), 'w') as f:
            f.write(styles)
        html_file.end_container()
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_about_page(self):
        """Wrapper function for _make_about_page
        """
        pages = ["About"]
        self.create_blank_html_pages(pages, stylesheets=pages)
        self._make_about_page(pages)

    def _make_about_page(self, pages):
        """Make a page informing the user of the run directory, user that ran
        the job etc

        Parameters
        ----------
        pages: list
            list of pages you wish to create
        """
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="About"
        )
        html_file = self.setup_page(
            "About", self.navbar["home"], title="About"
        )
        html_file.make_banner(approximant="About", key="About")
        html_file.make_banner(
            approximant="On the command-line", key="command_line",
            _style="font-size: 26px;", link=os.getcwd()
        )
        command = ""
        for i in sys.argv:
            command += " "
            if i[0] == "-":
                command += "\n"
            command += "{}".format(i)
        html_file.make_container()
        styles = html_file.make_code_block(language="shell", contents=command)
        with open('{0:s}/css/About.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        html_file.export(
            "pesummary.sh", csv=False, json=False, shell=True,
            margin_top="-4em"
        )
        html_file.make_footer(
            user=self.user, rundir=self.webdir, fix_bottom=True
        )
        html_file.close()

    def make_downloads_page(self):
        """Wrapper function for _make_downloads_page
        """
        pages = ["Downloads"]
        self.create_blank_html_pages(pages)
        self._make_downloads_page(pages)

    def _make_downloads_page(self, pages, fix_bottom=False):
        """Make a page with links to files which can be downloaded

        Parameters
        ----------
        pages: list
            list of pages you wish to create
        """
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="Downloads"
        )
        html_file = self.setup_page(
            "Downloads", self.navbar["home"], title="Downloads"
        )
        html_file.make_banner(approximant="Downloads", key="Downloads")
        html_file.make_container()
        base_string = "{} can be downloaded <a href={} download>here</a>"
        style = "margin-top:{}; margin-bottom:{};"
        headings = ["Description"]
        metafile_row = [
            [
                base_string.format(
                    "The complete metafile containing all information "
                    "about the analysis",
                    self.results_path["other"] + self._metafile
                )
            ]
        ]
        if self.external_hdf5_links:
            string = ""
            for label in self.labels:
                string += (
                    "The hdf5 sub file for {} can be downloaded "
                    "<a href={} download>here</a>. ".format(
                        label, self.results_path["other"] + "_{}.h5".format(
                            label
                        )
                    )
                )
            metafile_row += [
                [
                    "The complete metafile uses external hdf5 links. Each "
                    "analysis is therefore stored in seperate meta files. "
                    "{}".format(string)
                ]
            ]
        metafile_row += [
            [
                (
                    "Information about reading this metafile can be seen "
                    " <a href={}>here</a>".format(
                        "https://lscsoft.docs.ligo.org/pesummary/"
                        "stable_docs/data/reading_the_metafile.html"
                    )
                )
            ]
        ]
        html_file.make_table(
            headings=headings, contents=metafile_row,
            accordian=False, style=style.format("1em", "1em")
        )
        for num, i in enumerate(self.labels):
            table_contents = self._make_entry_in_downloads_table(
                html_file, i, num, base_string
            )
            if table_contents is not None:
                html_file.make_table(
                    headings=headings, contents=table_contents, accordian=False
                )
        html_file.end_container()
        html_file.make_footer(
            user=self.user, rundir=self.webdir, fix_bottom=fix_bottom
        )
        html_file.close()

    def _make_entry_in_downloads_table(self, html_file, label, num, base_string):
        """Make a label specific entry into the downloads table

        Parameters
        ----------
        label: str
            the label you wish to add to the downloads table
        base_string: str
            the download string
        """
        import glob
        identifier = label if not self.mcmc_samples else "chain"
        _original = glob.glob(
            os.path.join(self.webdir, "samples", "{}_*".format(identifier))
        )
        html_file.add_content(
            "<div class='banner', style='margin-left:-4em'>{}</div>".format(
                label
            )
        )
        if not self.mcmc_samples and len(_original) > 0:
            table_contents = [
                [
                    base_string.format(
                        "Original file provided to PESummary",
                        self.results_path["other"] + Path(_original[0]).name
                    )
                ]
            ]
        elif self.mcmc_samples and len(_original) > 0:
            table_contents = [
                [
                    base_string.format(
                        "Original chain {} provided to PESummary".format(num),
                        self.results_path["other"] + Path(_original[num]).name
                    )
                ] for num in range(len(_original))
            ]
        else:
            table_contents = []
        table_contents.append(
            [
                base_string.format(
                    "Dat file containing posterior samples",
                    self.results_path["other"] + "%s_pesummary.dat" % (label)
                )
            ]
        )
        if self.config is not None and self.config[num] is not None:
            table_contents.append(
                [
                    base_string.format(
                        "Config file used for this analysis",
                        self.config_path["other"] + "%s_config.ini" % (label)
                    )
                ]
            )
        return table_contents

    def make_notes_page(self):
        """Wrapper function for _make_notes_page
        """
        pages = ["Notes"]
        self.create_blank_html_pages(pages, stylesheets=pages)
        self._make_notes_page(pages)

    def _make_notes_page(self, pages):
        """Make a page to display the custom notes

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="Notes"
        )
        html_file = self.setup_page(
            "Notes", self.navbar["home"], title="Notes"
        )
        html_file.make_banner(approximant="Notes", key="Notes")
        html_file.make_container()
        styles = html_file.make_code_block(
            language='shell', contents=self.notes
        )
        with open('{0:s}/css/Notes.css'.format(self.webdir), 'w') as f:
            f.write(styles)
        html_file.end_container()
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def generate_specific_javascript(self):
        """Tailor the javascript to the specific situation.
        """
        path = self.webdir + "/js/grab.js"
        existing = open(path)
        existing = existing.readlines()
        ind = existing.index("        if ( param == approximant ) {\n")
        content = existing[:ind]
        for i in [list(j.keys())[0] for j in self._result_page_links()]:
            content.append("        if ( param == \"%s\" ) {\n" % (i))
            content.append("            approx = \"None\" \n")
            content.append("        }\n")
        for i in existing[ind + 1:]:
            content.append(i)
        new_file = open(path, "w")
        new_file.writelines(content)
        new_file.close()

    def default_categories(self):
        """Return the default categories
        """
        categories = {
            "A-D": {
                "accept": ["a", "A", "b", "B", "c", "C", "d", "D"], "reject": []
            },
            "E-F": {
                "accept": ["e", "E", "f", "F", "g", "G", "h", "H"], "reject": []
            },
            "I-L": {
                "accept": ["i", "I", "j", "J", "k", "K", "l", "L"], "reject": []
            },
            "M-P": {
                "accept": ["m", "M", "n", "N", "o", "O", "p", "P"], "reject": []
            },
            "Q-T": {
                "accept": ["q", "Q", "r", "R", "s", "S", "t", "T"], "reject": []
            },
            "U-X": {
                "accept": ["u", "U", "v", "V", "w", "W", "x", "X"], "reject": []
            },
            "Y-Z": {
                "accept": ["y", "Y", "z", "Z"], "reject": []
            }
        }
        return categories

    def default_images_for_result_page(self, label):
        """Return the default images that will be displayed on the result page
        """
        return [], [], []

    def default_popular_options(self):
        """Return a list of default options
        """
        return []

    def default_comparison_homepage_plots(self):
        """Return a list of default plots for the comparison homepage
        """
        return []

    def default_corner_params(self):
        """Return a list of default corner parameters used by the corner
        plotting function
        """
        return []

    def add_existing_data(self):
        """
        """
        from pesummary.utils.utils import _add_existing_data

        self = _add_existing_data(self)

    def add_to_expert_pages(self, path, label):
        """Additional expert plots to add beyond the default. This returns a
        dictionary keyed by the parameter, with values providing the path
        to the additional plots you wish to add. The plots are a 2d list
        where each sublist represents a row in the table of images.

        Parameters
        ----------
        path: str
            path to the image directory
        label: str
            label of the plot you wish to add
        """
        mydict = {}
        contour_base = path + "{}_2d_contour_{}_{}.png"
        histogram_base = path + "{}_1d_posterior_{}_{}.png"
        if not hasattr(self.samples[label], "debug_keys"):
            return mydict
        for param in self.samples[label].debug_keys():
            if "_non_reweighted" in param:
                base_param = param.split("_non_reweighted")[0][1:]
                if base_param in self.samples[label].keys():
                    mydict[base_param] = [[
                        histogram_base.format(label, base_param, param),
                        contour_base.format(label, base_param, param)
                    ]]
        return mydict

    @property
    def additional_1d_pages(self):
        """Additional 1d histogram pages beyond one for each parameter. You may,
        for instance, want a 1d histogram page which combines multiple
        parameters. This returns a dictionary, keyed by the new 1d histogram
        page, with values indicating the parameters you wish to include on this
        page. Only the 1d marginalized histograms are shown.
        """
        return None
