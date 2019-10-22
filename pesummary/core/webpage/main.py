# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org>
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
from glob import glob
from scipy import stats
import numpy as np

import pesummary
from pesummary.utils.utils import logger
from pesummary.core.webpage import webpage


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
    """
    def __init__(
        self, webdir=None, samples=None, labels=None, publication=None,
        user=None, config=None, same_parameters=None, base_url=None,
        file_versions=None, hdf5=None, colors=None, custom_plotting=None,
        existing_labels=None, existing_config=None, existing_file_version=None,
        existing_injection_data=None, existing_samples=None,
        existing_metafile=None, existing_file_kwargs=None,
        add_to_existing=False, notes=None
    ):
        self.webdir = webdir
        self.samples = samples
        self.labels = labels
        self.publication = publication
        self.user = user
        self.config = config
        self.same_parameters = same_parameters
        self.base_url = base_url
        self.file_versions = file_versions
        self.hdf5 = hdf5
        self.colors = colors
        self.custom_plotting = custom_plotting
        self.existing_labels = existing_labels
        self.existing_config = existing_config
        self.existing_file_version = existing_file_version
        self.existing_samples = existing_samples
        self.existing_metafile = existing_metafile
        self.existing_file_kwargs = existing_file_kwargs
        self.add_to_existing = add_to_existing
        self.notes = notes
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
        if len(self.samples) > 1:
            try:
                self.comparison_stats = self.generate_comparison_statistics()
            except Exception as e:
                self.comparison_stats = None
                logger.info(
                    "Failed to generate comparison statistics because {}. As a "
                    "result they will not be added to the webpages".format(e)
                )

    def generate_comparison_statistics(self):
        """Generate comparison statistics for all parameters that are common to
        all result files
        """
        data = {
            i: self._generate_comparison_statistics(
                [self.samples[j][i] for j in self.labels]
            ) for i in self.same_parameters
        }
        return data

    def _generate_comparison_statistics(self, samples):
        """Generate comparison statistics for a set of samples

        Parameters
        ----------
        samples: list
            list of samples for each result file
        """
        from scipy.stats import gaussian_kde

        rows = range(len(samples))
        columns = range(len(samples))
        kernel = [gaussian_kde(i) for i in samples]
        x = np.linspace(
            np.min([np.min(i) for i in samples]),
            np.max([np.max(i) for i in samples]),
            100
        )
        ks = [
            [
                self.kolmogorov_smirnov_test(samples[i], samples[j]) for i in
                rows
            ] for j in columns
        ]
        js = [
            [
                self.jension_shannon_divergence(kernel[i](x), kernel[j](x)) for
                i in rows
            ] for j in columns
        ]
        return [ks, js]

    @staticmethod
    def get_executable(executable):
        """Return the path to an executable

        Parameters
        ----------
        executable: str
            the name of the executable you wish to find
        """
        from subprocess import check_output

        path = check_output(["which", executable]).decode("utf-8").strip()
        return path

    @staticmethod
    def kolmogorov_smirnov_test(a, b):
        """Return the KS p value between two PDFs

        Parameters
        ----------
        a: list
            List containing the first PDF that you would like to compare
        b: list
            List containing the second PDF that you would like to compare
        """
        return stats.ks_2samp(a, b)[1]

    @staticmethod
    def jension_shannon_divergence(a, b):
        """Return the JS divergence test between two PDFs

        Parameters
        ----------
        a: list
            List containing the first PDF that you would like to compare
        b: list
            List containing the second PDF that you would like to compare
        """
        a = np.asarray(a)
        b = np.asarray(b)
        a /= a.sum()
        b /= b.sum()
        m = 1. / 2 * (a + b)
        kl_forward = stats.entropy(a, qk=m)
        kl_backward = stats.entropy(b, qk=m)
        return kl_forward / 2. + kl_backward / 2.

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
        if len(self.samples) > 1:
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
            i: ["1d Histograms", [{"Multiple": i}]] for i in self.labels
        }
        for num, label in enumerate(self.labels):
            for j in self.categorize_parameters(self.samples[label].keys()):
                j = [j[0], [{k: label} for k in j[1]]]
                links[label].append(j)

        final_links = {
            i: [
                "home", ["Result Pages", self._result_page_links()],
                {"Corner": i}, {"Config": i}, links[i]
            ] for i in self.labels
        }
        if len(self.samples) > 1:
            for label in self.labels:
                final_links[label][1][1] += ["Comparison"]
        return final_links

    def make_navbar_for_comparison_page(self):
        """Make a navbar for the comparison homepage
        """
        if self.same_parameters is not None:
            links = ["1d Histograms", ["Multiple"]]
            for i in self.categorize_parameters(self.same_parameters):
                links.append(i)
            final_links = [
                "home", ["Result Pages", self._result_page_links()], links
            ]
            final_links[1][1] += ["Comparison"]
            return final_links
        return None

    def categorize_parameters(self, parameters):
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
                cond = self._condition(category["accept"], category["reject"])
                params.append(
                    [heading, self._partition(cond, parameters)]
                )
        return params

    def _condition(self, true, false):
        """Setup a condition

        Parameters
        ----------
        true: list
            list of strings that you would like to include
        false: list
            list of strings that you would like to neglect
        """
        if len(true) != 0 and len(false) == 0:
            condition = lambda j: True if any(i in j[0] for i in true) else \
                False
        elif len(true) == 0 and len(false) != 0:
            condition = lambda j: True if any(i not in j[0] for i in false) \
                else False
        elif len(true) and len(false) != 0:
            condition = lambda j: True if any(i in j[0] for i in true) and \
                any(i not in j[0] for i in false) else False
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
        if len(self.labels) > 1:
            self.make_comparison_pages()
        self.make_error_page()
        self.make_version_page()
        self.make_logging_page()
        if self.notes is not None:
            self.make_notes_page()
        self.generate_specific_javascript()

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
        background_colour=None, histogram_download=False
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
        html_file.make_header(approximant=approximant)
        if html_page == "home" or html_page == "home.html":
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour,
                hdf5=self.hdf5
            )
        elif histogram_download:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["other"],
                histogram_download=os.path.join(
                    "..", "samples", "dat", label, "{}_{}_samples.dat".format(
                        label, html_page
                    )
                ), background_color=background_colour, hdf5=self.hdf5
            )
        else:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour, hdf5=self.hdf5
            )
        return html_file

    def make_home_pages(self):
        """Wrapper function for _make_home_pages()
        """
        pages = ["{}_{}".format(i, i) for i in self.labels]
        pages.append("home")
        self.create_blank_html_pages(pages)
        self._make_home_pages(pages)

    def _make_home_pages(self, pages):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        html_file = self.setup_page("home", self.navbar["home"])
        html_file.make_banner(approximant="Summary", key="Summary")

        for num, i in enumerate(self.labels):
            html_file = self.setup_page(
                i, self.navbar["result_page"][i], i,
                title="{} Summary page".format(i),
                background_colour=self.colors[num], approximant=i
            )
            html_file.make_banner(approximant=i, key=i)
            if self.custom_plotting:
                custom_plots = glob(
                    "{}/plots/{}_custom_plotting_*".format(self.webdir, i)
                )
                path = self.image_path["other"]
                for num, i in enumerate(custom_plots):
                    custom_plots[num] = path + i.split("/")[-1]
                image_contents = [
                    custom_plots[i:4 + i] for i in range(0, len(custom_plots), 4)]
                html_file.make_table_of_images(contents=image_contents)
                images = [y for x in image_contents for y in x]
                html_file.make_modal_carousel(images=images)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_1d_histogram_pages(self):
        """Wrapper function for _make_1d_histogram pages
        """
        pages = [
            "{}_{}_{}".format(i, i, j) for i in self.labels for j in
            self.samples[i].keys()
        ]
        pages += ["{}_{}_Multiple".format(i, i) for i in self.labels]
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
            for j in self.samples[i].keys():
                html_file = self.setup_page(
                    "{}_{}".format(i, j), self.navbar["result_page"][i],
                    i, title="{} Posterior PDF for {}".format(i, j),
                    approximant=i, background_colour=self.colors[num],
                    histogram_download=True
                )
                html_file.make_banner(approximant=i, key=i)
                path = self.image_path["other"]
                contents = [
                    [path + "{}_1d_posterior_{}.png".format(i, j)],
                    [
                        path + "{}_sample_evolution_{}.png".format(i, j),
                        path + "{}_autocorrelation_{}.png".format(i, j)
                    ]
                ]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage")
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()
            html_file = self.setup_page(
                "{}_Multiple".format(i), self.navbar["result_page"][i],
                i, title="{} Posteriors for multiple".format(i),
                approximant=i, background_colour=self.colors[num]
            )
            html_file.make_banner(approximant=i, key=i)
            ordered_parameters = self.categorize_parameters(
                self.samples[i].keys()
            )
            ordered_parameters = [i for j in ordered_parameters for i in j[1]]
            popular_options = self.popular_options
            popular_options.append({"all": ", ".join(ordered_parameters)})
            html_file.make_search_bar(
                sidebar=[i for i in self.samples[i].keys()],
                popular_options=popular_options,
                label=self.labels[num], code="combines"
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
            html_file.make_search_bar(
                sidebar=self.samples[i].keys(),
                popular_options=self.popular_options, label=i
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
            else:
                html_file.add_content(
                    "<div class='row justify-content-center'; "
                    "style='font-family: Arial-body; font-size: 14px'>"
                    "<p style='margin-top:2.5em'> No configuration file was "
                    "provided </p></div>"
                )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_comparison_pages(self):
        """Wrapper function for _make_comparison_pages
        """
        pages = ["Comparison_{}".format(i) for i in self.same_parameters]
        pages += ["Comparison_Multiple"]
        pages += ["Comparison"]
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
        if self.custom_plotting:
            custom_plots = glob(
                "{}/plots/combined_custom_plotting_*".format(self.webdir)
            )
            for num, i in enumerate(custom_plots):
                custom_plots[num] = path + i.split("/")[-1]
            image_contents = [
                custom_plots[i:4 + 1] for i in range(0, len(custom_plots), 4)
            ]
            html_file.make_table_of_images(contents=image_contents)
            images = [y for x in image_contents for y in x]
            html_file.make_modal_carousel(images=images)
        path = self.image_path["other"]
        if self.comparison_stats is not None:
            rows = range(len(self.labels))
            style = (
                "margin-top:0em; margin-bottom:{}; background-color:#FFFFFF; "
                "box-shadow: 0 0 5px grey;"
            )
            table_contents = {
                i: [
                    [self.labels[j]] + self.comparison_stats[i][0][j] for j in
                    rows
                ] for i in self.same_parameters
            }
            html_file.make_table(
                headings=[" "] + self.labels, contents=table_contents,
                heading_span=1, accordian_header="KS test total",
                style=style.format("1em")
            )
            table_contents = {
                i: [
                    [self.labels[j]] + self.comparison_stats[i][1][j] for j in
                    rows
                ] for i in self.same_parameters
            }
            html_file.make_table(
                headings=[" "] + self.labels, contents=table_contents,
                heading_span=1, accordian_header="JS divergence test total",
                style=style.format("5em")
            )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        for i in self.same_parameters:
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
            if self.comparison_stats is not None:
                table_contents = [
                    [self.labels[j]] + self.comparison_stats[i][0][j] for j in rows
                ]
                html_file.make_table(
                    headings=[" "] + self.labels, contents=table_contents,
                    heading_span=1, accordian_header="KS test",
                    style=style.format("1em")
                )
                table_contents = [
                    [self.labels[j]] + self.comparison_stats[i][1][j] for j in rows
                ]
                html_file.make_table(
                    headings=[" "] + self.labels, contents=table_contents,
                    heading_span=1, accordian_header="JS divergence test",
                    style=style.format("1em")
                )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        html_file = self.setup_page(
            "Comparison_Multiple", self.navbar["comparison"],
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
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.base_url, html_page="Version"
        )
        html_file = self.setup_page(
            "Version", self.navbar["home"], title="Version Information"
        )
        html_file.make_banner(approximant="Version", key="Version")
        path = pesummary.__file__[:-12]
        with open(path + "/.version", 'r') as f:
            contents = f.read()
        for i in self.labels:
            contents = (
                "# {} version information\n\n{}_version={}\n\n".format(
                    i, i, self.file_versions[i]
                )
            ) + contents
        for i in self.labels:
            contents = "# %s version information\n\n%s_version = %s\n\n" % (
                i, i, self.file_versions[i]) + contents
        html_file.make_container()
        styles = html_file.make_code_block(language='shell', contents=contents)
        with open('{0:s}/css/Version.css'.format(self.webdir), 'w') as f:
            f.write(styles)
        html_file.end_container()
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
        log_file = glob(".tmp/pesummary/*/pesummary.log")
        if len(log_file) == 0:
            log_file = ".tmp/pesummary/no_log_information.log"
            with open(log_file, "w") as f:
                f.writelines(["No log information stored"])
        else:
            log_file = log_file[0]
        with open(log_file, 'r') as f:
            contents = f.read()
        html_file.make_container()
        styles = html_file.make_code_block(language='shell', contents=contents)
        with open('{0:s}/css/Logging.css'.format(self.webdir), 'w') as f:
            f.write(styles)
        html_file.end_container()
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

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
        ind = existing.index("    if ( param == approximant ) {\n")
        content = existing[:ind]
        for i in [list(j.keys())[0] for j in self._result_page_links()]:
            content.append("    if ( param == \"%s\" ) {\n" % (i))
            content.append("        approx = \"None\" \n")
            content.append("    }\n")
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

    def default_popular_options(self):
        """Return a list of default options
        """
        return []

    def add_existing_data(self):
        """
        """
        from pesummary.utils.utils import _add_existing_data

        self = _add_existing_data(self)
