# Licensed under an MIT style license -- see LICENSE.md

import os
from .main import _WebpageGeneration as _GWWebpageGeneration
from pesummary.core.webpage import webpage

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _PublicWebpageGeneration(_GWWebpageGeneration):
    """
    """
    def __init__(
        self, webdir=None, samples=None, labels=None, publication=None,
        user=None, config=None, same_parameters=None, base_url=None,
        file_versions=None, hdf5=None, colors=None, custom_plotting=None,
        pepredicates_probs=None, gracedb=None, approximant=None, key_data=None,
        file_kwargs=None, existing_labels=None, existing_config=None,
        existing_file_version=None, existing_injection_data=None,
        existing_samples=None, existing_metafile=None, add_to_existing=False,
        existing_file_kwargs=None, existing_weights=None, result_files=None,
        notes=None, disable_comparison=False, pastro_probs=None, gwdata=None,
        disable_interactive=False, publication_kwargs={}, no_ligo_skymap=False,
        psd=None, priors=None, package_information={"packages": []},
        mcmc_samples=False, external_hdf5_links=False,
        preliminary_pages=False, existing_plot=None, disable_expert=False,
        analytic_priors=None
    ):
        super(_PublicWebpageGeneration, self).__init__(
            webdir=webdir, samples=samples, labels=labels,
            publication=publication, user=user, config=config,
            same_parameters=same_parameters, base_url=base_url,
            file_versions=file_versions, hdf5=hdf5, colors=colors,
            custom_plotting=custom_plotting,
            pepredicates_probs=pepredicates_probs, gracedb=gracedb,
            approximant=approximant, key_data=key_data, file_kwargs=file_kwargs,
            existing_labels=existing_labels, existing_config=existing_config,
            existing_file_version=existing_file_version,
            existing_injection_data=existing_injection_data,
            existing_samples=existing_samples,
            existing_metafile=existing_metafile,
            add_to_existing=add_to_existing,
            existing_file_kwargs=existing_file_kwargs,
            existing_weights=existing_weights, result_files=result_files,
            notes=notes, disable_comparison=disable_comparison,
            pastro_probs=pastro_probs, gwdata=gwdata,
            disable_interactive=disable_interactive,
            publication_kwargs=publication_kwargs,
            no_ligo_skymap=no_ligo_skymap, psd=psd, priors=priors,
            package_information=package_information,
            mcmc_samples=mcmc_samples, external_hdf5_links=external_hdf5_links,
            preliminary_pages=preliminary_pages, existing_plot=existing_plot,
            disable_expert=disable_expert, analytic_priors=analytic_priors
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
                hdf5=self.hdf5, about=False
            )
        elif histogram_download:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["other"],
                histogram_download=os.path.join(
                    "..", "samples", "dat", label, "{}_{}_samples.dat".format(
                        label, html_page
                    )
                ), background_color=background_colour, hdf5=self.hdf5,
                about=False
            )
        else:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour, hdf5=self.hdf5,
                about=False
            )
        return html_file

    def make_navbar_for_homepage(self):
        """Make a navbar for the homepage
        """
        links = [
            "home", ["Result Pages", self._result_page_links()], "Version"
        ]
        if len(self.samples) > 1:
            links[1][1] += ["Comparison"]
        if self.publication:
            links.insert(2, "Publication")
        if self.gwdata is not None:
            links.append(["Detchar", [i for i in self.gwdata.keys()]])
        if self.notes is not None:
            links.append("Notes")
        return links

    def make_navbar_for_result_page(self):
        """Make a navbar for the result page homepage
        """
        links = {
            i: ["1d Histograms", [{"Custom": i}, {"All": i}]] for i in self.labels
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
            for i in self.labels:
                final_links[i][1][1] += ["Comparison"]
        if self.make_interactive:
            for label in self.labels:
                final_links[label].append(
                    ["Interactive", [{"Interactive_Corner": label}]]
                )
        return final_links

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
        if self.publication:
            self.make_publication_pages()
        if self.gwdata is not None:
            self.make_detector_pages()
        self.make_error_page()
        self.make_version_page()
        if self.notes is not None:
            self.make_notes_page()
        self.make_downloads_page()
        self.generate_specific_javascript()
