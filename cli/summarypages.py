import pesummary
from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.utils import functions
from .summaryplots import PlotGeneration, GWPlotGeneration
from pesummary.core.file.meta_file import MetaFile
from pesummary.gw.file.meta_file import GWMetaFile

import numpy as np
from scipy import stats

import os

from pesummary.utils.utils import logger
from pesummary.core.webpage import webpage

__doc__ == "Classes to generate webpages"


class WebpageGeneration(pesummary.core.inputs.PostProcessing):
    """Class to generate all webpages

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    navbar_for_homepage: list
        List containing the structure of the homepage navbar for each results
        file
    navbar_for_approximant_homepage: list
        List containing the structure of the approximant homepage navbar for
        each results file
    navbar_for_comparison_homepage: list
        List containing the structure of the comparison navbar
    """
    def __init__(self, inputs, colors="default"):
        super(WebpageGeneration, self).__init__(inputs, colors)
        logger.info("Starting to generate webpages")
        self.same_parameters = []
        self.navbar_for_homepage = []
        self.navbar_for_approximant_homepage = []
        self.navbar_for_comparison_homepage = []
        self.image_path = {"home": "./plots/", "other": "../plots/"}
        self.results_path = {"home": "./samples", "other": "../samples"}
        self.generate_webpages()
        logger.info("Finished generating webpages")
        logger.debug("Tailoring the javascript")
        self.generate_specific_javascript()
        logger.debug("Finished Tailoring the javascript")

    @property
    def navbar_for_homepage(self):
        return self._navbar_for_homepage

    @navbar_for_homepage.setter
    def navbar_for_homepage(self, navbar_for_homepage):
        approximant_links = self._approximant_navbar_links()
        links = ["home", ["Approximants", approximant_links]]
        if len(self.result_files) > 1:
            links[1][1] = links[1][1] + ["Comparison"]
        self._navbar_for_homepage = links

    @property
    def navbar_for_approximant_homepage(self):
        return self._navbar_for_approximant_homepage

    @navbar_for_approximant_homepage.setter
    def navbar_for_approximant_homepage(self, navbar_for_approximant_homepage):
        links = [["1d_histograms", [{"multiple": i}]] for i in self.labels]
        for num, i in enumerate(self.parameters):
            for j in self._categorize_parameters(i):
                j = [j[0], [{k: self.labels[num]} for k in j[1]]]
                links[num].append(j)
        final_links = [[
            "home", ["Approximants", self._approximant_navbar_links()],
            {"corner": self.labels[num]}, {"config": self.labels[num]}, j]
            for num, j in enumerate(links)]
        if len(self.result_files) > 1:
            for i in final_links:
                i[1][1] = i[1][1] + ["Comparison"]
        self._navbar_for_approximant_homepage = final_links

    @property
    def navbar_for_comparison_homepage(self):
        return self._navbar_for_comparison_homepage

    @navbar_for_comparison_homepage.setter
    def navbar_for_comparison_homepage(self, navbar_for_comparison_homepage):
        links = ["1d_histograms", ["multiple"]]
        for i in self._categorize_parameters(self.same_parameters):
            links.append(i)
        final_links = [
            "home", ["Approximants", self._approximant_navbar_links()], links]
        final_links[1][1] = final_links[1][1] + ["Comparison"]
        self._navbar_for_comparison_homepage = final_links

    @property
    def comparison_statistics(self):
        data = []
        for ind, j in enumerate(self.same_parameters):
            indices = [k.index("%s" % (j)) for k in self.parameters]
            param_samples = [[k[indices[num]] for k in l] for num, l in
                             enumerate(self.samples)]
            data.append(self.compute_comparison_statistics(param_samples))
        return data

    @staticmethod
    def compute_comparison_statistics(samples):
        """Return the comparison statistics for a list of PDFs

        Parameters
        ----------
        samples: list
            list of samples for different PDFs
        """
        rows = range(len(samples))
        columns = range(len(samples))
        kernel = [stats.gaussian_kde(i) for i in samples]
        x = np.linspace(np.min([np.min(i) for i in samples]),
                        np.max([np.max(i) for i in samples]), 100)
        ks = [[WebpageGeneration._kolmogorov_smirnov_test(samples[i],
              samples[j]) for i in rows] for j in columns]
        js = [[WebpageGeneration._jenson_shannon_divergence(kernel[i](x),
              kernel[j](x)) for i in rows] for j in columns]
        return [ks, js]

    @staticmethod
    def _kolmogorov_smirnov_test(a, b):
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
    def _jenson_shannon_divergence(a, b):
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

    def _approximant_navbar_links(self):
        """Return the navbar structure for the approximant tab. If we have
        passed a coherence test, then we need to prepend the approximant by
        a label so we can determine which tab corresponds to which network.
        """
        return [{j: j} for j in self.labels]

    def _categorize_parameters(self, parameters):
        """Categorize the parameters into common headings.

        Parameters
        ----------
        parameters: list
            List of parameters that you would like to sort
        """
        params = []
        if any(any(i[0] in j for j in ["a", "A", "b", "B", "c", "C", "d", "D"])
                for i in parameters):
            cond = self._condition(["a", "A", "b", "B", "c", "C", "d", "D"], [])
            params.append(["A-D", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["e", "E", "f", "F", "g", "G", "h", "H"])
                for i in parameters):
            cond = self._condition(["e", "E", "f", "F", "g", "G", "h", "H"], [])
            params.append(["E-H", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["i", "I", "j", "J", "k", "K", "l", "L"])
                for i in parameters):
            cond = self._condition(["i", "I", "j", "J", "k", "K", "l", "L"], [])
            params.append(["I-L", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["m", "M", "n", "N", "o", "O", "p", "P"])
                for i in parameters):
            cond = self._condition(["m", "M", "n", "N", "o", "O", "p", "P"], [])
            params.append(["M-P", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["q", "Q", "r", "R", "s", "S", "t", "T"])
                for i in parameters):
            cond = self._condition(["q", "Q", "r", "R", "s", "S", "t", "T"], [])
            params.append(["Q-T", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["u", "U", "v", "V", "w", "W", "x", "X"])
                for i in parameters):
            cond = self._condition(["u", "U", "v", "V", "w", "W", "x", "X"], [])
            params.append(["U-X", self._partition(cond, parameters)])
        if any(any(i[0] in j for j in ["y", "Y", "z", "Z"])
                for i in parameters):
            cond = self._condition(["y", "Y", "z", "Z"], [])
            params.append(["Y-Z", self._partition(cond, parameters)])
        return params

    def _condition(self, true, false):
        """Set up a condition.

        Parameters
        ----------
        true: list
            list of strings that you would like to include
        false: list
            list of strings that you would like to neglect
        """
        if len(true) != 0 and len(false) == 0:
            condition = lambda j: True if any(i in j[0] for i in true) else False
        if len(true) == 0 and len(false) != 0:
            condition = lambda j: True if any(i not in j[0] for i in false) else \
                False
        if len(true) and len(false) != 0:
            condition = lambda j: True if any(i in j[0] for i in true) and \
                any(i not in j[0] for i in false) else False
        return condition

    def _partition(self, condition, array):
        """Filter the list according to a condition

        Parameters
        ----------
        condition: lambda function
            Lambda function containing the condition that you want to use to
            filter the array
        array: list
            List of parameters that you would like to filter
        """
        return list(filter(condition, array))

    def generate_webpages(self):
        """Generate all webpages that we need.
        """
        self.make_home_pages()
        self.make_1d_histogram_pages()
        self.make_corner_pages()
        self.make_config_pages()
        if len(self.labels) > 1:
            self.make_comparison_pages()
        self.make_error_page()

    def _setup_page(self, html_page, links, label=None, title=None,
                    approximant=None, background_colour=None,
                    histogram_download=False):
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
            web_dir=self.webdir, base_url=self.baseurl, html_page=html_page,
            label=label)
        html_file.make_header(
            approximant=approximant)
        if html_page == "home" or html_page == "home.html":
            html_file.make_navbar(
                links=links, samples_path=self.results_path["home"],
                background_color=background_colour,
                hdf5=self.hdf5)
        elif histogram_download:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["other"],
                histogram_download="../samples/dat/%s/%s_%s_samples.dat" % (
                    label, label, html_page),
                background_color=background_colour,
                hdf5=self.hdf5)
        else:
            html_file.make_navbar(
                links=links, samples_path=self.results_path["other"],
                background_color=background_colour,
                hdf5=self.hdf5)
        return html_file

    def make_home_pages(self):
        """Make the home pages.
        """
        pages = ["%s_%s" % (i, i) for i in self.labels]
        pages.append("home")
        webpage.make_html(web_dir=self.webdir, pages=pages)
        html_file = self._setup_page("home", self.navbar_for_homepage)
        html_file.make_banner(approximant="Summary", key="Summary")

        for num, i in enumerate(self.labels):
            html_file = self._setup_page(
                i, self.navbar_for_approximant_homepage[num],
                i, title="%s Summary page" % (i),
                background_colour=self.colors[num], approximant=i)
            html_file.make_banner(approximant=i, key=i)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_1d_histogram_pages(self):
        """Make the 1d histogram pages.
        """
        pages = ["%s_%s_%s" % (i, i, j) for num, i in enumerate(self.labels)
                 for j in self.parameters[num]]
        pages += ["%s_%s_multiple" % (i, i) for i in self.labels]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.labels):
            for j in self.parameters[num]:
                html_file = self._setup_page(
                    "%s_%s" % (app, j), self.navbar_for_approximant_homepage[num],
                    app, title="%s Posterior PDF for %s" % (app, j),
                    approximant=app, background_colour=self.colors[num],
                    histogram_download=True)
                html_file.make_banner(approximant=app, key=app)
                path = self.image_path["other"]
                contents = [[path + "%s_1d_posterior_%s.png" % (app, j)],
                            [path + "%s_sample_evolution_%s.png" % (app, j),
                             path + "%s_autocorrelation_%s.png" % (app, j)]]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage")
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()
            html_file = self._setup_page(
                "%s_multiple" % (app), self.navbar_for_approximant_homepage[num],
                app, title="%s Posteriors for multiple" % (app),
                approximant=app, background_colour=self.colors[num])
            html_file.make_banner(approximant=app, key=app)
            ordered_parameters = self._categorize_parameters(self.parameters[num])
            ordered_parameters = [i for j in ordered_parameters for i in j[1]]
            html_file.make_search_bar(sidebar=[i for i in self.parameters[num]],
                                      popular_options=[
                                          {"all": ", ".join(ordered_parameters)}],
                                      label=self.labels[num],
                                      code="combines")
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_corner_pages(self):
        """Make the corner pages.
        """
        pages = ["%s_%s_corner" % (i, i) for i in self.labels]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.labels):
            html_file = self._setup_page(
                "%s_corner" % (app), self.navbar_for_approximant_homepage[num],
                app, title="%s Corner Plots" % (app),
                background_colour=self.colors[num], approximant=app)
            html_file.make_banner(approximant=app, key="corner")
            html_file.make_search_bar(sidebar=self.parameters[num],
                                      popular_options=[],
                                      label=self.labels[num])
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_config_pages(self):
        """Make the configuration pages.
        """
        pages = ["%s_%s_config" % (i, i) for i in self.labels]
        webpage.make_html(web_dir=self.webdir, pages=pages, stylesheets=pages)
        for num, app in enumerate(self.labels):
            html_file = self._setup_page(
                "%s_config" % (app), self.navbar_for_approximant_homepage[num],
                app, title="%s configuration" % (app),
                background_colour=self.colors[num], approximant=app)
            html_file.make_banner(approximant=app, key="config")
            if self.config and num < len(self.config):
                with open(self.config[num], 'r') as f:
                    contents = f.read()
                styles = html_file.make_code_block(language='ini', contents=contents)
                with open('{0:s}/css/{1:s}_{2:s}_config.css'.format(self.webdir,
                          self.labels[num], self.labels[num]), 'w') as f:
                    f.write(styles)
            else:
                html_file.add_content(
                    "<div class='row justify-content-center'; style='font-family: Arial-body;"
                    "font-size: 14px'>"
                    "<p style='margin-top:2.5em'> No configuration file was "
                    "provided </p></div>")
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_comparison_pages(self):
        """Make the comparison pages.
        """
        webpage.make_html(web_dir=self.webdir, pages=["Comparison"])
        html_file = self._setup_page(
            "Comparison", self.navbar_for_comparison_homepage,
            title="Comparison Summary Page", approximant="Comparison")
        html_file.make_banner(approximant="Comparison", key="Comparison")
        path = self.image_path["other"]
        try:
            statistics = self.comparison_statistics
        except Exception as e:
            statistics = None
            logger.info("Failed to generate comparison statistics because "
                        "%s" % (e))
        if statistics:
            rows = range(len(self.result_files))
            style_ks = "margin-top:5em; margin-bottom:1em; background-color:#FFFFFF; " + \
                "box-shadow: 0 0 5px grey;"
            style_js = "margin-top:0em; margin-bottom:5em; background-color:#FFFFFF; " + \
                "box-shadow: 0 0 5px grey;"
            table_contents = {self.same_parameters[num]: [
                [self.labels[i]] + statistics[num][0][i] for i in rows]
                for num in range(len(self.same_parameters))}
            html_file.make_table(headings=[" "] + self.labels,
                                 contents=table_contents, heading_span=1,
                                 accordian_header="KS test total", style=style_ks)
            table_contents = {self.same_parameters[num]: [
                [self.labels[i]] + statistics[num][1][i] for i in rows]
                for num in range(len(self.same_parameters))}
            html_file.make_table(headings=[" "] + self.labels,
                                 contents=table_contents, heading_span=1,
                                 accordian_header="JS divergence test total", style=style_js)
        html_file.make_footer(user=self.user, rundir=self.webdir)
        pages = ["Comparison_%s" % (i) for i in self.same_parameters]
        pages += ["Comparison_multiple"]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, i in enumerate(self.same_parameters):
            html_file = self._setup_page(
                "Comparison_%s" % (i),
                self.navbar_for_comparison_homepage,
                title="Comparison PDF for %s" % (i),
                approximant="Comparison")
            html_file.make_banner(approximant="Comparison", key="Comparison")
            html_file.insert_image(path + "combined_1d_posterior_%s.png" % (i),
                                   code="changeimage")
            if statistics:
                table_contents = [
                    [self.labels[i]] + statistics[num][0][i] for i in rows]
                html_file.make_table(headings=[" "] + self.labels,
                                     contents=table_contents, heading_span=1,
                                     accordian_header="KS test", style=style_ks)
                table_contents = [
                    [self.labels[i]] + statistics[num][1][i] for i in rows]
                html_file.make_table(headings=[" "] + self.labels,
                                     contents=table_contents, heading_span=1,
                                     accordian_header="JS divergence test",
                                     style=style_js)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        html_file = self._setup_page(
            "Comparison_multiple", self.navbar_for_comparison_homepage,
            approximant="Comparison", title="Comparison Posteriors for multiple")
        html_file.make_search_bar(sidebar=self.same_parameters,
                                  popular_options=[
                                      "mass_1, mass_2",
                                      "luminosity_distance, iota, ra, dec",
                                      "iota, phi_12, phi_jl, tilt_1, tilt_2",
                                      {"all": ", ".join(self.same_parameters)}],
                                  label="None", code="combines")
        html_file.make_footer(user=self.user, rundir=self.webdir)

    def make_error_page(self):
        """Make a page that is shown when something goes wrong.
        """
        pages = ["error"]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        html_file = webpage.open_html(
            web_dir=self.webdir, base_url=self.baseurl, html_page="error")
        html_file.make_div(_class='jumbotron text-center',
                           _style='background-color: #D3D3D3; margin-botton:0')
        html_file.make_div(_class='container',
                           _style=('margin-top:1em; background-color: #D3D3D3;'
                                   'width: 45em'))
        html_file.add_content(
            "<h1 style='color:white; font-size: 8em'>404</h1>")
        html_file.add_content(
            "<h2 style='color:white;'> Something went wrong... </h2>")
        html_file.end_div()
        html_file.end_div()
        html_file.close()

    def generate_specific_javascript(self):
        """Tailor the javascript to the specific situation.
        """
        path = self.webdir + "/js/grab.js"
        existing = open(path)
        existing = existing.readlines()
        ind = existing.index("    if ( param == approximant ) {\n")
        content = existing[:ind]
        for i in [list(j.keys())[0] for j in self._approximant_navbar_links()]:
            content.append("    if ( param == \"%s\" ) {\n" % (i))
            content.append("        approx = \"None\" \n")
            content.append("    }\n")
        for i in existing[ind + 1:]:
            content.append(i)
        new_file = open(path, "w")
        new_file.writelines(content)
        new_file.close()


class GWWebpageGeneration(pesummary.gw.inputs.GWPostProcessing, WebpageGeneration):
    def __init__(self, inputs, colors="default"):
        super(GWWebpageGeneration, self).__init__(inputs, colors)
        logger.info("Starting to generate webpages")
        self.same_parameters = []
        self.navbar_for_homepage = []
        self.navbar_for_approximant_homepage = []
        self.navbar_for_comparison_homepage = []
        self.image_path = {"home": "./plots/", "other": "../plots/"}
        self.results_path = {"home": "./samples", "other": "../samples"}
        self.generate_webpages()
        logger.info("Finished generating webpages")
        logger.debug("Tailoring the javascript")
        self.generate_specific_javascript()
        logger.debug("Finished Tailoring the javascript")

    def _categorize_parameters(self, parameters):
        """Categorize the parameters into common headings.

        Parameters
        ----------
        parameters: list
            List of parameters that you would like to sort
        """
        params = []
        if any("mass" in j for j in parameters):
            cond = self._condition(["mass", "q", "symmetric_mass_ratio"],
                                   ["source"])
            params.append(["masses", self._partition(cond, parameters)])
            cond = self._condition(["source"], [])
            params.append(["source", self._partition(cond, parameters)])
        if any("theta" in j for j in parameters):
            cond = self._condition(["theta", "iota"], [])
            params.append(["inclination", self._partition(cond, parameters)])
        if any("a_1" in j for j in parameters):
            cond = self._condition(["spin", "chi_p", "chi_eff", "a_1", "a_2"],
                                   [])
            params.append(["spins", self._partition(cond, parameters)])
        if any("phi" in j for j in parameters):
            cond = self._condition(["phi", "tilt"], [])
            params.append(["spin_angles", self._partition(cond, parameters)])
        if any("ra" in j for j in parameters):
            cond = self._condition(
                ["ra", "dec", "psi", "luminosity_distance", "redshift",
                 "comoving_distance"], ["mass_ratio"])
            params.append(["location", self._partition(cond, parameters)])
        if any("geocent_time" in j for j in parameters):
            cond = self._condition(["geocent_time"], [])
            params.append(["timings", self._partition(cond, parameters)])
        if any("snr" in j for j in parameters):
            cond = self._condition(["snr"], [])
            params.append(["SNR", self._partition(cond, parameters)])
        if any("phase" in j for j in parameters):
            cond = self._condition(["phase", "likelihood"], [])
            params.append(["others", self._partition(cond, parameters)])
        return params

    def _condition(self, true, false):
        """Set up a condition.

        Parameters
        ----------
        true: list
            list of strings that you would like to include
        false: list
            list of strings that you would like to neglect
        """
        if len(true) != 0 and len(false) == 0:
            condition = lambda j: True if any(i in j for i in true) else False
        if len(true) == 0 and len(false) != 0:
            condition = lambda j: True if any(i not in j for i in false) else \
                False
        if len(true) and len(false) != 0:
            condition = lambda j: True if any(i in j for i in true) and \
                any(i not in j for i in false) else False
        return condition

    def _partition(self, condition, array):
        """Filter the list according to a condition

        Parameters
        ----------
        condition: lambda function
            Lambda function containing the condition that you want to use to
            filter the array
        array: list
            List of parameters that you would like to filter
        """
        return list(filter(condition, array))

    def make_home_pages(self):
        """Make the home pages.
        """
        pages = ["%s_%s" % (i, i) for i in self.labels]
        pages.append("home")
        webpage.make_html(web_dir=self.webdir, pages=pages)
        if self.gracedb:
            html_file = self._setup_page(
                "home", self.navbar_for_homepage,
                title="Parameter Estimation Summary Pages for %s" % (
                    self.gracedb))
        else:
            html_file = self._setup_page("home", self.navbar_for_homepage)
        if self.gracedb:
            html_file.make_banner(approximant="Summary for %s" % (self.gracedb),
                                  key="Summary")
        else:
            html_file.make_banner(approximant="Summary", key="Summary")
        path = self.image_path["home"]
        image_contents = []

        if len(self.labels) > 1:
            if self.approximant:
                image_contents.append(
                    path + "compare_time_domain_waveforms.png")
            image_contents = [image_contents]
            html_file.make_table_of_images(contents=image_contents)
            images = [y for x in image_contents for y in x]
            html_file.make_modal_carousel(images=images)
            for num, i in enumerate(self.labels):
                html_file.make_banner(approximant=i, key=i)
                image_contents = []
                if os.path.isfile(self.webdir + "/plots/%s_psd_plot.png" % (i)):
                    image_contents.append(path + "%s_psd_plot.png" % (i))
                if os.path.isfile(self.webdir + "/plots/%s_waveform_timedomain.png" % (i)):
                    image_contents.append(path + "%s_waveform_timedomain.png" % (i))
                if os.path.isfile(self.webdir + "/plots/%s_calibration_plot.png" % (i)):
                    image_contents.append(path + "%s_calibration_plot.png" % (i))
                image_contents = [image_contents]
                html_file.make_table_of_images(contents=image_contents)
                images = [y for x in image_contents for y in x]
                html_file.make_modal_carousel(images=images)
        else:
            if self.psds:
                image_contents.append(path + "%s_psd_plot.png" % (
                    self.labels[0]))
            if self.approximant:
                image_contents.append(path + "%s_waveform_timedomain.png" % (
                    self.labels[0]))
            if self.calibration:
                image_contents.append(path + "%s_calibration_plot.png" % (
                    self.labels[0]))
            image_contents = [image_contents]
            html_file.make_table_of_images(contents=image_contents)
            images = [y for x in image_contents for y in x]
            html_file.make_modal_carousel(images=images)

        html_file.close()
        key_data = self._key_data()
        table_data = [{j: i[j] for j in self.same_parameters} for i in
                      key_data]
        contents = []
        for i in self.same_parameters:
            row = []
            row.append(i)
            for j in range(len(table_data)):
                row.append(np.round(table_data[j][i]["maxL"], 3))
            for j in range(len(table_data)):
                row.append(np.round(table_data[j][i]["mean"], 3))
            for j in range(len(table_data)):
                row.append(np.round(table_data[j][i]["median"], 3))
            for j in range(len(table_data)):
                row.append(np.round(table_data[j][i]["std"], 3))
            contents.append(row)
        for num, i in enumerate(self.labels):
            html_file = self._setup_page(
                i, self.navbar_for_approximant_homepage[num],
                self.labels[num], title="%s Summary page" % (i),
                background_colour=self.colors[num], approximant=i)
            html_file.make_banner(approximant=i, key=i)
            path = self.image_path["other"]
            image_contents = [[path + "%s_1d_posterior_mass_1.png" % (i),
                               path + "%s_1d_posterior_mass_2.png" % (i),
                               path + "%s_1d_posterior_a_1.png" % (i),
                               path + "%s_1d_posterior_a_2.png" % (i)],
                              [path + "%s_skymap.png" % (i),
                               path + "%s_waveform.png" % (i),
                               path + "%s_1d_posterior_iota.png" % (i),
                               path + "%s_1d_posterior_luminosity_distance.png" % (i)]]
            html_file.make_table_of_images(contents=image_contents)
            images = [y for x in image_contents for y in x]
            html_file.make_modal_carousel(images=images)
            table_contents = []
            for j in contents:
                one_approx_content = [j[0]] + [j[k * len(self.result_files)
                                               + num + 1] for k in range(4)]
                table_contents.append(one_approx_content)
            html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                                 contents=table_contents, heading_span=1)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_1d_histogram_pages(self):
        """Make the 1d histogram pages.
        """
        pages = ["%s_%s_%s" % (i, i, j) for num, i in enumerate(self.labels)
                 for j in self.parameters[num]]
        pages += ["%s_%s_multiple" % (i, i) for i in self.labels]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.labels):
            for j in self.parameters[num]:
                html_file = self._setup_page(
                    "%s_%s" % (app, j), self.navbar_for_approximant_homepage[num],
                    app, title="%s Posterior PDF for %s" % (app, j),
                    approximant=app, background_colour=self.colors[num],
                    histogram_download=True)
                html_file.make_banner(approximant=app, key=app)
                path = self.image_path["other"]
                contents = [[path + "%s_1d_posterior_%s.png" % (app, j)],
                            [path + "%s_sample_evolution_%s.png" % (app, j),
                             path + "%s_autocorrelation_%s.png" % (app, j)]]
                html_file.make_table_of_images(
                    contents=contents, rows=1, columns=2, code="changeimage")
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()
            html_file = self._setup_page(
                "%s_multiple" % (app), self.navbar_for_approximant_homepage[num],
                app, title="%s Posteriors for multiple" % (app),
                approximant=app, background_colour=self.colors[num])
            ordered_parameters = self._categorize_parameters(self.parameters[num])
            ordered_parameters = [i for j in ordered_parameters for i in j[1]]
            html_file.make_search_bar(sidebar=[i for i in self.parameters[num]],
                                      popular_options=[
                                          "mass_1, mass_2",
                                          "luminosity_distance, iota, ra, dec",
                                          "iota, phi_12, phi_jl, tilt_1, tilt_2",
                                          {"all": ", ".join(ordered_parameters)}],
                                      label=self.labels[num],
                                      code="combines")
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_corner_pages(self):
        """Make the corner pages.
        """
        pages = ["%s_%s_corner" % (i, i) for i in self.labels]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.labels):
            html_file = self._setup_page(
                "%s_corner" % (app), self.navbar_for_approximant_homepage[num],
                app, title="%s Corner Plots" % (app),
                background_colour=self.colors[num], approximant=app)
            html_file.make_banner(approximant=app, key="corner")
            params = ["luminosity_distance", "dec", "a_2", "phase",
                      "a_1", "geocent_time", "phi_jl", "psi", "ra",
                      "mass_2", "mass_1", "phi_12", "tilt_2", "iota",
                      "tilt_1", "chi_p", "chirp_mass", "mass_ratio",
                      "symmetric_mass_ratio", "total_mass", "chi_eff",
                      "redshift", "mass_1_source", "mass_2_source",
                      "total_mass_source", "chirp_mass_source"]
            included_parameters = [i for i in self.parameters[num] if i in params]
            html_file.make_search_bar(sidebar=included_parameters,
                                      popular_options=[
                                          "mass_1, mass_2",
                                          "luminosity_distance, iota, ra, dec",
                                          "iota, phi_12, phi_jl, tilt_1, tilt_2"],
                                      label=self.labels[num])
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_comparison_pages(self):
        """Make the comparison pages.
        """
        webpage.make_html(web_dir=self.webdir, pages=["Comparison"])
        html_file = self._setup_page(
            "Comparison", self.navbar_for_comparison_homepage,
            title="Comparison Summary Page", approximant="Comparison")
        html_file.make_banner(approximant="Comparison", key="Comparison")
        path = self.image_path["other"]
        contents = [[path + "combined_skymap.png", path + "compare_waveforms.png"]]
        html_file.make_table_of_images(contents=contents)
        images = [y for x in contents for y in x]
        html_file.make_modal_carousel(images=images)
        if self.sensitivity:
            html_file.add_content(
                "<div class='row justify-content-center' "
                "style='margin=top: 2.0em;'><p>To see the sky sensitivity for "
                "the following networks, click the button</p></div>")
            html_file.add_content(
                "<div class='row justify-content-center' "
                "style='margin-top: 0.2em;'><button type='button' class='btn "
                "btn-info' onclick='%s.src=\"%s/plots/combined_skymap.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>Sky Map</button>"
                "<button type='button' class='btn btn-info' onclick='%s.src=\""
                "%s/plots/%s_sky_sensitivity_HL.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>HL</button>"
                % ("combined_skymap", self.baseurl, "combined_skymap", self.baseurl,
                   self.approximant[0]))
            html_file.add_content(
                "<button type='button' class='btn btn-info' "
                "onclick='%s.src=\"%s/plots/%s_sky_sensitivity_HLV.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>HLV</button></div>\n"
                % ("combined_skymap", self.baseurl, self.approximant[0]))
        try:
            statistics = self.comparison_statistics
        except Exception as e:
            statistics = None
            logger.info("Failed to generate comparison statistics because "
                        "%s" % (e))
        if statistics:
            rows = range(len(self.result_files))
            style_ks = "margin-top:5em; margin-bottom:1em; background-color:#FFFFFF; " + \
                "box-shadow: 0 0 5px grey;"
            style_js = "margin-top:0em; margin-bottom:5em; background-color:#FFFFFF; " + \
                "box-shadow: 0 0 5px grey;"
            table_contents = {self.same_parameters[num]: [
                [self.labels[i]] + statistics[num][0][i] for i in rows]
                for num in range(len(self.same_parameters))}
            html_file.make_table(headings=[" "] + self.labels,
                                 contents=table_contents, heading_span=1,
                                 accordian_header="KS test total", style=style_ks)
            table_contents = {self.same_parameters[num]: [
                [self.labels[i]] + statistics[num][1][i] for i in rows]
                for num in range(len(self.same_parameters))}
            html_file.make_table(headings=[" "] + self.labels,
                                 contents=table_contents, heading_span=1,
                                 accordian_header="JS divergence test total", style=style_js)
        html_file.make_footer(user=self.user, rundir=self.webdir)
        pages = ["Comparison_%s" % (i) for i in self.same_parameters]
        pages += ["Comparison_multiple"]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, i in enumerate(self.same_parameters):
            html_file = self._setup_page(
                "Comparison_%s" % (i),
                self.navbar_for_comparison_homepage,
                title="Comparison PDF for %s" % (i),
                approximant="Comparison")
            html_file.make_banner(approximant="Comparison", key="Comparison")
            html_file.insert_image(path + "combined_1d_posterior_%s.png" % (i),
                                   code="changeimage")
            if statistics:
                table_contents = [
                    [self.labels[i]] + statistics[num][0][i] for i in rows]
                html_file.make_table(headings=[" "] + self.labels,
                                     contents=table_contents, heading_span=1,
                                     accordian_header="KS test", style=style_ks)
                table_contents = [
                    [self.labels[i]] + statistics[num][1][i] for i in rows]
                html_file.make_table(headings=[" "] + self.labels,
                                     contents=table_contents, heading_span=1,
                                     accordian_header="JS divergence test",
                                     style=style_js)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        html_file = self._setup_page(
            "Comparison_multiple", self.navbar_for_comparison_homepage,
            approximant="Comparison", title="Comparison Posteriors for multiple")
        html_file.make_search_bar(sidebar=self.same_parameters,
                                  popular_options=[
                                      "mass_1, mass_2",
                                      "luminosity_distance, iota, ra, dec",
                                      "iota, phi_12, phi_jl, tilt_1, tilt_2",
                                      {"all": ", ".join(self.same_parameters)}],
                                  label="None", code="combines")
        html_file.make_footer(user=self.user, rundir=self.webdir)


def main():
    """Top level interface for `summarypages`
    """
    parser = command_line()
    insert_gwspecific_option_group(parser)
    opts = parser.parse_args()
    func = functions()
    args = func["input"](opts)
    func["PlotGeneration"](args)
    func["WebpageGeneration"](args)
    func["MetaFile"](args)
    func["FinishingTouches"](args)


if __name__ == "__main__":
    main()
