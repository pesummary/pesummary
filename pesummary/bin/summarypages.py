#! /usr/bin/env python

# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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

import logging

import subprocess
import os
import shutil

import numpy as np
import h5py

import pesummary
from pesummary.utils.utils import combine_hdf_files, logger
from pesummary.webpage import webpage
from pesummary.bin.inputs import command_line, Input, PostProcessing
from pesummary.bin.summaryplots import PlotGeneration

__doc__ == "Classes to generate webpages"


class WebpageGeneration(PostProcessing):
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
        self.image_path = "%s/plots/" %(self.baseurl)
        self.generate_webpages()
        logger.info("Finished generating webpages")

    @property
    def navbar_for_homepage(self):
        return self._navbar_for_homepage

    @navbar_for_homepage.setter
    def navbar_for_homepage(self, navbar_for_homepage):
        approximant_links = self._approximant_navbar_links()
        links = ["home", ["Approximants", approximant_links]]
        if len(self.approximant) > 1:
            links[1][1] = links[1][1]+["Comparison"]
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
        final_links = [["home", ["Approximants", self._approximant_navbar_links()],
            {"corner": self.labels[num]}, {"config": self.labels[num]}, j] \
            for num, j in enumerate(links)]
        if len(self.approximant) > 1:
            for i in final_links:
                i[1][1] = i[1][1]+["Comparison"]
        self._navbar_for_approximant_homepage = final_links

    @property
    def navbar_for_comparison_homepage(self):
        return self._navbar_for_comparison_homepage

    @navbar_for_comparison_homepage.setter
    def navbar_for_comparison_homepage(self, navbar_for_comparison_homepage):
        links = ["1d_histograms", ["multiple"]]
        for i in self._categorize_parameters(self.same_parameters):
            links.append(i)
        final_links = ["home", ["Approximants", self._approximant_navbar_links()], links]
        final_links[1][1] = final_links[1][1]+["Comparison"]
        self._navbar_for_comparison_homepage = final_links

    def _approximant_navbar_links(self):
        """Return the navbar structure for the approximant tab. If we have
        passed a coherence test, then we need to prepend the approximant by
        a label so we can determine which tab corresponds to which network.
        """
        if self.coherence_test:
            labels = [i[len(self.gracedb)+1:] if self.gracedb else i for i in \
                self.labels]
            links = [None]*len(self.approximant)
            duplicates=dict(set((x,self.approximant.count(x)) for x in \
                filter(lambda rec : self.approximant.count(rec)>1,self.approximant)))
            if len(duplicates.keys()) > 0:
                for num, i in enumerate(self.approximant):
                    if i in duplicates.keys():
                        links[num]  = "_".join([labels[num], i])
            return [{i:j} if k == None else {k: "None"} for i,j,k in \
                zip(self.approximant, self.labels, links)]
        return [{i:j} for i, j in zip(self.approximant, self.labels)]

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
        if any("spin" in j for j in parameters):
            cond = self._condition(["spin", "chi_p" "chi_eff", "a_1", "a_2"],
                [])
            params.append(["spins", self._partition(cond, parameters)])
        if any("phi" in j for j in parameters):
            cond = self._condition(["phi", "tilt"], [])
            params.append(["spin_angles", self._partition(cond, parameters)])
        if any("ra" in j for j in parameters):
            cond = self._condition(["ra", "dec", "psi"], ["mass_ratio"])
            params.append(["sky_location", self._partition(cond, parameters)])
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

    def generate_webpages(self):
        """Generate all webpages that we need.
        """
        self.make_home_pages()
        self.make_1d_histogram_pages()
        self.make_corner_pages()
        self.make_config_pages()
        if len(self.approximant) > 1:
            self.make_comparison_pages()

    def _setup_page(self, html_page, links, label=None, title=None,
            approximant=None, background_colour=None):
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
        """
        html_file = webpage.open_html(web_dir=self.webdir,
            base_url=self.baseurl, html_page=html_page, label=label)
        if title:
            html_file.make_header(title=title, approximant=approximant,
                background_colour=background_colour)
        else:
            html_file.make_header(approximant=approximant,
                background_colour=background_colour) 
        html_file.make_navbar(links=links)
        return html_file

    def make_home_pages(self):
        """Make the home pages.
        """
        pages = ["%s_%s" %(self.labels[num], i) for num, i in \
            enumerate(self.approximant)]
        pages.append("home")
        webpage.make_html(web_dir=self.webdir, pages=pages)
        if self.gracedb:
            html_file = self._setup_page("home", self.navbar_for_homepage,
                title="Parameter Estimation Summary Pages for %s" %(self.gracedb))
        else:
            html_file = self._setup_page("home", self.navbar_for_homepage)
        key_data = self._key_data()
        table_data = [{j:i[j] for j in self.same_parameters} for i in key_data]
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
        html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                             contents=contents, heading_span=len(self.samples),
                             colors=self.colors[:len(self.samples)])
        html_file.close()
        for num, i in enumerate(self.approximant):
            html_file = self._setup_page(i, self.navbar_for_approximant_homepage[num],
                self.labels[num], title="%s Summary page" %(i),
                background_colour=self.colors[num], approximant=i)
            path = self.image_path
            label = self.labels[num]
            image_contents = [[path+"%s_1d_posterior_%s_mass_1.png" %(label, i),
                               path+"%s_1d_posterior_%s_mass_2.png" %(label, i),
                               path+"%s_1d_posterior_%s_luminosity_distance.png" %(label, i)],
                              [path+"%s_%s_skymap.png" %(label, i),
                               path+"%s_%s_waveform.png" %(label, i)],
                              [path+"%s_1d_posterior_%s_iota.png" %(label, i),
                               path+"%s_1d_posterior_%s_a_1.png" %(label, i),
                               path+"%s_1d_posterior_%s_a_2.png" %(label, i)]]
            html_file.make_table_of_images(contents=image_contents)
            images = [y for x in image_contents for y in x]
            html_file.make_modal_carousel(images=images)
            table_contents = []
            for i in contents:
                one_approx_content = [i[0]]+[i[j*len(self.approximant)+num+1] \
                    for j in range(4)]
                table_contents.append(one_approx_content) 
            html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                                 contents=table_contents, heading_span=1)
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_1d_histogram_pages(self):
        """Make the 1d histogram pages.
        """
        pages = ["%s_%s_%s" %(self.labels[num], i, j) for num, i in \
            enumerate(self.approximant) for j in self.parameters[num]]
        pages += ["%s_%s_multiple" %(self.labels[num], i) for num, i in \
            enumerate(self.approximant)]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.approximant):
            for j in self.parameters[num]:
                html_file = self._setup_page("%s_%s" %(app, j),
                    self.navbar_for_approximant_homepage[num],
                    self.labels[num], title="%s Posterior PDF for %s" %(app, j),
                    approximant=app, background_colour=self.colors[num])
                path = self.image_path
                label = self.labels[num]
                contents = [[path+"%s_1d_posterior_%s_%s.png" %(label, app, j)],
                            [path+"%s_sample_evolution_%s_%s.png" %(label, app, j),
                             path+"%s_autocorrelation_%s_%s.png" %(label, app, j)]] 
                html_file.make_table_of_images(contents=contents, rows=1, columns=2)
                html_file.make_footer(user=self.user, rundir=self.webdir)
                html_file.close()
            html_file = self._setup_page("%s_multiple" %(app),
                self.navbar_for_approximant_homepage[num],
                self.labels[num], title="%s Posteriors for multiple" %(app),
                approximant=app, background_colour=self.colors[num])
            ordered_parameters = self._categorize_parameters(self.parameters[num])
            ordered_parameters = [i for j in ordered_parameters for i in j[1]]
            html_file.make_search_bar(sidebar=[i for i in self.parameters[num]],
                                      popular_options=["mass_1, mass_2",
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
        pages = ["%s_%s_corner" %(self.labels[num], i) for num, i in \
            enumerate(self.approximant)]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for num, app in enumerate(self.approximant):
            html_file = self._setup_page("%s_corner" %(app),
                self.navbar_for_approximant_homepage[num],
                self.labels[num], title="%s Corner Plots" %(app),
                background_colour=self.colors[num], approximant=app)
            params = ["luminosity_distance", "dec", "a_2", "phase",
                      "a_1", "geocent_time", "phi_jl", "psi", "ra",
                      "mass_2", "mass_1", "phi_12", "tilt_2", "iota",
                      "tilt_1", "chi_p", "chirp_mass", "mass_ratio",
                      "symmetric_mass_ratio", "total_mass", "chi_eff",
                      "redshift", "mass_1_source", "mass_2_source",
                      "total_mass_source", "chirp_mass_source"]
            included_parameters = [i for i in self.parameters[num] if i in params]
            html_file.make_search_bar(sidebar=included_parameters,
                                      popular_options=["mass_1, mass_2",
                                          "luminosity_distance, iota, ra, dec",
                                          "iota, phi_12, phi_jl, tilt_1, tilt_2"],
                                      label=self.labels[num])
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_config_pages(self):
        """Make the configuration pages.
        """
        pages = ["%s_%s_config" %(self.labels[num], i) for num, i in \
            enumerate(self.approximant)]
        webpage.make_html(web_dir=self.webdir, pages=pages, stylesheets=pages)
        if not self.config:
            configs = [None]*len(self.approximant)
        for num, app in enumerate(self.approximant):
            html_file = self._setup_page("%s_config" %(app),
                self.navbar_for_approximant_homepage[num],
                self.labels[num], title="%s configuration" %(app),
                background_colour=self.colors[num], approximant=app)
            if self.config:
                with open(self.config[num], 'r') as f:
                    contents = f.read()
                styles = html_file.make_code_block(language='ini', contents=contents)
                with open('{0:s}/css/{1:s}_config.css'.format(self.webdir, app), 'w') as f:
                    f.write(styles)
            else:
                html_file.add_content("<div class='row justify-content-center'>"
                    "<p style='margin-top:2.5em'> No configuration file was "
                    "provided </p></div>")
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_comparison_pages(self):
        """Make the comparison pages.
        """
        webpage.make_html(web_dir=self.webdir, pages=["Comparison"])
        html_file = self._setup_page("Comparison",
            self.navbar_for_comparison_homepage,
            title="Comparison Summary Page",
            approximant="Comparison")
        path = self.image_path
        contents = [[path+"combined_skymap.png", path+"compare_waveforms.png"]]
        html_file.make_table_of_images(contents=contents)
        if self.sensitivity:
            html_file.add_content("<div class='row justify-content-center' "
                "style='margin=top: 2.0em;'><p>To see the sky sensitivity for "
                "the following networks, click the button</p></div>")
            html_file.add_content("<div class='row justify-content-center' "
                "style='margin-top: 0.2em;'><button type='button' class='btn "
                "btn-info' onclick='%s.src=\"%s/plots/combined_skymap.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>Sky Map</button>"
                "<button type='button' class='btn btn-info' onclick='%s.src=\""
                "%s/plots/%s_sky_sensitivity_HL.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>HL</button>"
                %("combined_skymap", self.baseurl, "combined_skymap", self.baseurl,
                  self.approximant[0]))
            html_file.add_content("<button type='button' class='btn btn-info' "
                "onclick='%s.src=\"%s/plots/%s_sky_sensitivity_HLV.png\"'"
                "style='margin-left:0.25em; margin-right:0.25em'>HLV</button></div>\n"
                %("combined_skymap", self.baseurl, self.approximant[0]))
        html_file.make_footer(user=self.user, rundir=self.webdir)
        pages = ["Comparison_%s" %(i) for i in self.same_parameters]
        pages += ["Comparison_multiple"]
        webpage.make_html(web_dir=self.webdir, pages=pages)
        for i in self.same_parameters:
            html_file=self._setup_page("Comparison_%s" %(i),
                self.navbar_for_comparison_homepage,
                title="Comparison PDF for %s" %(i),
                approximant="Comparison")
            html_file.insert_image(path+"combined_posterior_%s.png" %(i))
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()
        html_file = self._setup_page("Comparison_multiple",
            self.navbar_for_comparison_homepage, approximant="Comparison",
            title="Comparison Posteriors for multiple")
        html_file.make_search_bar(sidebar=self.same_parameters,
                                  popular_options=["mass_1, mass_2",
                                      "luminosity_distance, iota, ra, dec",
                                      "iota, phi_12, phi_jl, tilt_1, tilt_2",
                                      {"all": ", ".join(self.same_parameters)}],
                                  label="None", code="combines")
        html_file.make_footer(user=self.user, rundir=self.webdir)


class FinishingTouches(PostProcessing):
    """Class to handle the finishing touches

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    meta_file: str
        String containing the path to the meta file containing all information
    """
    def __init__(self, inputs):
        super(FinishingTouches, self).__init__(inputs)
        self.meta_file = self.webdir+"/samples/posterior_samples.h5"
        logger.info("Combining results files into a single meta file")
        self.generate_posterior_samples_meta_file()
        logger.info("Combined results files can be found at %s" %(
            self.webdir+"/samples/posterior_samples.h5"))
        self.send_email()
        self.tidy_up()
        logger.info("Complete. Webpages can be viewed at the following url "
            "%s" %(self.baseurl+"/home.html"))

    def generate_posterior_samples_meta_file(self):
        """Combine the given results files into a single meta file.
        """
        if self.meta_file not in self.result_files:
            for num, i in enumerate(self.result_files):
                if num == 0:
                    shutil.copyfile(i, self.meta_file)
                else:
                    combine_hdf_files(self.meta_file, i)
        else:
            for num, i in enumerate(self.result_files):
                if "posterior_samples" not in i:
                    combine_hdf_files(self.meta_file, i)

    def send_email(self, message=None):
        """Send notification email.
        """
        if self.email:
            logger.info("Sending email to %s" %(self.email))
            try:
                self._email_notify(message)
            except Exception as e:
                logger.info("Unable to send notification email because %s" %(e))

    def _email_message(self, message=None):
        """Message that will be send in the email.
        """
        if not message:
            message=("Hi %s,\n\nYour output page is ready on %s. You can "
                     "view the result at %s"
                     "\n" %(self.user, self.host, self.baseurl+"/home.html"))
        return message

    def _email_notify(self, message):
        """Subprocess to send the notification email.
        """
        from_address = "%s@%s" %(self.user, self.host)
        subject = "Output page available at %s" %(self.host)
        message = self._email_message(message)
        cmd = 'echo -e "%s" | mail -s "%s" "%s"' %(message, subject, address)
        ess = subprocess.Popen(cmd, shell=True)
        ess.wait()

    def tidy_up(self):
        """Remove all unnecessary files.
        """
        for i in self.result_files:
            if "posterior_samples" not in i:
                os.remove(i) 


if __name__ == '__main__':
    parser = command_line()
    opts = parser.parse_args()
    inputs = Input(opts)
    PlotGeneration(inputs)
    WebpageGeneration(inputs)
    FinishingTouches(inputs)
