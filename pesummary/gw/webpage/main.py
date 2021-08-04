# Licensed under an MIT style license -- see LICENSE.md

import os
import numpy as np

import pesummary
from pesummary.core.webpage import webpage
from pesummary.core.webpage.main import _WebpageGeneration as _CoreWebpageGeneration
from pesummary.core.webpage.main import PlotCaption
from pesummary.gw.file.standard_names import descriptive_names
from pesummary.utils.utils import logger, safe_round
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class CommandLineCaption(object):
    """Class to handle generating the command line used to generate a plot and
    a caption to describe the plot

    Parameters
    ----------
    """
    def __init__(self, command_line_arguments, samples=None):
        self.args = command_line_arguments
        self.samples = samples
        self.executable = self.get_executable("summaryplots")

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

    @property
    def command_line(self):
        """Generate the command line used to generate the plot
        """
        return "{} {}".format(self.executable, " ".join(self.args))

    @property
    def caption(self):
        """Make a caption to describe the plot
        """
        general_cli = ""
        if "1d_histogram" in self.args[0]:
            return self.histogram_caption()
        if "skymap" in self.args[0]:
            return self.skymap_caption()
        return general_cli

    def histogram_caption(self):
        """Return a caption to describe the 1d histogram plot
        """
        args = self.args[0].split(" ")
        parameter = args[args.index("--parameter") + 1]
        general_cli = (
            "1d histogram showing the posterior distribution for "
            "{}.".format(parameter)
        )
        if parameter == "chi_p":
            general_cli += (
                " chi_p is the precession parameter and quantifies how much "
                "precession there is in the system. It ranges from 0 to 1 "
                "with 0 meaning there is no precession in the system and 1 "
                "meaning there is maximal precession in the system."
            )
        if self.samples is not None:
            general_cli += (
                " The median of the distribution is {} with 90% confidence "
                "interval {}".format(
                    np.round(self.samples.average(type="median"), 3),
                    [np.round(i, 3) for i in self.samples.confidence_interval()]
                )
            )
        return general_cli

    def skymap_caption(self):
        """Return a caption to describe the skymap plot
        """
        general_cli = (
            "Skymap showing the possible location of the source of the "
            "source of the gravitational waves"
        )


class _WebpageGeneration(_CoreWebpageGeneration):
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
        mcmc_samples=False, external_hdf5_links=False, preliminary_pages=False,
        existing_plot=None, disable_expert=False, analytic_priors=None
    ):
        self.pepredicates_probs = pepredicates_probs
        self.pastro_probs = pastro_probs
        self.gracedb = gracedb
        self.approximant = approximant
        self.file_kwargs = file_kwargs
        self.publication = publication
        self.publication_kwargs = publication_kwargs
        self.result_files = result_files
        self.gwdata = gwdata
        self.no_ligo_skymap = no_ligo_skymap
        self.psd = psd
        self.priors = priors

        super(_WebpageGeneration, self).__init__(
            webdir=webdir, samples=samples, labels=labels,
            publication=publication, user=user, config=config,
            same_parameters=same_parameters, base_url=base_url,
            file_versions=file_versions, hdf5=hdf5, colors=colors,
            custom_plotting=custom_plotting,
            existing_labels=existing_labels, existing_config=existing_config,
            existing_file_version=existing_file_version,
            existing_injection_data=existing_injection_data,
            existing_samples=existing_samples,
            existing_metafile=existing_metafile,
            existing_file_kwargs=existing_file_kwargs,
            existing_weights=existing_weights,
            add_to_existing=add_to_existing, notes=notes,
            disable_comparison=disable_comparison,
            disable_interactive=disable_interactive,
            package_information=package_information, mcmc_samples=mcmc_samples,
            external_hdf5_links=external_hdf5_links, key_data=key_data,
            existing_plot=existing_plot, disable_expert=disable_expert,
            analytic_priors=analytic_priors
        )
        if self.file_kwargs is None:
            self.file_kwargs = {
                label: {"sampler": {}, "meta_data": {}} for label in self.labels
            }
        if self.approximant is None:
            self.approximant = {label: None for label in self.labels}
        if self.result_files is None:
            self.result_files = [None] * len(self.labels)
        self.psd_path = {"other": os.path.join("..", "psds")}
        self.calibration_path = {"other": os.path.join("..", "calibration")}
        self.preliminary_pages = preliminary_pages
        if not isinstance(self.preliminary_pages, dict):
            if self.preliminary_pages:
                self.preliminary_pages = {
                    label: True for label in self.labels
                }
            else:
                self.preliminary_pages = {
                    label: False for label in self.labels
                }
        if all(value for value in self.preliminary_pages.values()):
            self.all_pages_preliminary = True
        if len(self.labels) > 1:
            if any(value for value in self.preliminary_pages.values()):
                self.preliminary_pages["Comparison"] = True

    def categorize_parameters(self, parameters):
        """Categorize the parameters into common headings

        Parameters
        ----------
        parameters: list
            list of parameters that you would like to sort
        """
        return super(_WebpageGeneration, self).categorize_parameters(
            parameters, starting_letter=False
        )

    def _jensen_shannon_divergence(self, param, samples):
        """Return the Jensen Shannon divergence between two sets of samples

        Parameters
        ----------
        param: str
            The parameter that the samples belong to
        samples: list
            2d list containing the samples you wish to calculate the Jensen
            Shannon divergence between
        """
        from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde
        from pesummary.gw.plots.plot import _return_bounds

        xlow, xhigh = _return_bounds(param, samples, comparison=True)
        return super(_WebpageGeneration, self)._jensen_shannon_divergence(
            param, samples, kde=Bounded_1d_kde, xlow=xlow, xhigh=xhigh
        )

    def make_navbar_for_homepage(self):
        """Make a navbar for the homepage
        """
        links = super(_WebpageGeneration, self).make_navbar_for_homepage()
        if self.gwdata is not None:
            links.append(["Detchar", [i for i in self.gwdata.keys()]])
        return links

    def make_navbar_for_result_page(self):
        """Make a navbar for the result page homepage
        """
        links = super(_WebpageGeneration, self).make_navbar_for_result_page()
        for num, label in enumerate(self.labels):
            if self.pepredicates_probs[label] is not None:
                links[label].append({"Classification": label})
        return links

    def generate_webpages(self):
        """Generate all webpages for all result files passed
        """
        super(_WebpageGeneration, self).generate_webpages()
        if self.publication:
            self.make_publication_pages()
        if self.gwdata is not None:
            self.make_detector_pages()
        if all(val is not None for key, val in self.pepredicates_probs.items()):
            self.make_classification_pages()

    def _make_home_pages(self, pages):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        title = None if self.gracedb is None else (
            "Parameter Estimation Summary Pages for {}".format(self.gracedb)
        )
        banner = "Summary" if self.gracedb is None else (
            "Summary for {}".format(self.gracedb)
        )
        html_file = self.setup_page("home", self.navbar["home"], title=title)
        html_file.make_banner(approximant=banner, key="Summary")
        path = self.image_path["home"]

        if self.make_comparison:
            if not all(self.approximant[i] is not None for i in self.labels):
                image_contents = []
                _plot = os.path.join(path, "compare_time_domain_waveforms.png")
                if os.path.isfile(_plot):
                    image_contents.append(_plot)
                    image_contents = [image_contents]
                    html_file = self.make_modal_carousel(
                        html_file, image_contents, unique_id=True
                    )

        for i in self.labels:
            html_file.make_banner(approximant=i, key=i)
            image_contents, captions = [], []
            basic_string = os.path.join(self.webdir, "plots", "{}.png")
            relative_path = os.path.join(path, "{}.png")
            if os.path.isfile(basic_string.format("%s_strain" % (i))):
                image_contents.append(relative_path.format("%s_strain" % (i)))
                captions.append(PlotCaption("strain"))
            if os.path.isfile(basic_string.format("%s_psd_plot" % (i))):
                image_contents.append(relative_path.format("%s_psd_plot" % (i)))
                captions.append(PlotCaption("psd"))
            if os.path.isfile(
                basic_string.format("%s_waveform_time_domain" % (i))
            ):
                image_contents.append(
                    relative_path.format("%s_waveform_time_domain" % (i))
                )
                captions.append(PlotCaption("time_waveform"))
            if os.path.isfile(
                basic_string.format("%s_calibration_plot" % (i))
            ):
                image_contents.append(
                    relative_path.format("%s_calibration_plot" % (i))
                )
                captions.append(PlotCaption("calibration"))
            image_contents = [image_contents]
            html_file = self.make_modal_carousel(
                html_file, image_contents, unique_id=True, extra_div=True,
                captions=[captions]
            )

        for _key in ["sampler", "meta_data"]:
            if _key == "sampler":
                html_file.make_banner(
                    approximant="Sampler kwargs", key="sampler_kwargs",
                    _style="font-size: 26px;"
                )
            else:
                html_file.make_banner(
                    approximant="Meta data", key="meta_data",
                    _style="font-size: 26px;"
                )
            _style = "margin-top:3em; margin-bottom:5em; max-width:1400px"
            _class = "row justify-content-center"
            html_file.make_container(style=_style)
            html_file.make_div(4, _class=_class, _style=None)

            base_label = self.labels[0]
            total_keys = list(self.file_kwargs[base_label][_key].keys())
            if len(self.labels) > 1:
                for _label in self.labels[1:]:
                    total_keys += [
                        key for key in self.file_kwargs[_label][_key].keys()
                        if key not in total_keys
                    ]
            _headings = ["label"] + total_keys
            table_contents = [
                [i] + [
                    self.file_kwargs[i][_key][key] if key in
                    self.file_kwargs[i][_key].keys() else "-" for key in
                    total_keys
                ] for i in self.labels
            ]
            html_file.make_table(
                headings=_headings, format="table-hover", heading_span=1,
                contents=table_contents, accordian=False
            )
            html_file.end_div(4)
            html_file.end_container()
            html_file.export("{}.csv".format(_key))

        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()
        super(_WebpageGeneration, self)._make_home_pages(pages, make_home=False)

    def make_publication_pages(self):
        """Wrapper function for _make_publication_pages()
        """
        pages = ["Publication"]
        self.create_blank_html_pages(pages)
        self._make_publication_pages(pages)

    def _make_publication_pages(self, pages):
        """Make the publication pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        from glob import glob

        executable = self.get_executable("summarypublication")
        general_cli = "%s --webdir %s --samples %s --labels %s --plot {}" % (
            executable, os.path.join(self.webdir, "plots", "publication"),
            " ".join(self.result_files), " ".join(self.labels)
        )
        if self.publication_kwargs != {}:
            general_cli += "--publication_kwargs %s" % (
                " ".join(
                    [
                        "{}:{}".format(key, value) for key, value in
                        self.publication_kwargs.items()
                    ]
                )
            )
        html_file = self.setup_page(
            "Publication", self.navbar["home"], title="Publication Plots"
        )
        html_file.make_banner(approximant="Publication", key="Publication")
        path = self.image_path["other"]
        pub_plots = glob(
            os.path.join(self.webdir, "plots", "publication", "*.png")
        )
        for num, i in enumerate(pub_plots):
            shortened_path = i.split("/plots/")[-1]
            pub_plots[num] = path + shortened_path
        cli = []
        cap = []
        posterior_name = \
            lambda i: "{} ({})".format(i, descriptive_names[i]) if i in \
            descriptive_names.keys() and descriptive_names[i] != "" else i
        for i in pub_plots:
            filename = i.split("/")[-1]
            if "violin_plot" in filename:
                parameter = filename.split("violin_plot_")[-1].split(".png")[0]
                cli.append(
                    general_cli.format("violin") + " --parameters %s" % (
                        parameter
                    )
                )
                cap.append(
                    PlotCaption("violin").format(posterior_name(parameter))
                )
            elif "spin_disk" in filename:
                cli.append(general_cli.format("spin_disk"))
                cap.append(PlotCaption("spin_disk"))
            elif "2d_contour" in filename:
                parameters = filename.split("2d_contour_plot_")[-1].split(".png")[0]
                cli.append(
                    general_cli.format("2d_contour") + " --parameters %s" % (
                        parameters.replace("_and_", " ")
                    )
                )
                pp = parameters.split("_and_")
                cap.append(
                    PlotCaption("2d_contour").format(
                        posterior_name(pp[0]), posterior_name(pp[1])
                    )
                )
        image_contents = [
            pub_plots[i:3 + i] for i in range(0, len(pub_plots), 3)
        ]
        command_lines = [
            cli[i:3 + i] for i in range(0, len(cli), 3)
        ]
        captions = [cap[i:3 + i] for i in range(0, len(cap), 3)]
        html_file = self.make_modal_carousel(
            html_file, image_contents, cli=command_lines, captions=captions
        )
        html_file.make_footer(user=self.user, rundir=self.webdir)
        html_file.close()

    def make_detector_pages(self):
        """Wrapper function for _make_publication_pages()
        """
        pages = [i for i in self.gwdata.keys()]
        self.create_blank_html_pages(pages)
        self._make_detector_pages(pages)

    def _make_detector_pages(self, pages):
        """Make the detector characterisation pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        from glob import glob
        from pesummary.utils.utils import (
            determine_gps_time_and_window, command_line_dict
        )
        from astropy.time import Time

        executable = self.get_executable("summarydetchar")
        try:
            command_line = command_line_dict()
        except SystemExit:
            command_line = {"gwdata": {}}
        if isinstance(command_line["gwdata"], dict):
            gwdata_command_line = [
                "{}:{}".format(key, val) for key, val in
                command_line["gwdata"].items()
            ]
        else:
            gwdata_command_line = command_line["gwdata"]
            if gwdata_command_line is None:
                gwdata_command_line = []
        general_cli = "%s --webdir %s --gwdata %s --plot {}{}" % (
            executable, os.path.join(self.webdir, "plots"),
            " ".join(gwdata_command_line)
        )
        path = self.image_path["other"]
        base = os.path.join(path, "{}_{}.png")
        ADD_DETCHAR_LINK = True
        try:
            maxL_samples = {
                i: {
                    "geocent_time": self.key_data[i]["geocent_time"]["maxL"]
                } for i in self.labels
            }
        except KeyError:
            # trying a different name for time
            try:
                maxL_samples = {
                    i: {
                        "geocent_time": self.key_data[i][
                            "marginalized_geocent_time"
                        ]["maxL"]
                    } for i in self.labels
                }
            except KeyError:
                logger.warn(
                    "Failed to find a time parameter to link to detchar/"
                    "summary pages. Not adding link to webpages."
                )
                ADD_DETCHAR_LINK = False
        if ADD_DETCHAR_LINK:
            gps_time, window = determine_gps_time_and_window(maxL_samples, self.labels)
            t = Time(gps_time, format='gps')
            t = Time(t, format='datetime')
            link = (
                "https://ldas-jobs.ligo-wa.caltech.edu/~detchar/summary/day"
                "/{}{}{}/".format(
                    t.value.year,
                    "0{}".format(t.value.month) if t.value.month < 10 else t.value.month,
                    "0{}".format(t.value.day) if t.value.day < 10 else t.value.day
                )
            )
        else:
            link = None
            gps_time, window = None, None
        for det in self.gwdata.keys():
            html_file = self.setup_page(
                det, self.navbar["home"], title="{} Detchar".format(det)
            )
            html_file.make_banner(approximant=det, key="detchar", link=link)
            image_contents = [
                [base.format("spectrogram", det), base.format("omegascan", det)]
            ]
            command_lines = [
                [
                    general_cli.format("spectrogram", ""),
                    general_cli.format(
                        "omegascan", "--gps %s --vmin 0 --vmax 25 --window %s" % (
                            gps_time, window
                        )
                    )
                ]
            ]
            html_file = self.make_modal_carousel(
                html_file, image_contents, cli=command_lines, autoscale=True,
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
            html_file.close()

    def make_classification_pages(self):
        """Wrapper function for _make_publication_pages()
        """
        pages = ["{}_{}_Classification".format(i, i) for i in self.labels]
        self.create_blank_html_pages(pages)
        self._make_classification_pages(pages)

    def _make_classification_pages(self, pages):
        """Make the classification pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        executable = self.get_executable("summaryclassification")
        general_cli = "%s --samples {}" % (executable)
        for num, label in enumerate(self.labels):
            html_file = self.setup_page(
                "{}_Classification".format(label),
                self.navbar["result_page"][label], label,
                title="{} Classification".format(label),
                background_colour=self.colors[num], approximant=label
            )
            html_file.make_banner(approximant=label, key="classification")

            if self.pepredicates_probs[label] is not None:
                html_file.make_container()
                _class = "row justify-content-center"
                html_file.make_div(4, _class=_class, _style=None)
                keys = list(self.pepredicates_probs[label]["default"].keys())
                table_contents = [
                    ["{} prior".format(i)] + [
                        self.pepredicates_probs[label][i][j] for j in keys
                    ] for i in ["default", "population"]
                ]
                if self.pastro_probs[label] is not None:
                    keys += ["HasNS"]
                    keys += ["HasRemnant"]
                    table_contents[0].append(self.pastro_probs[label]["default"]["HasNS"])
                    table_contents[0].append(
                        self.pastro_probs[label]["default"]["HasRemnant"]
                    )
                    try:
                        table_contents[1].append(
                            self.pastro_probs[label]["population"]["HasNS"]
                        )
                        table_contents[1].append(
                            self.pastro_probs[label]["population"]["HasRemnant"]
                        )
                    except KeyError:
                        table_contents[1].append("-")
                        table_contents[1].append("-")
                        logger.warning(
                            "Failed to add 'em_bright' probabilities for population "
                            "reweighted prior"
                        )
                html_file.make_table(
                    headings=[" "] + keys, contents=table_contents,
                    heading_span=1, accordian=False
                )
                html_file.make_cli_button(
                    general_cli.format(self.result_files[num])
                )
                html_file.export(
                    "classification_{}.csv".format(label),
                    margin_top="-1.5em", margin_bottom="0.5em", json=True
                )
                html_file.end_div(4)
                html_file.end_container()
            path = self.image_path["other"]
            base = os.path.join(path, "%s_{}_pepredicates{}.png" % (label))
            image_contents = [
                [
                    base.format("default", ""), base.format("default", "_bar"),
                    base.format("population", ""),
                    base.format("population", "_bar")
                ]
            ]
            base = (
                "%s --webdir %s --labels %s --plot {} --prior {}" % (
                    general_cli.format(self.result_files[num]),
                    os.path.join(self.webdir, "plots"), label
                )
            )
            command_lines = [
                [
                    base.format("mass_1_mass_2", "default"),
                    base.format("bar", "default"),
                    base.format("mass_1_mass_2", "population"),
                    base.format("bar", "population")
                ]
            ]
            captions = [
                [
                    PlotCaption("default_classification_mass_1_mass_2"),
                    PlotCaption("default_classification_bar"),
                    PlotCaption("population_classification_mass_1_mass_2"),
                    PlotCaption("population_classification_bar")
                ]
            ]
            html_file = self.make_modal_carousel(
                html_file, image_contents, cli=command_lines, autoscale=True,
                captions=captions
            )
            html_file.make_footer(user=self.user, rundir=self.webdir)
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
        table = super(_WebpageGeneration, self)._make_entry_in_downloads_table(
            html_file, label, num, base_string
        )
        if not self.no_ligo_skymap:
            table.append(
                [
                    base_string.format(
                        "Fits file containing skymap for this analysis",
                        self.results_path["other"] + "%s_skymap.fits" % (label)
                    )
                ]
            )
        if self.psd is not None and self.psd != {} and label in self.psd.keys():
            for ifo in self.psd[label].keys():
                if len(self.psd[label][ifo]):
                    table.append(
                        [
                            base_string.format(
                                "%s psd file used for this analysis" % (ifo),
                                os.path.join(
                                    self.psd_path["other"],
                                    "%s_%s_psd.dat" % (label, ifo)
                                )
                            )
                        ]
                    )
        if self.priors is not None and "calibration" in self.priors.keys():
            if label in self.priors["calibration"].keys():
                for ifo in self.priors["calibration"][label].keys():
                    if len(self.priors["calibration"][label][ifo]):
                        table.append(
                            [
                                base_string.format(
                                    "%s calibration envelope file used for "
                                    "this analysis" % (ifo), os.path.join(
                                        self.calibration_path["other"],
                                        "%s_%s_cal.txt" % (label, ifo)
                                    )
                                )
                            ]
                        )
        return table

    def default_images_for_result_page(self, label):
        """Return the default images that will be displayed on the result page
        """
        path = self.image_path["other"]
        base_string = path + "%s_{}.png" % (label)
        image_contents = [
            [
                base_string.format("1d_posterior_mass_1"),
                base_string.format("1d_posterior_mass_2"),
            ], [
                base_string.format("1d_posterior_a_1"),
                base_string.format("1d_posterior_a_2"),
                base_string.format("1d_posterior_chi_eff")
            ], [
                base_string.format("1d_posterior_iota"),
                base_string.format("skymap"),
                base_string.format("waveform"),
                base_string.format("1d_posterior_luminosity_distance"),
            ]
        ]
        executable = self.get_executable("summaryplots")
        general_cli = (
            "%s --webdir %s --samples %s --burnin %s --plot {} {} "
            "--labels %s" % (
                executable, os.path.join(self.webdir, "plots"),
                self.result_files[self.labels.index(label)], conf.burnin, label
            )
        )
        cli = [
            [
                general_cli.format("1d_histogram", "--parameter mass_1"),
                general_cli.format("1d_histgram", "--parameter mass_2"),
            ], [
                general_cli.format("1d_histogram", "--parameter a_1"),
                general_cli.format("1d_histogram", "--parameter a_2"),
                general_cli.format("1d_histogram", "--parameter chi_eff")
            ], [
                general_cli.format("1d_histogram", "--parameter iota"),
                general_cli.format("skymap", ""),
                general_cli.format("waveform", ""),
                general_cli.format(
                    "1d_histogram", "--parameter luminosity_distance"
                ),
            ]
        ]

        caption_1d_histogram = PlotCaption("1d_histogram")
        posterior_name = \
            lambda i: "{} ({})".format(i, descriptive_names[i]) if i in \
            descriptive_names.keys() and descriptive_names[i] != "" else i
        captions = [
            [
                caption_1d_histogram.format(posterior_name("mass_1")),
                caption_1d_histogram.format(posterior_name("mass_2")),
            ], [
                caption_1d_histogram.format(posterior_name("a_1")),
                caption_1d_histogram.format(posterior_name("a_2")),
                caption_1d_histogram.format(posterior_name("chi_eff"))
            ], [
                caption_1d_histogram.format(posterior_name("iota")),
                PlotCaption("skymap"), PlotCaption("frequency_waveform"),
                caption_1d_histogram.format(posterior_name("luminosity_distance"))
            ]
        ]
        return image_contents, cli, captions

    def default_categories(self):
        """Return the default categories
        """
        categories = self.categories = {
            "masses": {
                "accept": ["mass"],
                "reject": ["source", "final", "torus"]
            },
            "source": {
                "accept": ["source"], "reject": ["final", "torus"]
            },
            "remnant": {
                "accept": ["final", "torus"], "reject": []
            },
            "inclination": {
                "accept": ["theta", "iota", "viewing"], "reject": []
            },
            "spins": {
                "accept": ["spin", "chi_p", "chi_eff", "a_1", "a_2", "precession"],
                "reject": ["lambda", "final", "gamma", "order"]
            },
            "spin_angles": {
                "accept": ["phi", "tilt", "beta"], "reject": []
            },
            "tidal": {
                "accept": [
                    "lambda", "gamma_", "log_pressure",
                    "spectral_decomposition_gamma_", "compactness_",
                    "tidal_disruption"
                ],
                "reject": []
            },
            "location": {
                "accept": [
                    "ra", "dec", "psi", "luminosity_distance", "redshift",
                    "comoving_distance"
                ],
                "reject": ["mass_ratio", "radiated", "ram", "ran", "rat"]
            },
            "timings": {
                "accept": ["time"], "reject": []
            },
            "SNR": {
                "accept": ["snr"], "reject": []
            },
            "calibration": {
                "accept": ["spcal", "recalib", "frequency"],
                "reject": ["minimum", "tidal_disruption", "quasinormal"]
            },
            "energy": {
                "accept": ["peak_luminosity", "radiated"],
                "reject": []
            },
            "others": {
                "accept": ["phase", "likelihood", "prior", "quasinormal"],
                "reject": ["spcal", "recalib"]
            }
        }
        return categories

    def default_popular_options(self):
        """Return a list of popular options
        """
        popular_options = [
            "mass_1, mass_2", "luminosity_distance, iota, ra, dec",
            "iota, phi_12, phi_jl, tilt_1, tilt_2"
        ]
        return popular_options

    def default_comparison_homepage_plots(self):
        """Return a list of default plots for the comparison homepage
        """
        path = self.image_path["other"]
        base = os.path.join(path, "{}.png")
        contents = [
            [base.format("combined_skymap"), base.format("compare_waveforms")]
        ]
        return contents

    def default_corner_params(self):
        """Return a list of default corner parameters used by the corner
        plotting function
        """
        return conf.gw_corner_parameters

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
        mydict = super(_WebpageGeneration, self).add_to_expert_pages(
            path, label
        )
        contour_base = path + "{}_2d_contour_{}_log_likelihood.png"
        mydict.update({
            "network_precessing_snr": [
                [
                    contour_base.format(label, "_b_bar"),
                    contour_base.format(label, "_precessing_harmonics_overlap"),
                ]
            ]
        })
        return mydict

    @property
    def additional_1d_pages(self):
        """Additional 1d histogram pages beyond one for each parameter. You may,
        for instance, want a 1d histogram page which combines multiple
        parameters. This returns a dictionary, keyed by the new 1d histogram
        page, with values indicating the parameters you wish to include on this
        page. Only the 1d marginalized histograms are shown.
        """
        return conf.additional_1d_pages
