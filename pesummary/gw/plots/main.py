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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from pesummary.core.plots.main import _PlotGeneration as _BasePlotGeneration
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.utils.utils import logger, resample_posterior_distribution
from pesummary.gw.plots import publication
from pesummary.gw.plots import plot as gw
from pesummary import conf


latex_labels.update(GWlatex_labels)


class _PlotGeneration(_BasePlotGeneration):
    def __init__(
        self, savedir=None, webdir=None, labels=None, samples=None,
        kde_plot=False, existing_labels=None, existing_injection_data=None,
        existing_file_kwargs=None, existing_samples=None,
        existing_metafile=None, same_parameters=None, injection_data=None,
        result_files=None, file_kwargs=None, colors=None, custom_plotting=None,
        add_to_existing=False, priors={}, no_ligo_skymap=False,
        nsamples_for_skymap=None, detectors=None, maxL_samples=None,
        gwdata=None, calibration=None, psd=None,
        multi_threading_for_skymap=None, approximant=None,
        pepredicates_probs=None, include_prior=False, publication=False,
        existing_approximant=None, existing_psd=None, existing_calibration=None,
        weights=None
    ):
        super(_PlotGeneration, self).__init__(
            savedir=savedir, webdir=webdir, labels=labels,
            samples=samples, kde_plot=kde_plot, existing_labels=existing_labels,
            existing_injection_data=existing_injection_data,
            existing_samples=existing_samples,
            same_parameters=same_parameters,
            injection_data=injection_data,
            colors=colors, custom_plotting=custom_plotting,
            add_to_existing=add_to_existing, priors=priors,
            include_prior=include_prior, weights=weights
        )
        self.file_kwargs = file_kwargs
        self.existing_file_kwargs = existing_file_kwargs
        self.no_ligo_skymap = no_ligo_skymap
        self.nsamples_for_skymap = nsamples_for_skymap
        self.detectors = detectors
        self.maxL_samples = maxL_samples
        self.gwdata = gwdata
        self.calibration = calibration
        self.existing_calibration = existing_calibration
        self.psd = psd
        self.existing_psd = existing_psd
        self.multi_threading_for_skymap = multi_threading_for_skymap
        self.approximant = approximant
        self.existing_approximant = existing_approximant
        self.pepredicates_probs = pepredicates_probs
        self.publication = publication

        self.plot_type_dictionary = {
            "corner": self.corner_plot,
            "oned_histogram": self.oned_histogram_plot,
            "sample_evolution": self.sample_evolution_plot,
            "autocorrelation": self.autocorrelation_plot,
            "oned_cdf": self.oned_cdf_plot,
            "oned_histogram_comparison": self.oned_histogram_comparison_plot,
            "oned_cdf_comparison": self.oned_cdf_comparison_plot,
            "box_plot_comparison": self.box_plot_comparison_plot,
            "custom": self.custom_plot,
            "psd": self.psd_plot,
            "calibration": self.calibration_plot,
            "skymap": self.skymap_plot,
            "waveform_fd": self.waveform_fd_plot,
            "waveform_td": self.waveform_td_plot,
            "data": self.strain_plot,
            "skymap_comparison": self.skymap_comparison_plot,
            "waveform_comparison_fd": self.waveform_comparison_fd_plot,
            "waveform_comparison_td": self.waveform_comparison_td_plot,
            "2d_comparison_contour": self.twod_comparison_contour_plot,
            "violin": self.violin_plot,
            "spin_disk": self.spin_dist_plot,
            "pepredicates": self.pepredicates_plot
        }

    def generate_plots(self):
        """Generate all plots for all result files
        """
        for i in self.labels:
            logger.debug("Starting to generate plots for {}".format(i))
            self._generate_plots(i)
        if self.calibration or "calibration" in list(self.priors.keys()):
            self.try_to_make_a_plot("calibration")
        if self.psd:
            self.try_to_make_a_plot("psd")
        if self.add_to_existing:
            self.add_existing_data()
        if len(self.samples) > 1:
            logger.debug("Starting to generate comparison plots")
            self._generate_comparison_plots()

    def _generate_plots(self, label):
        """Generate all plots for a given result file
        """
        self.try_to_make_a_plot("corner", label=label)
        self.try_to_make_a_plot("skymap", label=label)
        self.try_to_make_a_plot("waveform_td", label=label)
        self.try_to_make_a_plot("waveform_fd", label=label)
        if self.pepredicates_probs[label] is not None:
            self.try_to_make_a_plot("pepredicates", label=label)
        if self.gwdata:
            self.try_to_make_a_plot("data", label=label)
        self.try_to_make_a_plot("oned_histogram", label=label)
        self.try_to_make_a_plot("sample_evolution", label=label)
        self.try_to_make_a_plot("autocorrelation", label=label)
        self.try_to_make_a_plot("oned_cdf", label=label)
        if self.custom_plotting:
            self.try_to_make_a_plot("custom", label=label)

    def _generate_comparison_plots(self):
        """Generate all comparison plots
        """
        self.try_to_make_a_plot("skymap_comparison")
        self.try_to_make_a_plot("waveform_comparison_td")
        self.try_to_make_a_plot("waveform_comparison_fd")
        self.try_to_make_a_plot("oned_histogram_comparison")
        self.try_to_make_a_plot("oned_cdf_comparison")
        self.try_to_make_a_plot("box_plot_comparison")
        if self.publication:
            self.try_to_make_a_plot("2d_comparison_contour")
            self.try_to_make_a_plot("violin")
            self.try_to_make_a_plot("spin_disk")

    @staticmethod
    def _corner_plot(savedir, label, samples, latex_labels, webdir):
        """Generate a corner plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        label: str
            the label corresponding to the results file
        samples: dict
            dictionary of samples for a given result file
        latex_labels: dict
            dictionary of latex labels
        webdir: str
            directory where the javascript is written
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, params = gw._make_corner_plot(samples, latex_labels)
            plt.savefig(
                os.path.join(
                    savedir, "corner", "{}_all_density_plots.png".format(
                        label
                    )
                )
            )
            plt.close()
            combine_corner = open(
                os.path.join(webdir, "js", "combine_corner.js")
            )
            combine_corner = combine_corner.readlines()
            params = [str(i) for i in params]
            ind = [
                linenumber for linenumber, line in enumerate(combine_corner)
                if "var list = {}" in line
            ][0]
            combine_corner.insert(
                ind + 1, "    list['{}'] = {};\n".format(label, params)
            )
            new_file = open(
                os.path.join(webdir, "js", "combine_corner.js"), "w"
            )
            new_file.writelines(combine_corner)
            new_file.close()
            fig = gw._make_source_corner_plot(samples, latex_labels)
            plt.savefig(
                os.path.join(
                    savedir, "corner", "{}_sourceframe.png".format(label)
                )
            )
            plt.close()
            fig = gw._make_extrinsic_corner_plot(samples, latex_labels)
            plt.savefig(
                os.path.join(
                    savedir, "corner", "{}_extrinsic.png".format(label)
                )
            )
            plt.close()

    def skymap_plot(self, label):
        """Generate a skymap plot for a given result file

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        try:
            import ligo.skymap
            SKYMAP = True
        except ImportError:
            SKYMAP = False

        self._skymap_plot(
            self.savedir, self.samples[label]["ra"], self.samples[label]["dec"],
            label, self.weights[label]
        )

        if SKYMAP and not self.no_ligo_skymap:
            from pesummary.utils.utils import RedirectLogger

            logger.info("Launching subprocess to generate skymap plot with "
                        "ligo.skymap")
            with RedirectLogger("ligo.skymap", level="DEBUG") as redirector:
                process = mp.Process(
                    target=self._ligo_skymap_plot,
                    args=[
                        self.savedir, self.samples[label]["ra"],
                        self.samples[label]["dec"], label,
                        self.nsamples_for_skymap, self.webdir,
                        self.multi_threading_for_skymap
                    ]
                )
                process.start()

    @staticmethod
    def _skymap_plot(savedir, ra, dec, label, weights):
        """Generate a skymap plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        ra: pesummary.utils.utils.Array
            array containing the samples for right ascension
        dec: pesummary.utils.utils.Array
            array containing the samples for declination
        label: str
            the label corresponding to the results file
        weights: list
            list of weights for the samples
        """
        fig = gw._default_skymap_plot(ra, dec, weights)
        plt.savefig(
            os.path.join(savedir, "{}_skymap.png".format(label))
        )
        plt.close()

    @staticmethod
    def _ligo_skymap_plot(savedir, ra, dec, label, nsamples_for_skymap,
                          webdir, multi_threading_for_skymap):
        """Generate a skymap plot for a given set of samples using the
        ligo.skymap package

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        ra: pesummary.utils.utils.Array
            array containing the samples for right ascension
        dec: pesummary.utils.utils.Array
            array containing the samples for declination
        label: str
            the label corresponding to the results file
        nsamples_for_skymap: int
            the number of samples used to generate skymap
        webdir: str
            the directory to store the fits file
        """
        downsampled = False
        if nsamples_for_skymap is not None:
            ra, dec = resample_posterior_distribution(
                [ra, dec], nsamples_for_skymap
            )
            downsampled = True
        fig = gw._ligo_skymap_plot(
            ra, dec, savedir=os.path.join(webdir, "samples"),
            nprocess=multi_threading_for_skymap, downsamples=downsampled
        )
        plt.savefig(
            os.path.join(savedir, "{}_skymap.png".format(label))
        )
        plt.close()

    def waveform_fd_plot(self, label):
        """Generate a frequency domain waveform plot for a given result file

        Parameters
        ----------
        label: str
            the label corresponding to the results file
        """
        if self.approximant[label] == {}:
            return
        self._waveform_fd_plot(
            self.savedir, self.detectors[label], self.maxL_samples[label], label
        )

    @staticmethod
    def _waveform_fd_plot(savedir, detectors, maxL_samples, label):
        """Generate a frequency domain waveform plot for a given detector
        network and set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        detectors: list
            list of detectors used in your analysis
        maxL_samples: dict
            dictionary of maximum likelihood values
        label: str
            the label corresponding to the results file
        """
        if detectors is None:
            detectors = ["H1", "L1"]
        else:
            detectors = detectors.split("_")

        fig = gw._waveform_plot(detectors, maxL_samples)
        plt.savefig(
            os.path.join(savedir, "{}_waveform.png".format(label))
        )
        plt.close()

    def waveform_td_plot(self, label):
        """Generate a time domain waveform plot for a given result file

        Parameters
        ----------
        label: str
            the label corresponding to the results file
        """
        if self.approximant[label] == {}:
            return
        self._waveform_td_plot(
            self.savedir, self.detectors[label], self.maxL_samples[label], label
        )

    @staticmethod
    def _waveform_td_plot(savedir, detectors, maxL_samples, label):
        """Generate a time domain waveform plot for a given detector network
        and set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        detectors: list
            list of detectors used in your analysis
        maxL_samples: dict
            dictionary of maximum likelihood values
        label: str
            the label corresponding to the results file
        """
        if detectors is None:
            detectors = ["H1", "L1"]
        else:
            detectors = detectors.split("_")

        fig = gw._time_domain_waveform(detectors, maxL_samples)
        plt.savefig(
            os.path.join(
                savedir, "{}_waveform_time_domain.png".format(label)
            )
        )
        plt.close()

    def strain_plot(self, label):
        """Generate a plot showing the comparison between the data and the
        maxL waveform gfor a given result file

        Parameters
        ----------
        label: str
            the label corresponding to the results file
        """
        from pesummary.utils.utils import RedirectLogger

        logger.info("Launching subprocess to generate strain plot")
        process = mp.Process(
            target=self._strain_plot,
            args=[self.savedir, self.gwdata, self.maxL_samples[label], label]
        )
        process.start()

    @staticmethod
    def _strain_plot(savedir, gwdata, maxL_samples, label):
        """Generate a strain plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory to save the plot
        gwdata: dict
            dictionary of strain data for each detector
        maxL_samples: dict
            dictionary of maximum likelihood values
        label: str
            the label corresponding to the results file
        """
        fig = gw._strain_plot(gwdata, maxL_samples)
        plt.savefig(
            os.path.join(savedir, "{}_strain.png".format(label))
        )
        plt.close()

    def skymap_comparison_plot(self, label):
        """Generate a plot to compare skymaps for all result files

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        self._skymap_comparison_plot(
            self.savedir, self.same_samples["ra"], self.same_samples["dec"],
            self.labels, self.colors
        )

    @staticmethod
    def _skymap_comparison_plot(savedir, ra, dec, labels, colors):
        """Generate a plot to compare skymaps for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        ra: dict
            dictionary of right ascension samples for each result file
        dec: dict
            dictionary of declination samples for each result file
        labels: list
            list of labels to distinguish each result file
        colors: list
            list of colors to be used to distinguish different result files
        """
        ra_list = [ra[key] for key in labels]
        dec_list = [dec[key] for key in labels]
        fig = gw._sky_map_comparison_plot(ra_list, dec_list, labels, colors)
        plt.savefig(os.path.join(savedir, "combined_skymap.png"))
        plt.close()

    def waveform_comparison_fd_plot(self, label):
        """Generate a plot to compare the frequency domain waveform

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        if any(self.approximant[i] == {} for i in self.labels):
            return

        self._waveform_comparison_fd_plot(
            self.savedir, self.maxL_samples, self.labels, self.colors
        )

    @staticmethod
    def _waveform_comparison_fd_plot(savedir, maxL_samples, labels, colors):
        """Generate a plot to compare the frequency domain waveforms

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        maxL_samples: dict
            dictionary of maximum likelihood samples for each result file
        labels: list
            list of labels to distinguish each result file
        colors: list
            list of colors to be used to distinguish different result files
        """
        samples = [maxL_samples[i] for i in labels]
        fig = gw._waveform_comparison_plot(samples, colors, labels)
        plt.savefig(os.path.join(savedir, "compare_waveforms.png"))
        plt.close()

    def waveform_comparison_td_plot(self, label):
        """Generate a plot to compare the time domain waveform

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        if any(self.approximant[i] == {} for i in self.labels):
            return

        self._waveform_comparison_fd_plot(
            self.savedir, self.maxL_samples, self.labels, self.colors
        )

    @staticmethod
    def _waveform_comparison_td_plot(savedir, maxL_samples, labels, colors):
        """Generate a plot to compare the time domain waveforms

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        maxL_samples: dict
            dictionary of maximum likelihood samples for each result file
        labels: list
            list of labels to distinguish each result file
        colors: list
            list of colors to be used to distinguish different result files
        """
        samples = [maxL_samples[i] for i in labels]
        fig = gw._time_domainwaveform_comparison_plot(samples, colors, labels)
        plt.savefig(os.path.join(savedir, "compare_time_domain_waveforms.png"))
        plt.close()

    def twod_comparison_contour_plot(self, label):
        """Generate 2d comparison contour plots

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        error_message = (
            "Failed to generate a 2d contour plot for %s because {}"
        )
        twod_plots = [
            ["mass_ratio", "chi_eff"], ["mass_1", "mass_2"],
            ["luminosity_distance", "chirp_mass_source"],
            ["mass_1_source", "mass_2_source"],
            ["theta_jn", "luminosity_distance"],
            ["network_optimal_snr", "chirp_mass_source"]
        ]
        for plot in twod_plots:
            if not all(
                all(
                    i in self.samples[j].keys() for i in plot
                ) for j in self.labels
            ):
                logger.warn(
                    "Failed to generate 2d contour plots for {} because {} are not "
                    "common in all result files".format(
                        " and ".join(plot), " and ".join(plot)
                    )
                )
                continue
            samples = [[self.samples[i][j] for j in plot] for i in self.labels]
            arguments = [
                self.savedir, plot, samples, self.labels, latex_labels
            ]
            self._try_to_make_a_plot(
                arguments, self._twod_comparison_contour_plot,
                error_message % (" and ".join(plot))
            )

    @staticmethod
    def _twod_comparison_contour_plot(
        savedir, plot_parameters, samples, labels, latex_labels
    ):
        """Generate a 2d comparison contour plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        plot_parameters: list
            list of parameters to use for the 2d contour plot
        samples: list
            list of samples for each parameter
        labels: list
            list of labels used to distinguish each result file
        latex_labels: dict
            dictionary containing the latex labels for each parameter
        """
        fig = publication.twod_contour_plots(
            plot_parameters, samples, labels, latex_labels
        )
        plt.savefig(
            os.path.join(
                savedir, "publication", "2d_contour_plot_{}.png".format(
                    "_and_".join(plot_parameters)
                )
            )
        )
        plt.close()

    def violin_plot(self, label):
        """Generate violin plot to compare certain parameters in all result
        files

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        error_message = (
            "Failed to generate a violin plot for %s because {}"
        )
        violin_plots = ["mass_ratio", "chi_eff", "chi_p", "luminosity_distance"]

        for plot in violin_plots:
            if not all(plot in self.samples[j].keys() for j in self.labels):
                logger.warn(
                    "Failed to generate violin plots for {} because {} is not "
                    "common in all result files".format(plot, plot)
                )
            samples = [self.samples[i][plot] for i in self.labels]
            arguments = [
                self.savedir, plot, samples, self.labels, latex_labels[plot]
            ]
            self._try_to_make_a_plot(
                arguments, self._violin_plot, error_message % (plot)
            )

    @staticmethod
    def _violin_plot(savedir, plot_parameter, samples, labels, latex_label):
        """Generate a violin plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        plot_parameter: str
            name of the parameter you wish to generate a violin plot for
        samples: list
            list of samples for each parameter
        labels: list
            list of labels used to distinguish each result file
        latex_label: str
             latex_label correspondig to parameter
        """
        fig = publication.violin_plots(
            plot_parameter, samples, labels, latex_labels
        )
        plt.savefig(
            os.path.join(
                savedir, "publication", "violin_plot_{}.png".format(
                    plot_parameter
                )
            )
        )
        plt.close()

    def spin_dist_plot(self, label):
        """Generate a spin disk plot to compare spins in all result
        files

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        error_message = (
            "Failed to generate a spin disk plot for %s because {}"
        )
        parameters = ["a_1", "a_2", "tilt_1", "tilt_2"]
        for num, label in enumerate(self.labels):
            if not all(i in self.samples[label].keys() for i in parameters):
                logger.warn(
                    "Failed to generate spin disk plots because {} are not "
                    "common in all result files".format(
                        " and ".join(parameters)
                    )
                )
                continue
            samples = [self.samples[label][i] for i in parameters]
            arguments = [
                self.savedir, parameters, samples, label, self.colors[num]
            ]

            self._try_to_make_a_plot(
                arguments, self._spin_dist_plot, error_message % (label)
            )

    @staticmethod
    def _spin_dist_plot(savedir, parameters, samples, label, color):
        """Generate a spin disk plot for a given set of samples

        Parameters
        ----------
        """
        fig = publication.spin_distribution_plots(
            parameters, samples, label, color
        )
        plt.savefig(
            os.path.join(
                savedir, "publication", "spin_disk_plot_{}.png".format(
                    label
                )
            )
        )
        plt.close()

    def pepredicates_plot(self, label):
        """Generate plots with the PEPredicates package

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        self._pepredicates_plot(
            self.savedir, self.samples[label], label,
            self.pepredicates_probs[label]["default"], population_prior=False
        )
        self._pepredicates_plot(
            self.savedir, self.samples[label], label,
            self.pepredicates_probs[label]["population"], population_prior=True
        )

    @staticmethod
    def _pepredicates_plot(
        savedir, samples, label, probabilities, population_prior=False
    ):
        """Generate a plot with the PEPredicates package for a given set of
        samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        samples: dict
            dictionary of samples for each parameter
        label: str
            the label corresponding to the result file
        probabilities: dict
            dictionary of classification probabilities
        population_prior: Bool, optional
            if True, the samples will be reweighted according to a population
            prior
        """
        from pesummary.gw.pepredicates import PEPredicates

        parameters = list(samples.keys())
        samples = [
            [samples[parameter][j] for parameter in parameters] for j in
            range(len(samples[parameters[0]]))
        ]
        fig = PEPredicates.plot(
            samples, parameters, population_prior=population_prior
        )
        if not population_prior:
            plt.savefig(
                os.path.join(
                    savedir, "{}_default_pepredicates.png".format(
                        label
                    )
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    savedir, "{}_population_pepredicates.png".format(
                        label
                    )
                )
            )
        fig = gw._classification_plot(probabilities)
        if not population_prior:
            plt.savefig(
                os.path.join(
                    savedir, "{}_default_pepredicates_bar.png".format(
                        label
                    )
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    savedir, "{}_population_pepredicates_bar.png".format(
                        label
                    )
                )
            )
        plt.close()

    def psd_plot(self, label):
        """Generate a psd plot for a given result file

        Parameters
        ----------
        label: str
            the label corresponding to the result file
        """
        error_message = (
            "Failed to generate a PSD plot for %s because {}"
        )

        fmin = None

        for num, label in enumerate(self.labels):
            if list(self.psd[label].keys()) == [None]:
                return
            if list(self.psd[label].keys()) == []:
                return
            if "f_low" in list(self.file_kwargs[label]["sampler"].keys()):
                fmin = self.file_kwargs[label]["sampler"]["f_low"]
            labels = list(self.psd[label].keys())
            frequencies = [np.array(self.psd[label][i]).T[0] for i in labels]
            strains = [np.array(self.psd[label][i]).T[1] for i in labels]
            arguments = [
                self.savedir, frequencies, strains, fmin, labels, label
            ]

            self._try_to_make_a_plot(
                arguments, self._psd_plot, error_message % (label)
            )

    @staticmethod
    def _psd_plot(savedir, frequencies, strains, fmin, psd_labels, label):
        """Generate a psd plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        frequencies: list
            list of psd frequencies for each IFO
        strains: list
            list of psd strains for each IFO
        fmin: float
            frequency to start the psd plotting
        psd_labels: list
            list of IFOs used
        label: str
            the label used to distinguish the result file
        """
        fig = gw._psd_plot(
            frequencies, strains, labels=psd_labels, fmin=fmin
        )
        plt.savefig(
            os.path.join(savedir, "{}_psd_plot.png".format(label))
        )
        plt.close()

    def calibration_plot(self, label):
        """Generate a calibration plot for a given result file

        Parameters
        ----------
        label: str
            the label corresponding to the result file
        """
        import numpy as np

        error_message = (
            "Failed to generate calibration plot for %s because {}"
        )
        frequencies = np.arange(20., 1024., 1. / 4)

        for num, label in enumerate(self.labels):
            if list(self.calibration[label].keys()) == [None]:
                return
            if list(self.calibration[label].keys()) == []:
                return

            ifos = list(self.calibration[label].keys())
            calibration_data = [
                self.calibration[label][i] for i in ifos
            ]
            if "calibration" in self.priors.keys():
                prior = [self.priors["calibration"][label][i] for i in ifos]
            else:
                prior = None
            arguments = [
                self.savedir, frequencies, calibration_data, ifos, prior,
                label
            ]
            self._try_to_make_a_plot(
                arguments, self._calibration_plot, error_message % (label)
            )

    @staticmethod
    def _calibration_plot(
        savedir, frequencies, calibration_data, calibration_labels, prior, label
    ):
        """Generate a calibration plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        frequencies: list
            list of frequencies used to interpolate the calibration data
        calibration_data: list
            list of calibration data for each IFO
        calibration_labels: list
            list of IFOs used
        prior: list
            list containing the priors used for each IFO
        label: str
            the label used to distinguish the result file
        """
        fig = gw._calibration_envelope_plot(
            frequencies, calibration_data, calibration_labels, prior=prior
        )
        plt.savefig(
            os.path.join(savedir, "{}_calibration_plot.png".format(label))
        )
        plt.close()
