#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import os

from pesummary.core.plots.main import _PlotGeneration as _BasePlotGeneration
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.core.plots import interactive
from pesummary.core.plots.figure import figure
from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.utils.utils import (
    logger, resample_posterior_distribution, get_matplotlib_backend,
    get_matplotlib_style_file
)
from pesummary.utils.decorators import no_latex_plot
from pesummary.gw.plots import publication
from pesummary.gw.plots import plot as gw
from pesummary import conf

import multiprocessing as mp
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
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
        existing_weights=None, weights=None, disable_comparison=False,
        linestyles=None, disable_interactive=False, disable_corner=False,
        publication_kwargs={}, multi_process=1, mcmc_samples=False,
        skymap=None, existing_skymap=None, corner_params=None,
        preliminary_pages=False, expert_plots=True, checkpoint=False
    ):
        super(_PlotGeneration, self).__init__(
            savedir=savedir, webdir=webdir, labels=labels,
            samples=samples, kde_plot=kde_plot, existing_labels=existing_labels,
            existing_injection_data=existing_injection_data,
            existing_samples=existing_samples,
            existing_weights=existing_weights,
            same_parameters=same_parameters,
            injection_data=injection_data, mcmc_samples=mcmc_samples,
            colors=colors, custom_plotting=custom_plotting,
            add_to_existing=add_to_existing, priors=priors,
            include_prior=include_prior, weights=weights,
            disable_comparison=disable_comparison, linestyles=linestyles,
            disable_interactive=disable_interactive, disable_corner=disable_corner,
            multi_process=multi_process, corner_params=corner_params,
            expert_plots=expert_plots, checkpoint=checkpoint
        )
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
        self.preliminary_comparison_pages = any(
            value for value in self.preliminary_pages.values()
        )
        self.package = "gw"
        self.file_kwargs = file_kwargs
        self.existing_file_kwargs = existing_file_kwargs
        self.no_ligo_skymap = no_ligo_skymap
        self.nsamples_for_skymap = nsamples_for_skymap
        self.detectors = detectors
        self.maxL_samples = maxL_samples
        self.gwdata = gwdata
        if skymap is None:
            skymap = {label: None for label in self.labels}
        self.skymap = skymap
        self.existing_skymap = skymap
        self.calibration = calibration
        self.existing_calibration = existing_calibration
        self.psd = psd
        self.existing_psd = existing_psd
        self.multi_threading_for_skymap = multi_threading_for_skymap
        self.approximant = approximant
        self.existing_approximant = existing_approximant
        self.pepredicates_probs = pepredicates_probs
        self.publication = publication
        self.publication_kwargs = publication_kwargs
        self._ligo_skymap_PID = {}

        self.plot_type_dictionary.update({
            "psd": self.psd_plot,
            "calibration": self.calibration_plot,
            "skymap": self.skymap_plot,
            "waveform_fd": self.waveform_fd_plot,
            "waveform_td": self.waveform_td_plot,
            "data": self.gwdata_plots,
            "violin": self.violin_plot,
            "spin_disk": self.spin_dist_plot,
            "pepredicates": self.pepredicates_plot
        })
        if self.make_comparison:
            self.plot_type_dictionary.update({
                "skymap_comparison": self.skymap_comparison_plot,
                "waveform_comparison_fd": self.waveform_comparison_fd_plot,
                "waveform_comparison_td": self.waveform_comparison_td_plot,
                "2d_comparison_contour": self.twod_comparison_contour_plot,
            })

    @property
    def ligo_skymap_PID(self):
        return self._ligo_skymap_PID

    def generate_plots(self):
        """Generate all plots for all result files
        """
        if self.calibration or "calibration" in list(self.priors.keys()):
            self.try_to_make_a_plot("calibration")
        if self.psd:
            self.try_to_make_a_plot("psd")
        super(_PlotGeneration, self).generate_plots()

    def _generate_plots(self, label):
        """Generate all plots for a given result file
        """
        super(_PlotGeneration, self)._generate_plots(label)
        self.try_to_make_a_plot("skymap", label=label)
        self.try_to_make_a_plot("waveform_td", label=label)
        self.try_to_make_a_plot("waveform_fd", label=label)
        if self.pepredicates_probs[label] is not None:
            self.try_to_make_a_plot("pepredicates", label=label)
        if self.gwdata:
            self.try_to_make_a_plot("data", label=label)

    def _generate_comparison_plots(self):
        """Generate all comparison plots
        """
        super(_PlotGeneration, self)._generate_comparison_plots()
        self.try_to_make_a_plot("skymap_comparison")
        self.try_to_make_a_plot("waveform_comparison_td")
        self.try_to_make_a_plot("waveform_comparison_fd")
        if self.publication:
            self.try_to_make_a_plot("2d_comparison_contour")
            self.try_to_make_a_plot("violin")
            self.try_to_make_a_plot("spin_disk")

    @staticmethod
    def _corner_plot(
        savedir, label, samples, latex_labels, webdir, params, preliminary=False,
        checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filename = os.path.join(
                savedir, "corner", "{}_all_density_plots.png".format(label)
            )
            if os.path.isfile(filename) and checkpoint:
                pass
            else:
                fig, params, data = gw._make_corner_plot(
                    samples, latex_labels, corner_parameters=params
                )
                fig.savefig(filename)
                fig.close()
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
                combine_corner = open(
                    os.path.join(webdir, "js", "combine_corner.js")
                )
                combine_corner = combine_corner.readlines()
                params = [str(i) for i in params]
                ind = [
                    linenumber for linenumber, line in enumerate(combine_corner)
                    if "var data = {}" in line
                ][0]
                combine_corner.insert(
                    ind + 1, "    data['{}'] = {};\n".format(label, data)
                )
                new_file = open(
                    os.path.join(webdir, "js", "combine_corner.js"), "w"
                )
                new_file.writelines(combine_corner)
                new_file.close()

            filename = os.path.join(
                savedir, "corner", "{}_sourceframe.png".format(label)
            )
            if os.path.isfile(filename) and checkpoint:
                pass
            else:
                fig = gw._make_source_corner_plot(samples, latex_labels)
                fig.savefig(filename)
                fig.close()
            filename = os.path.join(
                savedir, "corner", "{}_extrinsic.png".format(label)
            )
            if os.path.isfile(filename) and checkpoint:
                pass
            else:
                fig = gw._make_extrinsic_corner_plot(samples, latex_labels)
                fig.savefig(filename)
                fig.close()

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

        if self.mcmc_samples:
            samples = self.samples[label].combine
        else:
            samples = self.samples[label]
        _injection = [
            self.injection_data[label]["ra"], self.injection_data[label]["dec"]
        ]
        self._skymap_plot(
            self.savedir, samples["ra"], samples["dec"], label,
            self.weights[label], _injection,
            preliminary=self.preliminary_pages[label]
        )

        if SKYMAP and not self.no_ligo_skymap and self.skymap[label] is None:
            from pesummary.utils.utils import RedirectLogger

            logger.info("Launching subprocess to generate skymap plot with "
                        "ligo.skymap")
            try:
                _time = samples["geocent_time"]
            except KeyError:
                logger.warning(
                    "Unable to find 'geocent_time' in the posterior table for {}. "
                    "The ligo.skymap fits file will therefore not store the "
                    "DATE_OBS field in the header".format(label)
                )
                _time = None
            with RedirectLogger("ligo.skymap", level="DEBUG") as redirector:
                process = mp.Process(
                    target=self._ligo_skymap_plot,
                    args=[
                        self.savedir, samples["ra"], samples["dec"],
                        samples["luminosity_distance"], _time,
                        label, self.nsamples_for_skymap, self.webdir,
                        self.multi_threading_for_skymap, _injection,
                        self.preliminary_pages[label]
                    ]
                )
                process.start()
                PID = process.pid
            self._ligo_skymap_PID[label] = PID
        elif SKYMAP and not self.no_ligo_skymap:
            self._ligo_skymap_array_plot(
                self.savedir, self.skymap[label], label,
                self.preliminary_pages[label]
            )

    @staticmethod
    @no_latex_plot
    def _skymap_plot(
        savedir, ra, dec, label, weights, injection=None, preliminary=False
    ):
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
        injection: list, optional
            list containing the injected value of ra and dec
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        import math

        if injection is not None and any(math.isnan(inj) for inj in injection):
            injection = None
        fig = gw._default_skymap_plot(ra, dec, weights, injection=injection)
        _PlotGeneration.save(
            fig, os.path.join(savedir, "{}_skymap".format(label)),
            preliminary=preliminary
        )

    @staticmethod
    @no_latex_plot
    def _ligo_skymap_plot(savedir, ra, dec, dist, time, label, nsamples_for_skymap,
                          webdir, multi_threading_for_skymap, injection,
                          preliminary=False):
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
        dist: pesummary.utils.utils.Array
            array containing the samples for luminosity distance
        time: pesummary.utils.utils.Array
            array containing the samples for the geocentric time of merger
        label: str
            the label corresponding to the results file
        nsamples_for_skymap: int
            the number of samples used to generate skymap
        webdir: str
            the directory to store the fits file
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        import math

        downsampled = False
        if nsamples_for_skymap is not None:
            ra, dec, dist = resample_posterior_distribution(
                [ra, dec, dist], nsamples_for_skymap
            )
            downsampled = True
        if injection is not None and any(math.isnan(inj) for inj in injection):
            injection = None
        fig = gw._ligo_skymap_plot(
            ra, dec, dist=dist, savedir=os.path.join(webdir, "samples"),
            nprocess=multi_threading_for_skymap, downsampled=downsampled,
            label=label, time=time, injection=injection
        )
        _PlotGeneration.save(
            fig, os.path.join(savedir, "{}_skymap".format(label)),
            preliminary=preliminary
        )

    @staticmethod
    @no_latex_plot
    def _ligo_skymap_array_plot(savedir, skymap, label, preliminary=False):
        """Generate a skymap based on skymap probability array already generated with
        `ligo.skymap`

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        skymap: np.ndarray
            array of skymap probabilities
        label: str
            the label corresponding to the results file
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        fig = gw._ligo_skymap_plot_from_array(skymap)
        _PlotGeneration.save(
            fig, os.path.join(savedir, "{}_skymap".format(label)),
            preliminary=preliminary
        )

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
            self.savedir, self.detectors[label], self.maxL_samples[label], label,
            self.preliminary_pages[label], self.checkpoint
        )

    @staticmethod
    def _waveform_fd_plot(
        savedir, detectors, maxL_samples, label, preliminary=False,
        checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(savedir, "{}_waveform.png".format(label))
        if os.path.isfile(filename) and checkpoint:
            return
        if detectors is None:
            detectors = ["H1", "L1"]
        else:
            detectors = detectors.split("_")

        fig = gw._waveform_plot(detectors, maxL_samples)
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
            self.savedir, self.detectors[label], self.maxL_samples[label], label,
            self.preliminary_pages[label], self.checkpoint
        )

    @staticmethod
    def _waveform_td_plot(
        savedir, detectors, maxL_samples, label, preliminary=False,
        checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(
            savedir, "{}_waveform_time_domain.png".format(label)
        )
        if os.path.isfile(filename) and checkpoint:
            return
        if detectors is None:
            detectors = ["H1", "L1"]
        else:
            detectors = detectors.split("_")

        fig = gw._time_domain_waveform(detectors, maxL_samples)
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

    def gwdata_plots(self, label):
        """Generate all plots associated with the gwdata

        Parameters
        ----------
        label: str
            the label corresponding to the results file
        """
        from pesummary.utils.utils import determine_gps_time_and_window

        base_error = "Failed to generate a %s because {}"
        gps_time, window = determine_gps_time_and_window(
            self.maxL_samples, self.labels
        )
        functions = [
            self.strain_plot, self.spectrogram_plot, self.omegascan_plot
        ]
        args = [[label], [], [gps_time, window]]
        func_names = ["strain_plot", "spectrogram plot", "omegascan plot"]

        for func, args, name in zip(functions, args, func_names):
            self._try_to_make_a_plot(args, func, base_error % (name))
            continue

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
    def _strain_plot(savedir, gwdata, maxL_samples, label, checkpoint=False):
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
        filename = os.path.join(savedir, "{}_strain.png".format(label))
        if os.path.isfile(filename) and checkpoint:
            return
        fig = gw._strain_plot(gwdata, maxL_samples)
        _PlotGeneration.save(fig, filename)

    def spectrogram_plot(self):
        """Generate a plot showing the spectrogram for all detectors
        """
        figs = self._spectrogram_plot(self.savedir, self.gwdata)

    @staticmethod
    def _spectrogram_plot(savedir, strain):
        """Generate a plot showing the spectrogram for all detectors

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        strain: dict
            dictionary of gwpy timeseries objects containing the strain data for
            each IFO
        """
        from pesummary.gw.plots import detchar

        figs = detchar.spectrogram(strain)
        for det, fig in figs.items():
            _PlotGeneration.save(
                fig, os.path.join(savedir, "spectrogram_{}".format(det))
            )

    def omegascan_plot(self, gps_time, window):
        """Generate a plot showing the omegascan for all detectors

        Parameters
        ----------
        gps_time: float
            time around which to centre the omegascan
        window: float
            window around gps time to generate plot for
        """
        figs = self._omegascan_plot(
            self.savedir, self.gwdata, gps_time, window
        )

    @staticmethod
    def _omegascan_plot(savedir, strain, gps, window):
        """Generate a plot showing the spectrogram for all detectors

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        strain: dict
            dictionary of gwpy timeseries objects containing the strain data for
            each IFO
        gps: float
            time around which to centre the omegascan
        window: float
            window around gps time to generate plot for
        """
        from pesummary.gw.plots import detchar

        figs = detchar.omegascan(strain, gps, window=window)
        for det, fig in figs.items():
            _PlotGeneration.save(
                fig, os.path.join(savedir, "omegascan_{}".format(det))
            )

    def skymap_comparison_plot(self, label):
        """Generate a plot to compare skymaps for all result files

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        self._skymap_comparison_plot(
            self.savedir, self.same_samples["ra"], self.same_samples["dec"],
            self.labels, self.colors, self.preliminary_comparison_pages,
            self.checkpoint
        )

    @staticmethod
    def _skymap_comparison_plot(
        savedir, ra, dec, labels, colors, preliminary=False, checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(savedir, "combined_skymap.png")
        if os.path.isfile(filename) and checkpoint:
            return
        ra_list = [ra[key] for key in labels]
        dec_list = [dec[key] for key in labels]
        fig = gw._sky_map_comparison_plot(ra_list, dec_list, labels, colors)
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
            self.savedir, self.maxL_samples, self.labels, self.colors,
            self.preliminary_comparison_pages, self.checkpoint
        )

    @staticmethod
    def _waveform_comparison_fd_plot(
        savedir, maxL_samples, labels, colors, preliminary=False,
        checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(savedir, "compare_waveforms.png")
        if os.path.isfile(filename) and checkpoint:
            return
        samples = [maxL_samples[i] for i in labels]
        fig = gw._waveform_comparison_plot(samples, colors, labels)
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
            self.savedir, self.maxL_samples, self.labels, self.colors,
            self.preliminary_comparison_pages, self.checkpoint
        )

    @staticmethod
    def _waveform_comparison_td_plot(
        savedir, maxL_samples, labels, colors, preliminary=False,
        checkpoint=False
    ):
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(savedir, "compare_time_domain_waveforms.png")
        if os.path.isfile(filename) and checkpoint:
            return
        samples = [maxL_samples[i] for i in labels]
        fig = gw._time_domainwaveform_comparison_plot(samples, colors, labels)
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
        gridsize = (
            int(self.publication_kwargs["gridsize"]) if "gridsize" in
            self.publication_kwargs.keys() else 100
        )
        for plot in twod_plots:
            if not all(
                all(
                    i in self.samples[j].keys() for i in plot
                ) for j in self.labels
            ):
                logger.warning(
                    "Failed to generate 2d contour plots for {} because {} are not "
                    "common in all result files".format(
                        " and ".join(plot), " and ".join(plot)
                    )
                )
                continue
            samples = [[self.samples[i][j] for j in plot] for i in self.labels]
            arguments = [
                self.savedir, plot, samples, self.labels, latex_labels,
                self.colors, self.linestyles, gridsize,
                self.preliminary_comparison_pages, self.checkpoint
            ]
            self._try_to_make_a_plot(
                arguments, self._twod_comparison_contour_plot,
                error_message % (" and ".join(plot))
            )

    @staticmethod
    def _twod_comparison_contour_plot(
        savedir, plot_parameters, samples, labels, latex_labels, colors,
        linestyles, gridsize, preliminary=False, checkpoint=False
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
        gridsize: int
            the number of points to use when estimating the KDE
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(
            savedir, "publication", "2d_contour_plot_{}.png".format(
                "_and_".join(plot_parameters)
            )
        )
        if os.path.isfile(filename) and checkpoint:
            return
        fig = publication.twod_contour_plots(
            plot_parameters, samples, labels, latex_labels, colors=colors,
            linestyles=linestyles, gridsize=gridsize
        )
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
            injection = [self.injection_data[label][plot] for label in self.labels]
            if not all(plot in self.samples[j].keys() for j in self.labels):
                logger.warning(
                    "Failed to generate violin plots for {} because {} is not "
                    "common in all result files".format(plot, plot)
                )
            samples = [self.samples[i][plot] for i in self.labels]
            arguments = [
                self.savedir, plot, samples, self.labels, latex_labels[plot],
                injection, self.preliminary_comparison_pages, self.checkpoint
            ]
            self._try_to_make_a_plot(
                arguments, self._violin_plot, error_message % (plot)
            )

    @staticmethod
    def _violin_plot(
        savedir, plot_parameter, samples, labels, latex_label, inj_values=None,
        preliminary=False, checkpoint=False, kde=Bounded_1d_kde,
        default_bounds=True
    ):
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
        inj_value: list
             list of injected values for each sample
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(
            savedir, "publication", "violin_plot_{}.png".format(plot_parameter)
        )
        if os.path.isfile(filename) and checkpoint:
            return
        xlow, xhigh = None, None
        if default_bounds:
            xlow, xhigh = gw._return_bounds(
                plot_parameter, samples, comparison=True
            )
        fig = publication.violin_plots(
            plot_parameter, samples, labels, latex_labels, kde=kde,
            kde_kwargs={"xlow": xlow, "xhigh": xhigh}, inj_values=inj_values
        )
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

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
        parameters = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        for num, label in enumerate(self.labels):
            if not all(i in self.samples[label].keys() for i in parameters):
                logger.warning(
                    "Failed to generate spin disk plots because {} are not "
                    "common in all result files".format(
                        " and ".join(parameters)
                    )
                )
                continue
            samples = [self.samples[label][i] for i in parameters]
            arguments = [
                self.savedir, parameters, samples, label, self.colors[num],
                self.preliminary_comparison_pages, self.checkpoint
            ]

            self._try_to_make_a_plot(
                arguments, self._spin_dist_plot, error_message % (label)
            )

    @staticmethod
    def _spin_dist_plot(
        savedir, parameters, samples, label, color, preliminary=False,
        checkpoint=False
    ):
        """Generate a spin disk plot for a given set of samples

        Parameters
        ----------
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        filename = os.path.join(
            savedir, "publication", "spin_disk_plot_{}.png".format(label)
        )
        if os.path.isfile(filename) and checkpoint:
            return
        fig = publication.spin_distribution_plots(
            parameters, samples, label, color=color
        )
        _PlotGeneration.save(
            fig, filename, preliminary=preliminary
        )

    def pepredicates_plot(self, label):
        """Generate plots with the PEPredicates package

        Parameters
        ----------
        label: str
            the label for the results file that you wish to plot
        """
        if self.mcmc_samples:
            samples = self.samples[label].combine
        else:
            samples = self.samples[label]
        self._pepredicates_plot(
            self.savedir, samples, label,
            self.pepredicates_probs[label]["default"], population_prior=False,
            preliminary=self.preliminary_pages[label], checkpoint=self.checkpoint
        )
        self._pepredicates_plot(
            self.savedir, samples, label,
            self.pepredicates_probs[label]["population"], population_prior=True,
            preliminary=self.preliminary_pages[label], checkpoint=self.checkpoint
        )

    @staticmethod
    @no_latex_plot
    def _pepredicates_plot(
        savedir, samples, label, probabilities, population_prior=False,
        preliminary=False, checkpoint=False
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
        preliminary: Bool, optional
            if True, add a preliminary watermark to the plot
        """
        from pesummary.gw.pepredicates import PEPredicates

        parameters = list(samples.keys())
        samples = np.array([samples[param] for param in parameters]).T
        if not population_prior:
            filename = os.path.join(
                savedir, "{}_default_pepredicates.png".format(label)
            )
        else:
            filename = os.path.join(
                savedir, "{}_population_pepredicates.png".format(label)
            )

        if os.path.isfile(filename) and checkpoint:
            pass
        else:
            fig = PEPredicates.plot(
                samples, parameters, population_prior=population_prior
            )
            if not population_prior:
                _PlotGeneration.save(
                    fig, filename, preliminary=preliminary
                )
            else:
                _PlotGeneration.save(
                    fig, filename, preliminary=preliminary
                )

        if not population_prior:
            filename = os.path.join(
                savedir, "{}_default_pepredicates_bar.png".format(label)
            )
        else:
            filename = os.path.join(
                savedir, "{}_population_pepredicates_bar.png".format(label)
            )
        if os.path.isfile(filename) and checkpoint:
            pass
        else:
            fig = gw._classification_plot(probabilities)
            if not population_prior:
                _PlotGeneration.save(
                    fig, filename, preliminary=preliminary
                )
            else:
                _PlotGeneration.save(
                    fig, filename, preliminary=preliminary
                )

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
                self.savedir, frequencies, strains, fmin, labels, label,
                self.checkpoint
            ]

            self._try_to_make_a_plot(
                arguments, self._psd_plot, error_message % (label)
            )

    @staticmethod
    def _psd_plot(
        savedir, frequencies, strains, fmin, psd_labels, label, checkpoint=False
    ):
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
        filename = os.path.join(savedir, "{}_psd_plot.png".format(label))
        if os.path.isfile(filename) and checkpoint:
            return
        fig = gw._psd_plot(
            frequencies, strains, labels=psd_labels, fmin=fmin
        )
        _PlotGeneration.save(fig, filename)

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
                label, self.checkpoint
            ]
            self._try_to_make_a_plot(
                arguments, self._calibration_plot, error_message % (label)
            )

    @staticmethod
    def _calibration_plot(
        savedir, frequencies, calibration_data, calibration_labels, prior, label,
        checkpoint=False
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
        filename = os.path.join(
            savedir, "{}_calibration_plot.png".format(label)
        )
        if os.path.isfile(filename) and checkpoint:
            return
        fig = gw._calibration_envelope_plot(
            frequencies, calibration_data, calibration_labels, prior=prior
        )
        _PlotGeneration.save(fig, filename)

    @staticmethod
    def _interactive_corner_plot(
        savedir, label, samples, latex_labels, checkpoint=False
    ):
        """Generate an interactive corner plot for a given set of samples

        Parameters
        ----------
        savedir: str
            the directory you wish to save the plot in
        label: str
            the label corresponding to the results file
        samples: dict
            dictionary containing PESummary.utils.utils.Array objects that
            contain samples for each parameter
        latex_labels: str
            latex labels for each parameter in samples
        """
        filename = os.path.join(
            savedir, "corner", "{}_interactive_source.html".format(label)
        )
        if os.path.isfile(filename) and checkpoint:
            pass
        else:
            source_parameters = [
                "luminosity_distance", "mass_1_source", "mass_2_source",
                "total_mass_source", "chirp_mass_source", "redshift"
            ]
            parameters = [i for i in samples.keys() if i in source_parameters]
            data = [samples[parameter] for parameter in parameters]
            labels = [latex_labels[parameter] for parameter in parameters]
            _ = interactive.corner(
                data, labels, write_to_html_file=filename,
                dimensions={"width": 900, "height": 900}
            )

        filename = os.path.join(
            savedir, "corner", "{}_interactive_extrinsic.html".format(label)
        )
        if os.path.isfile(filename) and checkpoint:
            pass
        else:
            extrinsic_parameters = ["luminosity_distance", "psi", "ra", "dec"]
            parameters = [i for i in samples.keys() if i in extrinsic_parameters]
            data = [samples[parameter] for parameter in parameters]
            labels = [latex_labels[parameter] for parameter in parameters]
            _ = interactive.corner(
                data, labels, write_to_html_file=filename
            )
