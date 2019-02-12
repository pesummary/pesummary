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

import warnings

import numpy as np
import h5py
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pesummary
from pesummary.plot import plot
from pesummary.utils.utils import logger

from pesummary.inputs import command_line, Input, PostProcessing

__doc__ == "Class to generate plots"

latex_labels={"luminosity_distance": r"$d_{L} [Mpc]$",
              "geocent_time": r"$t_{c} [s]$",
              "dec": r"$\delta [rad]$",
              "ra": r"$\alpha [rad]$",
              "a_1": r"$a_{1}$",
              "a_2": r"$a_{2}$",
              "phi_jl": r"$\phi_{JL} [rad]$",
              "phase": r"$\phi [rad]$",
              "psi": r"$\Psi [rad]$",
              "iota": r"$\iota [rad]$",
              "tilt_1": r"$\theta_{1} [rad]$",
              "tilt_2": r"$\theta_{2} [rad]$",
              "phi_12": r"$\phi_{12} [rad]$",
              "mass_2": r"$m_{2} [M_{\odot}]$",
              "mass_1": r"$m_{1} [M_{\odot}]$",
              "total_mass": r"$M [M_{\odot}]$",
              "chirp_mass": r"$\mathcal{M} [M_{\odot}]$",
              "log_likelihood": r"$\log{\mathcal{L}}$",
              "H1_matched_filter_snr": r"$\rho^{H}_{mf}$",
              "L1_matched_filter_snr": r"$\rho^{L}_{mf}$",
              "H1_optimal_snr": r"$\rho^{H}_{opt}$",
              "L1_optimal_snr": r"$\rho^{L}_{opt}$",
              "V1_optimal_snr": r"$\rho^{V}_{opt}$",
              "E1_optimal_snr": r"$\rho^{E}_{opts}$",
              "spin_1x": r"$S_{1}x$",
              "spin_1y": r"$S_{1}y$",
              "spin_1z": r"$S_{1}z$",
              "spin_2x": r"$S_{2}x$",
              "spin_2y": r"$S_{2}y$",
              "spin_2z": r"$S_{2}z$",
              "chi_p": r"$\chi_{p}$",
              "chi_eff": r"$\chi_{eff}$",
              "mass_ratio": r"$q$",
              "symmetric_mass_ratio": r"$\eta$",
              "phi_1": r"$\phi_{1} [rad]$",
              "phi_2": r"$\phi_{2} [rad]$",
              "cos_tilt_1": r"$\cos{\theta_{1}}$",
              "cos_tilt_2": r"$\cos{\theta_{2}}$",
              "redshift": r"$z$",
              "comoving_distance": r"$d_{com} [Mpc]$",
              "mass_1_source": r"$m_{1}^{source} [M_{\odot}]$",
              "mass_2_source": r"$m_{2}^{source} [M_{\odot}]$",
              "chirp_mass_source": r"$\mathcal{M}^{source} [M_{\odot}]$",
              "total_mass_source": r"$M^{source} [M_{\odot}]$",
              "cos_iota": r"$\cos{\iota}$"}


class PlotGeneration(PostProcessing):
    """Class to generate all available plots for each results file.

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments

    Attributes
    ----------
    savedir: str
        The path to the directory where all plots will be saved
    """
    def __init__(self, inputs, colors="default"):
        super(PlotGeneration, self).__init__(inputs, colors)
        logger.info("Starting to generate plots")
        self.generate_plots()
        logger.info("Finished generating the plots")

    @staticmethod
    def _check_latex_labels(parameters):
        for i in parameters:
            if i not in list(latex_labels.keys()):
                latex_labels[i] = i

    @property
    def label_to_prepend_approximant(self):
        labels = [i[len(self.gracedb)+1:] if self.gracedb else i for i in \
            self.labels]
        prepend = [None]*len(self.approximant)
        duplicates=dict(set((x,self.approximant.count(x)) for x in \
            filter(lambda rec : self.approximant.count(rec)>1,self.approximant)))
        if len(duplicates.keys()) > 0:
            for num, i in enumerate(self.approximant):
                if i in duplicates.keys():
                    prepend[num]  = labels[num]
        return prepend

    @property
    def savedir(self):
        return self.webdir+"/plots/"

    def generate_plots(self):
        """Generate all plots for all results files.
        """
        for num, i in enumerate(self.approximant):
            logger.debug("Starting to generate plots for %s\n" %(i))
            self._check_latex_labels(self.parameters[num])
            self.try_to_make_a_plot("corner", num)
            self.try_to_make_a_plot("skymap", num)
            self.try_to_make_a_plot("waveform", num)
            self.try_to_make_a_plot("1d_histogram", num)
        if self.sensitivity:
            self.try_to_make_a_plot("sensitivity", 0)
        if self.add_to_existing:
            existing_file = h5py.File(self.existing+"/samples/posterior_samples.h5")
            labels = list(existing_file.keys())
            for i in labels:
                approximants = list(existing_file[i].keys())
                parameters = [[i.decode("utf-8") for i in \
                    existing_file["%s/%s/parameter_names" %(i, j)]] for j in \
                    approximants]
                samples = [[i for i in existing_file["%s/%s/samples" %(i, j)]] \
                    for j in approximants]
                existing_file.close()
                proposed_tags = ["%s_%s" %(i, j) for j in approximants]
                existing_tags = ["%s_%s" %(i, j) for i,j in \
                    zip(self.labels, self.approximant)]
                for num, j in enumerate(proposed_tags):
                    if j not in existing_tags:
                        self.approximant.append(approximants[num])
                        self.result_files.append(self.existing+"/samples/posterior_samples.h5")
                        self.samples.append(samples[num])
                        self.parameters.append(parameters[num])
                        self.labels.append(i)
                    else:
                        logger.warning("Data for the approximant %s already "
                            "exists. This approximant is being ignored" %(approximants[num]))
            key_data = self._key_data()
            maxL_list = []
            for idx, j in enumerate(self.parameters):
                dictionary = {k: key_data[idx][k]["maxL"] for k in j}
                dictionary["approximant"] = self.approximant[idx]
                maxL_list.append(dictionary)
            self.maxL_samples = maxL_list
            self.same_parameters = list(set.intersection(*[set(l) for l in \
                self.parameters])) 
        if len(self.samples) > 1:
            logger.debug("Starting to generate comparison plots\n")
            self.try_to_make_a_plot("1d_histogram_comparison", "all")
            self.try_to_make_a_plot("skymap_comparison", "all")
            self.try_to_make_a_plot("waveform_comparison", "all")

    def try_to_make_a_plot(self, plot_type, idx=None):
        """Try and make a plot. If it fails, return an error.

        Parameters
        ----------
        plot_type: str
            String for the plot that you wish to try and make
        idx: int
            The index of the results file that you wish to analyse.
        """
        plot_type_dictionary = {"corner": self._corner_plot,
                                "skymap": self._skymap_plot,
                                "waveform": self._waveform_plot,
                                "1d_histogram": self._1d_histogram_plots,
                                "1d_histogram_comparison": \
                                    self._1d_histogram_comparison_plots,
                                "skymap_comparison": self._skymap_comparison_plot,
                                "waveform_comparison": self._waveform_comparison_plot,
                                "sensitivity": self._sensitivity_plot}
        try:
            plot_type_dictionary[plot_type](idx)
        except Exception as e:
            logger.info("Failed to generate %s plot because "
                        "%s" %(plot_type, e))

    def _corner_plot(self, idx):
        """Generate a corner plot for a given results file.

        Parameters
        ----------
        idx: int
            The index of the results file that you wish to analyse
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, params = plot._make_corner_plot(self.samples[idx],
                self.parameters[idx], latex_labels)
            plt.savefig("%s/corner/%s_%s_all_density_plots.png" %(self.savedir,
                self.labels[idx], self.approximant[idx]))
            plt.close()
            combine_corner = open("%s/js/combine_corner.js" %(self.webdir))
            combine_corner = combine_corner.readlines()
            for linenumber, line in enumerate(combine_corner):
                if "var list = [" in line:
                    combine_corner[linenumber] = "    var list = %s;\n" %(params)
            new_file = open("%s/js/combine_corner.js" %(self.webdir), "w")
            new_file.writelines(combine_corner)
            new_file.close()

    def _skymap_plot(self, idx):
        """Generate a skymap showing the confidence regions for a given results
        file.


        Parameters
        ----------
        idx: int
            The index of the results file that you wish to analyse
        """
        ind_ra = self.parameters[idx].index("ra")
        ind_dec = self.parameters[idx].index("dec")
        ra = [j[ind_ra] for j in self.samples[idx]]
        dec = [j[ind_dec] for j in self.samples[idx]]
        fig = plot._sky_map_plot(ra, dec)
        plt.savefig(self.savedir+"/%s_%s_skymap.png" %(self.labels[idx],
            self.approximant[idx]))
        plt.close()

    def _sensitivity_plot(self, idx):
        """Generate a plot showing the network sensitivity for the maximum
        likelihood waveform.

        Parameters
        ----------
        idx: int
            The index of the results file that you wish to analyse
        """
        fig = plot._sky_sensitivity(["H1", "L1"], 0.2, self.maxL_samples[idx])
        plt.savefig(self.savedir+"%s_sky_sensitivity_HL" %(
            self.approximant[idx]))
        plt.close()
        fig = plot._sky_sensitivity(["H1", "L1", "V1"], 0.2, self.maxL_samples[idx])
        plt.savefig(self.savedir+"%s_sky_sensitivity_HLV" %(
            self.approximant[idx]))
        plt.close()

    def _waveform_plot(self, idx):
        """Generate a plot showing the maximum likelihood waveform in each
        available detector.

        Parameters
        ----------
        idx: int
            The index of the results file that you wish to analyse
        """
        detectors = self.detectors[idx].split("_")
        fig = plot._waveform_plot(detectors, self.maxL_samples[idx])
        plt.savefig(self.savedir+"%s_%s_waveform.png" %(self.labels[idx],
            self.approximant[idx]))
        plt.close()

    def _1d_histogram_plots(self, idx):
        """Generate 1d_histogram plots, sample evolution plots and plots
        showing the autocorrelation function for all parameters in the
        results file.

        Parameters
        ----------
        idx: int
            The index of the results file that you wish to analyse
        """
        for ind, j in enumerate(self.parameters[idx]):
            index = self.parameters[idx].index("%s" %(j))
            inj_value = self.injection_data[idx]["%s" %(j)]
            if math.isnan(inj_value):
                inj_value = None 
            param_samples = [k[index] for k in self.samples[idx]]
            fig = plot._1d_histogram_plot(j, param_samples, latex_labels[j],
                inj_value)
            plt.savefig(self.savedir+"%s_1d_posterior_%s_%s.png" %(
                self.labels[idx], self.approximant[idx], j))
            plt.close()
            fig = plot._sample_evolution_plot(j, param_samples,
                latex_labels[j], inj_value)
            plt.savefig(self.savedir+"%s_sample_evolution_%s_%s.png" %(
                self.labels[idx], self.approximant[idx], j))
            plt.close()
            fig = plot._autocorrelation_plot(j, param_samples)
            plt.savefig(self.savedir+"%s_autocorrelation_%s_%s.png" %(
                self.labels[idx], self.approximant[idx], j))
            plt.close()

    def _1d_histogram_comparison_plots(self, idx="all"):
        """Generate comparison plots for all parameters that are consistent
        across all results files.

        Parameters
        ----------
        idx: int, optional
            The indicies of the results files that you wish to be included
            in the comparsion plots.
        """
        for ind, j in enumerate(self.same_parameters):
            indices = [k.index("%s" %(j)) for k in self.parameters]
            param_samples = [[k[indices[num]] for k in l] for num, l in \
                enumerate(self.samples)]
            fig = plot._1d_comparison_histogram_plot(j, self.approximant,
                param_samples, self.colors, latex_labels[j],
                approximant_labels = self.label_to_prepend_approximant)
            plt.savefig(self.savedir+"combined_posterior_%s" %(j))
            plt.close()

    def _skymap_comparison_plot(self, idx="all"):
        """Generate a comparison skymap plot.

        Parameters
        ----------
        idx: int, optional
            The indicies of the results files that you wish to be included
            in the comparsion plots.
        """
        ind_ra = [i.index("ra") for i in self.parameters]
        ind_dec = [i.index("dec") for i in self.parameters]
        ra_list = [[k[ind_ra[num]] for k in l] for num,l in \
            enumerate(self.samples)]
        dec_list = [[k[ind_dec[num]] for k in l] for num,l in \
            enumerate(self.samples)]
        fig = plot._sky_map_comparison_plot(ra_list, dec_list, self.approximant,
            self.colors, approximant_labels = self.label_to_prepend_approximant)
        plt.savefig(self.savedir+"combined_skymap.png")
        plt.close()

    def _waveform_comparison_plot(self, idx="all"):
        """Generate a plot to compare waveforms as seen in the Hanford
        detector.

        Parameters
        ----------
        idx: int, optional
            The indicies of the results files that you wish to be included
            in the comparsion plots.
        """
        fig = plot._waveform_comparison_plot(self.maxL_samples, self.colors,
            approximant_labels = self.label_to_prepend_approximant)
        plt.savefig(self.savedir+"compare_waveforms.png")
        plt.close()


if __name__ == '__main__':
    parser = command_line()
    opts = parser.parse_args()
    inputs = Input(opts)
    PlotGeneration(inputs)
