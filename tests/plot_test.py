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
import shutil

import argparse

from pesummary.core.plots import plot
from pesummary.gw.plots import plot as gwplot
from pesummary.utils.utils import Array

import numpy as np
import matplotlib

import pytest

class TestPlot(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
            shutil.rmtree("./.outdir")
        os.makedirs("./.outdir")

    def _grab_frequencies_from_psd_data_file(self, file):
        """Return the frequencies stored in the psd data files

        Parameters
        ----------
        file: str
            path to the psd data file
        """
        fil = open(file)
        fil = fil.readlines()
        fil = [i.strip().split() for i in fil]
        return [float(i[0]) for i in fil]

    def _grab_strains_from_psd_data_file(sef, file):
        """Return the strains stored in the psd data files

        Parameters
        ----------
        file: str
            path to the psd data file
        """
        fil = open(file)
        fil = fil.readlines()
        fil = [i.strip().split() for i in fil]
        return [float(i[1]) for i in fil]

    @pytest.mark.parametrize("param, samples, latex_label", [("mass1",
        Array([10,20,30,40]), r"$m_{1}$"),])
    def test_1d_histogram_plot(self, param, samples, latex_label):
        fig = plot._1d_histogram_plot(param, samples, latex_label)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, colors, latex_label, labels",
        [("mass1", [[10,20,30,40], [1,2,3,4]],
        ["b", "r"], r"$m_{1}$", "approx1"),])
    def test_1d_comparison_plot(self, param, samples, colors,
                                latex_label, labels):
        fig = plot._1d_comparison_histogram_plot(param, samples, colors,
                                                 latex_label, labels)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, colors, latex_label, labels",
        [("mass1", [[10,20,30,40], [1,2,3,4]],
        ["b", "r"], r"$m_{1}$", ["approx1", "approx2"]),])
    def test_comparison_box_plot(self, param, samples, colors,
                                 latex_label, labels):
        fig = plot._comparison_box_plot(param, samples, colors, latex_label,
                                        labels)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_waveform_plot(self):
        maxL_params = {"approximant": "IMRPhenomPv2", "mass_1": 10., "mass_2": 5.,
                       "theta_jn": 1., "phi_jl": 0., "tilt_1": 0., "tilt_2": 0.,
                       "phi_12": 0., "a_1": 0.5, "a_2": 0., "phase": 0.,
                       "ra": 1., "dec": 1., "psi": 0., "geocent_time": 0.,
                       "luminosity_distance": 100}
        fig = gwplot._waveform_plot(["H1"], maxL_params)
        assert isinstance(fig, matplotlib.figure.Figure) == True
    
    def test_timedomain_waveform_plot(self):
        maxL_params = {"approximant": "IMRPhenomPv2", "mass_1": 10., "mass_2": 5.,
                       "theta_jn": 1., "phi_jl": 0., "tilt_1": 0., "tilt_2": 0.,
                       "phi_12": 0., "a_1": 0.5, "a_2": 0., "phase": 0.,
                       "ra": 1., "dec": 1., "psi": 0., "geocent_time": 0.,
                       "luminosity_distance": 100}
        fig = gwplot._time_domain_waveform(["H1"], maxL_params)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_waveform_comparison_plot(self):
        maxL_params = {"approximant": "IMRPhenomPv2", "mass_1": 10., "mass_2": 5.,
                       "theta_jn": 1., "phi_jl": 0., "tilt_1": 0., "tilt_2": 0.,
                       "phi_12": 0., "a_1": 0.5, "a_2": 0., "phase": 0.,
                       "ra": 1., "dec": 1., "psi": 0., "geocent_time": 0.,
                       "luminosity_distance": 100}
        maxL_params = [maxL_params, maxL_params]
        maxL_params[1]["mass_1"] = 7.
        fig = gwplot._waveform_comparison_plot(maxL_params, ["b", "r"],
                                               ["IMRPhenomPv2"]*2)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_time_domain_waveform_comparison_plot(self):
        maxL_params = {"approximant": "IMRPhenomPv2", "mass_1": 10., "mass_2": 5.,
                       "theta_jn": 1., "phi_jl": 0., "tilt_1": 0., "tilt_2": 0.,
                       "phi_12": 0., "a_1": 0.5, "a_2": 0., "phase": 0.,
                       "ra": 1., "dec": 1., "psi": 0., "geocent_time": 0.,
                       "luminosity_distance": 100}
        maxL_params = [maxL_params, maxL_params]
        maxL_params[1]["mass_1"] = 7.
        fig = gwplot._time_domain_waveform_comparison_plot(maxL_params, ["b", "r"],
                                                           ["IMRPhenomPv2"]*2)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("ra, dec", [([1,2,3,4], [1,1,1,1]),])
    def test_sky_map_plot(self, ra, dec):
        fig = gwplot._default_skymap_plot(ra, dec)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("ra, dec, approx, colors", [([[1,2,3,4],[1,2,2,1]],
        [[1,1,2,1],[1,1,1,1]], ["approx1", "approx2"], ["b", "r"]),])
    def test_sky_map_comparison_plot(self, ra, dec, approx, colors):
        fig = gwplot._sky_map_comparison_plot(ra, dec, approx, colors)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_corner_plot(self):
        latex_labels = {"luminosity_distance": r"$d_{L}$",
                        "dec": r"$\delta$",
                        "a_2": r"$a_{2}$", "a_1": r"$a_{1}$",
                        "geocent_time": r"$t$", "phi_jl": r"$\phi_{JL}$",
                        "psi": r"$\Psi$", "ra": r"$\alpha$", "phase": r"$\psi$",
                        "mass_2": r"$m_{2}$", "mass_1": r"$m_{1}$",
                        "phi_12": r"$\phi_{12}$", "tilt_2": r"$t_{1}$",
                        "iota": r"$\iota$", "tilt_1": r"$t_{1}$",
                        "chi_p": r"$\chi_{p}$", "chirp_mass": r"$\mathcal{M}$",
                        "mass_ratio": r"$q$", "symmetric_mass_ratio": r"$\eta$",
                        "total_mass": r"$M$", "chi_eff": r"$\chi_{eff}$"}
        samples = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]*21
        samples = [np.random.random(21).tolist() for i in range(21)]
        params = list(latex_labels.keys())
        samples = {
            i: samples[num] for num, i in enumerate(params)}
        fig, included_params = gwplot._make_corner_plot(samples, latex_labels) 
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_source_corner_plot(self):
        latex_labels = {"luminosity_distance": r"$d_{L}$",
                        "dec": r"$\delta$",
                        "a_2": r"$a_{2}$", "a_1": r"$a_{1}$",
                        "geocent_time": r"$t$", "phi_jl": r"$\phi_{JL}$",
                        "psi": r"$\Psi$", "ra": r"$\alpha$", "phase": r"$\psi$",
                        "mass_2": r"$m_{2}$", "mass_1": r"$m_{1}$",
                        "phi_12": r"$\phi_{12}$", "tilt_2": r"$t_{1}$",
                        "iota": r"$\iota$", "tilt_1": r"$t_{1}$",
                        "chi_p": r"$\chi_{p}$", "chirp_mass": r"$\mathcal{M}$",
                        "mass_ratio": r"$q$", "symmetric_mass_ratio": r"$\eta$",
                        "total_mass": r"$M$", "chi_eff": r"$\chi_{eff}$"}
        samples = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]*21
        samples = [np.random.random(21).tolist() for i in range(21)]
        params = list(latex_labels.keys())
        samples = {i: j for i, j in zip(params, samples)}
        fig = gwplot._make_source_corner_plot(samples, latex_labels) 
        assert isinstance(fig, matplotlib.figure.Figure) == True
    
    def test_extrinsic_corner_plot(self):
        latex_labels = {"luminosity_distance": r"$d_{L}$",
                        "dec": r"$\delta$",
                        "a_2": r"$a_{2}$", "a_1": r"$a_{1}$",
                        "geocent_time": r"$t$", "phi_jl": r"$\phi_{JL}$",
                        "psi": r"$\Psi$", "ra": r"$\alpha$", "phase": r"$\psi$",
                        "mass_2": r"$m_{2}$", "mass_1": r"$m_{1}$",
                        "phi_12": r"$\phi_{12}$", "tilt_2": r"$t_{1}$",
                        "iota": r"$\iota$", "tilt_1": r"$t_{1}$",
                        "chi_p": r"$\chi_{p}$", "chirp_mass": r"$\mathcal{M}$",
                        "mass_ratio": r"$q$", "symmetric_mass_ratio": r"$\eta$",
                        "total_mass": r"$M$", "chi_eff": r"$\chi_{eff}$"}
        samples = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]*21
        samples = [np.random.random(21).tolist() for i in range(21)]
        params = list(latex_labels.keys())
        samples = {i: j for i, j in zip(params, samples)}
        fig = gwplot._make_extrinsic_corner_plot(samples, latex_labels) 
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_sensitivity_plot(self):
        maxL_params = {"approximant": "IMRPhenomPv2", "mass_1": 10., "mass_2": 5.,
                       "iota": 1., "phi_jl": 0., "tilt_1": 0., "tilt_2": 0.,
                       "phi_12": 0., "a_1": 0.5, "a_2": 0., "phase": 0.,
                       "ra": 1., "dec": 1., "psi": 0., "geocent_time": 0.,
                       "luminosity_distance": 100}
        fig = gwplot._sky_sensitivity(["H1", "L1"], 1.0, maxL_params)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_psd_plot(self):
        with open("./.outdir/psd.dat", "w") as f:
            f.writelines(["0.5 100"])
            f.writelines(["1.0 150"])
            f.writelines(["5.0 200"])
        frequencies = [self._grab_frequencies_from_psd_data_file("./.outdir/psd.dat")]
        strains = [self._grab_frequencies_from_psd_data_file("./.outdir/psd.dat")]
        fig = gwplot._psd_plot(frequencies, strains, labels=["H1"])
        assert isinstance(fig, matplotlib.figure.Figure) == True

    def test_calibration_plot(self):
        frequencies = np.arange(20, 100, 0.2)
        ifos = ["H1"]
        calibration = [[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2000.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ]]
        fig = gwplot._calibration_envelope_plot(frequencies, calibration, ifos)
        assert isinstance(fig, matplotlib.figure.Figure) == True
