# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil

import argparse

from pesummary.core.plots import plot
from pesummary.gw.plots import plot as gwplot
from pesummary.utils.array import Array
from subprocess import CalledProcessError

import numpy as np
import matplotlib
from matplotlib import rcParams
import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
rcParams["text.usetex"] = False

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

    @pytest.mark.parametrize("param, samples", [("mass_1",
        Array([10, 20, 30, 40])),])
    def test_autocorrelation_plot(self, param, samples):
        fig = plot._autocorrelation_plot(param, samples)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples", [("mass_1",
        [Array([10, 20, 30, 40]), Array([10, 20, 30, 40])]), ])
    def test_autocorrelation_plot_mcmc(self, param, samples):
        fig = plot._autocorrelation_plot_mcmc(param, samples)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label", [("mass_1",
        Array([10, 20, 30, 40]), r"$m_{1}$"),])
    def test_sample_evolution_plot(self, param, samples, latex_label):
        fig = plot._sample_evolution_plot(param, samples, latex_label)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label", [("mass_1",
        [Array([10, 20, 30, 40]), Array([10, 20, 30, 40])], r"$m_{1}$"), ])
    def test_sample_evolution_plot_mcmc(self, param, samples, latex_label):
        fig = plot._autocorrelation_plot_mcmc(param, samples, latex_label)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label", [("mass_1",
        Array([10, 20, 30, 40]), r"$m_{1}$"),])
    def test_1d_cdf_plot(self, param, samples, latex_label):
        fig = plot._1d_cdf_plot(param, samples, latex_label)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label", [("mass_1",
        [Array([10, 20, 30, 40]), Array([10, 20, 30, 40])], r"$m_{1}$"), ])
    def test_1d_cdf_plot_mcmc(self, param, samples, latex_label):
        fig = plot._1d_cdf_plot_mcmc(param, samples, latex_label)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, colors, latex_label, labels",
        [("mass1", [[10,20,30,40], [1,2,3,4]],
        ["b", "r"], r"$m_{1}$", "approx1"),])
    def test_1d_cdf_comparison_plot(self, param, samples, colors,
                                    latex_label, labels):
        fig = plot._1d_cdf_comparison_plot(param, samples, colors,
                                           latex_label, labels)
        assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label", [("mass1",
        Array([10,20,30,40]), r"$m_{1}$"),])
    def test_1d_histogram_plot(self, param, samples, latex_label):
        for module in [plot, gwplot]:
            fig = getattr(module, "_1d_histogram_plot")(param, samples, latex_label)
            assert isinstance(fig, matplotlib.figure.Figure) == True
            fig = getattr(module, "_1d_histogram_plot")(param, samples, latex_label, kde=True)
            assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, latex_label",
        [("mass1", [[10,20,30,40], [1,2,3,4]], r"$m_{1}$"),])
    def test_1d_histogram_plot_mcmc(self, param, samples, latex_label):
        for module in [plot, gwplot]:
            fig = getattr(module, "_1d_histogram_plot_mcmc")(param, samples, latex_label)
            assert isinstance(fig, matplotlib.figure.Figure) == True
            fig = getattr(module, "_1d_histogram_plot_mcmc")(param, samples, latex_label)
            assert isinstance(fig, matplotlib.figure.Figure) == True

    @pytest.mark.parametrize("param, samples, colors, latex_label, labels",
        [("mass1", [[10,20,30,40], [1,2,3,4]],
        ["b", "r"], r"$m_{1}$", "approx1"),])
    def test_1d_comparison_histogram_plot(self, param, samples, colors,
                                          latex_label, labels):
        for module in [plot, gwplot]:
            fig = getattr(module, "_1d_comparison_histogram_plot")(
                param, samples, colors, latex_label, labels
            )
            assert isinstance(fig, matplotlib.figure.Figure) == True
            fig = getattr(module, "_1d_comparison_histogram_plot")(
                param, samples, colors, latex_label, labels, kde=True
            )
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
        fig, included_params, data = gwplot._make_corner_plot(samples, latex_labels) 
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

    def test_comparison_corner_plot(self):
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
        _samples = {
            i: samples[num] for num, i in enumerate(params)}
        _samples = {"one": _samples, "two": _samples}
        fig = gwplot._make_comparison_corner_plot(
            _samples, latex_labels, corner_parameters=params
        )
        assert isinstance(fig, matplotlib.figure.Figure) == True
        fig.close()

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

    def test_classification_plot(self):
        classifications = {"BBH": 0.95, "NSBH": 0.05}
        fig = gwplot._classification_plot(classifications)
        assert isinstance(fig, matplotlib.figure.Figure) == True


class TestPopulation(object):
    """Class to test the `pesummary.core.plot.population` module
    """
    def test_scatter_plot(self):
        from pesummary.core.plots.population import scatter_plot

        parameters = ["a", "b"]
        sample_dict = {"one": {"a": 10, "b": 20}, "two": {"a": 15, "b": 5}}
        latex_labels = {"a": "a", "b": "b"}
        fig = scatter_plot(parameters, sample_dict, latex_labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        fig = scatter_plot(
            parameters, sample_dict, latex_labels, xerr=sample_dict,
            yerr=sample_dict
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestDetchar(object):
    """Class to test the `pesummary.gw.plot.detchar` module
    """
    def test_spectrogram(self):
        from gwpy.timeseries.core import TimeSeriesBase
        from pesummary.gw.plots.detchar import spectrogram

        strain = {"H1": TimeSeriesBase(np.random.normal(size=200), x0=0, dx=1)}
        fig = spectrogram(strain)
        assert isinstance(fig["H1"], matplotlib.figure.Figure)

    def test_omegascan(self):
        from gwpy.timeseries.core import TimeSeriesBase
        from pesummary.gw.plots.detchar import omegascan

        strain = {"H1": TimeSeriesBase(np.random.normal(size=200), x0=0, dx=1)}
        fig = omegascan(strain, 0)
        assert isinstance(fig["H1"], matplotlib.figure.Figure)


class TestPublication(object):
    """Class to test the `pesummary.gw.plots.publication` module
    """
    def test_twod_contour_plots(self):
        from pesummary.gw.plots.publication import twod_contour_plots

        parameters = ["a", "b"]
        samples = [np.array([
            np.random.uniform(0., 3000, 1000),
            np.random.uniform(0., 3000, 1000)
        ])]
        labels = ["a", "b"]
        fig = twod_contour_plots(
            parameters, samples, labels, {"a": "a", "b": "b"}
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_violin(self):
        from pesummary.gw.plots.publication import violin_plots
        from pesummary.core.plots.seaborn.violin import split_dataframe

        parameter = "a"
        samples = [
            np.random.uniform(0., 3000, 1000),
            np.random.uniform(0., 3000, 1000)
        ]
        labels = ["a", "b"]
        fig = violin_plots(parameter, samples, labels, {"a": "a", "b": "b"})
        assert isinstance(fig, matplotlib.figure.Figure)
        samples2 = [
            np.random.uniform(0., 3000, 1000),
            np.random.uniform(0., 3000, 1000)
        ]
        split = split_dataframe(samples, samples2, labels)
        fig = violin_plots(
            parameter, split, labels, {"a": "a", "b": "b"},
            cut=0, x="label", y="data", hue="side", split=True
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_spin_distribution_plots(self):
        from pesummary.gw.plots.publication import spin_distribution_plots

        parameters = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        samples = [
            np.random.uniform(0, 1, 1000), np.random.uniform(0, 1, 1000),
            np.random.uniform(-1, 1, 1000), np.random.uniform(-1, 1, 1000)
        ]
        label = "test"
        color = "r"
        fig = spin_distribution_plots(parameters, samples, label, color)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_triangle(self):
        from pesummary.core.plots.publication import triangle_plot
        import numpy as np

        x = [np.random.normal(10, i, 1000) for i in [2, 3]]
        y = [np.random.normal(10, i, 1000) for i in [2, 2.5]]

        fig, _, _, _ = triangle_plot(
            x, y, fill_alpha=0.2, xlabel=r"$x$", ylabel=r"$y$",
            linestyles=["-", "--"], percentiles=[5, 95]
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_reverse_triangle(self):
        from pesummary.core.plots.publication import reverse_triangle_plot
        import numpy as np

        x = [np.random.normal(10, i, 1000) for i in [2, 3]]
        y = [np.random.normal(10, i, 1000) for i in [2, 2.5]]

        fig, _, _, _ = reverse_triangle_plot(
            x, y, fill_alpha=0.2, xlabel=r"$x$", ylabel=r"$y$",
            linestyles=["-", "--"], percentiles=[5, 95]
        )
        assert isinstance(fig, matplotlib.figure.Figure)
