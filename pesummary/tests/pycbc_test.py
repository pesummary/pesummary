# Licensed under an MIT style license -- see LICENSE.md

from pycbc.psd import aLIGOZeroDetHighPower
from pycbc import pnutils
from pesummary.gw.waveform import fd_waveform
from pesummary.gw.pycbc import optimal_snr, compute_the_overlap
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestPyCBCModule(object):
    """Class to test pesummary.gw.pycbc module
    """
    def setup(self):
        """Setup the testing class
        """
        self.n_samples = 20
        self.approx = "IMRPhenomPv2"
        self.mass_1 = np.random.uniform(20, 100, self.n_samples)
        self.mass_2 = np.random.uniform(5, self.mass_1, self.n_samples)
        self.a_1 = np.random.uniform(0, 1, self.n_samples)
        self.a_2 = np.random.uniform(0, 1, self.n_samples)
        self.tilt_1 = np.arccos(np.random.uniform(-1, 1, self.n_samples))
        self.tilt_2 = np.arccos(np.random.uniform(-1, 1, self.n_samples))
        self.phi_12 = np.random.uniform(0, 1, self.n_samples)
        self.theta_jn = np.arccos(np.random.uniform(-1, 1, self.n_samples))
        self.phi_jl = np.random.uniform(0, 1, self.n_samples)
        self.f_low = [20.] * self.n_samples
        self.f_final = [1024.] * self.n_samples
        self.phase = np.random.uniform(0, 1, self.n_samples)
        self.distance = np.random.uniform(100, 500, self.n_samples)
        self.ra = np.random.uniform(0, np.pi, self.n_samples)
        self.dec = np.arccos(np.random.uniform(-1, 1, self.n_samples))
        self.psi_l = np.random.uniform(0, 1, self.n_samples)
        self.time = [10000.] * self.n_samples
        self.spin_1z = self.a_1 * np.cos(self.tilt_1)
        self.spin_2z = self.a_2 * np.cos(self.tilt_2)

    def _make_psd(self, index):
        """Make a PSD for testing

        Parameters
        ----------
        index: int
            index of mass_1/mass_2/... array to use when computing the PSD
        """
        duration = pnutils.get_imr_duration(
            self.mass_1[index], self.mass_2[index], self.spin_1z[index],
            self.spin_2z[index], self.f_low[index], "IMRPhenomD"
        )
        t_len = 2**np.ceil(np.log2(duration) + 1)
        df = 1./t_len
        flen = int(self.f_final[index] / df) + 1
        aLIGOpsd = aLIGOZeroDetHighPower(flen, df, self.f_low[index])
        psd = aLIGOpsd
        return psd, df

    def _make_waveform(self, index, df):
        """Make a waveform for testing

        Parameters
        ----------
        index: int
            index of mass_1/mass_2/... array to use for waveform generation
        df: float
            difference in frequency samples to use when generating waveform.
            This should match the difference in frequency samples used for the
            PSD
        """
        wvfs = fd_waveform(
            dict(
                theta_jn=self.theta_jn[index],
                phi_jl=self.phi_jl[index], phase=self.phase[index],
                mass_1=self.mass_1[index], mass_2=self.mass_2[index],
                tilt_1=self.tilt_1[index], tilt_2=self.tilt_2[index],
                phi_12=self.phi_12[index], a_1=self.a_1[index],
                a_2=self.a_2[index], luminosity_distance=self.distance[index]
            ), self.approx, df, self.f_low[index], self.f_final[index] / 2,
            f_ref=self.f_low[index], pycbc=True
        )
        hp, hc = wvfs["h_plus"], wvfs["h_cross"]
        return hp, hc

    def test_optimal_snr(self):
        """Test the pesummary.gw.pycbc.optimal_snr function
        """
        from pycbc.filter import sigma

        for i in range(self.n_samples):
            psd, df = self._make_psd(i)
            hp, hc = self._make_waveform(i, df)
            pycbc = sigma(
                hp, psd, low_frequency_cutoff=self.f_low[i],
                high_frequency_cutoff=self.f_final[i] / 2
            )
            np.testing.assert_almost_equal(optimal_snr(
                hp, psd, low_frequency_cutoff=self.f_low[i],
                high_frequency_cutoff=self.f_final[i] / 2
            ), pycbc)

    def test_compute_the_overlap(self):
        """Test the pesummary.gw.pycbc.compute_the_overlap function
        """
        from pycbc.filter import overlap_cplx

        for i in range(self.n_samples):
            psd, df = self._make_psd(i)
            hp, hc = self._make_waveform(i, df)
            np.testing.assert_almost_equal(compute_the_overlap(
                hp, hp, psd, low_frequency_cutoff=self.f_low[i]
            ), 1.0)
            np.testing.assert_almost_equal(compute_the_overlap(
                hc, hc, psd, low_frequency_cutoff=self.f_low[i]
            ), 1.0)
            pycbc = overlap_cplx(
                hp, hc, psd=psd, low_frequency_cutoff=self.f_low[i],
                normalized=True
            )
            np.testing.assert_almost_equal(compute_the_overlap(
                hp, hc, psd, low_frequency_cutoff=self.f_low[i]
            ), pycbc)
