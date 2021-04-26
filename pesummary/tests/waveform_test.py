# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.gw.waveform import fd_waveform, td_waveform
from pycbc.waveform import get_fd_waveform, get_td_waveform
from lal import MSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
np.random.seed(1234)


class TestWaveformModule():
    """Class to test pesummary.gw.waveform module
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

    def _make_waveform(self, index, df, func=fd_waveform, approx=None, **kwargs):
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
        if approx is None:
            approx = self.approx
        base = dict(
            theta_jn=self.theta_jn[index],
            phi_jl=self.phi_jl[index], phase=self.phase[index],
            mass_1=self.mass_1[index], mass_2=self.mass_2[index],
            tilt_1=self.tilt_1[index], tilt_2=self.tilt_2[index],
            phi_12=self.phi_12[index], a_1=self.a_1[index],
            a_2=self.a_2[index], luminosity_distance=self.distance[index]
        )
        try:
            wvfs = func(
                base, approx, df, self.f_low[index], self.f_final[index] / 2,
                f_ref=self.f_low[index], pycbc=True, **kwargs
            )
        except Exception:
            wvfs = func(
                base, approx, df, self.f_low[index], f_ref=self.f_low[index],
                pycbc=True, **kwargs
            )
        hp, hc = wvfs["h_plus"], wvfs["h_cross"]
        return hp, hc

    def test_fd_waveform(self):
        """Test the pesummary.gw.waveform.fd_waveform function
        """
        df = 1./128
        for i in range(self.n_samples):
            iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                SimInspiralTransformPrecessingNewInitialConditions(
                    self.theta_jn[i], self.phi_jl[i], self.tilt_1[i],
                    self.tilt_2[i], self.phi_12[i], self.a_1[i],
                    self.a_2[i], self.mass_1[i] * MSUN_SI,
                    self.mass_2[i] * MSUN_SI, self.f_low[i],
                    self.phase[i]
                )
            pycbc_hp, pycbc_hc = get_fd_waveform(
                approximant=self.approx, mass1=self.mass_1[i],
                mass2=self.mass_2[i], spin1x=spin1x,
                spin1y=spin1y, spin1z=spin1z, spin2x=spin2x,
                spin2y=spin2y, spin2z=spin2z,
                inclination=iota, distance=self.distance[i],
                coa_phase=self.phase[i], f_lower=self.f_low[i],
                f_final=self.f_final[i] / 2, delta_f=df,
                f_ref=self.f_low[i]
            )
            pesummary_hp, pesummary_hc = self._make_waveform(i, df)
            np.testing.assert_almost_equal(
                10**30 * np.array(pycbc_hp), 10**30 * np.array(pesummary_hp)
            )
            np.testing.assert_almost_equal(
                np.array(pycbc_hp.sample_frequencies),
                np.array(pesummary_hp.sample_frequencies)
            )

    def test_td_waveform(self):
        """Test the pesummary.gw.waveform.td_waveform function
        """
        dt = 1./1024
        for i in range(5):
            iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                SimInspiralTransformPrecessingNewInitialConditions(
                    self.theta_jn[i], self.phi_jl[i], self.tilt_1[i],
                    self.tilt_2[i], self.phi_12[i], self.a_1[i],
                    self.a_2[i], self.mass_1[i] * MSUN_SI,
                    self.mass_2[i] * MSUN_SI, self.f_low[i],
                    self.phase[i]
                )
            pycbc_hp, pycbc_hc = get_td_waveform(
                approximant="SEOBNRv3", mass1=self.mass_1[i],
                mass2=self.mass_2[i], spin1x=spin1x,
                spin1y=spin1y, spin1z=spin1z, spin2x=spin2x,
                spin2y=spin2y, spin2z=spin2z,
                inclination=iota, distance=self.distance[i],
                coa_phase=self.phase[i], f_lower=self.f_low[i],
                delta_t=dt, f_ref=self.f_low[i]
            )
            pesummary_hp, pesummary_hc = self._make_waveform(
                i, dt, func=td_waveform, approx="SEOBNRv3"
            )
            np.testing.assert_almost_equal(
                10**30 * np.array(pycbc_hp), 10**30 * np.array(pesummary_hp)
            )
            np.testing.assert_almost_equal(
                np.array(pycbc_hp.sample_times),
                np.array(pesummary_hp.sample_times)
            )

    def test_mode_array(self):
        """Test that the mode array option works for both td and fd waveforms
        """
        df = 1./256
        iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
            SimInspiralTransformPrecessingNewInitialConditions(
                self.theta_jn[0], self.phi_jl[0], self.tilt_1[0],
                self.tilt_2[0], self.phi_12[0], self.a_1[0],
                self.a_2[0], self.mass_1[0] * MSUN_SI,
                self.mass_2[0] * MSUN_SI, self.f_low[0],
                self.phase[0]
            )
        pycbc_hp, pycbc_hc = get_fd_waveform(
            approximant="IMRPhenomPv3HM", mass1=self.mass_1[0],
            mass2=self.mass_2[0], spin1x=spin1x,
            spin1y=spin1y, spin1z=spin1z, spin2x=spin2x,
            spin2y=spin2y, spin2z=spin2z,
            inclination=iota, distance=self.distance[0],
            coa_phase=self.phase[0], f_lower=self.f_low[0],
            f_final=self.f_final[0] / 2, delta_f=df,
            f_ref=self.f_low[0], mode_array=[[2,2], [3,3]]
        )
        pesummary_hp, pesummary_hc = self._make_waveform(
            0, df, approx="IMRPhenomPv3HM", mode_array=[[2,2], [3,3]]
        )
        np.testing.assert_almost_equal(
            10**30 * np.array(pycbc_hp), 10**30 * np.array(pesummary_hp)
        )
        np.testing.assert_almost_equal(
            np.array(pycbc_hp.sample_frequencies),
            np.array(pesummary_hp.sample_frequencies)
        )
        dt = 1./2048
        pycbc_hp, pycbc_hc = get_td_waveform(
            approximant="SEOBNRv4PHM", mass1=self.mass_1[0],
            mass2=self.mass_2[0], spin1x=spin1x,
            spin1y=spin1y, spin1z=spin1z, spin2x=spin2x,
            spin2y=spin2y, spin2z=spin2z,
            inclination=iota, distance=self.distance[0],
            coa_phase=self.phase[0], f_lower=self.f_low[0],
            delta_t=dt, f_ref=self.f_low[0], mode_array=[[2,2], [3,3]]
        )
        pesummary_hp, pesummary_hc = self._make_waveform(
            0, dt, approx="SEOBNRv4PHM", mode_array=[[2,2], [3,3]],
            func=td_waveform
        )
        np.testing.assert_almost_equal(
            10**30 * np.array(pycbc_hp), 10**30 * np.array(pesummary_hp)
        )
        np.testing.assert_almost_equal(
            np.array(pycbc_hp.sample_times),
            np.array(pesummary_hp.sample_times)
        )
