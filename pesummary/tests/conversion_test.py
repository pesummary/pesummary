# Licensed under an MIT style license -- see LICENSE.md

import os
import socket
import shutil

import numpy as np
import h5py

import deepdish

from pesummary.gw.conversions import *
from pesummary.gw.conversions.nrutils import *
from pycbc import conversions
import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def conversion_check(
    pesummary_function, pesummary_args, other_function, other_args,
    dp=8
):
    """Check that the conversions made by PESummary are the same as those
    from pycbc
    """
    _pesummary = pesummary_function(*pesummary_args)
    _other = other_function(*other_args)
    assert np.testing.assert_almost_equal(_pesummary, _other, dp) is None


class TestConversions(object):
    @classmethod
    def setup_class(cls):
        class Arguments(object):
            mass1 = 10.
            mass2 = 5.
            mtotal = 30.
            mchirp = 10.
            q = 1. / 4.
            eta = 0.214
            iota = 0.5
            spin1x = 0.75
            spin1y = 0.
            spin1z = 0.5
            spin2x = 0.
            spin2y = 0.
            spin2z = 0.5
            lambda1 = 500.
            lambda2 = 500.
            lambda_tilde = 1000.

            theta_jn = [0.5, 0.5]
            phi_jl = [0.3, 0.3]
            tilt_1 = [0.5, 0.]
            tilt_2 = [0., 0.]
            phi_12 = [0., 0.]
            a_1 = [0.5, 0.5]
            a_2 = [0., 0.]
            f_ref = [20., 20.]
            phase = [0., 0.]

            redshift = 0.5
            l_distance = 500.

        cls.opts = Arguments()

    def test_z_from_dL(self):
        from bilby.gw.conversion import luminosity_distance_to_redshift

        l_distance = np.random.randint(100, 5000, 20)
        bilby_function = luminosity_distance_to_redshift
        pesummary_function = z_from_dL_exact
        conversion_check(
            pesummary_function, [l_distance], bilby_function, [l_distance]
        )
        pesummary_function = z_from_dL_approx
        conversion_check(
            pesummary_function, [l_distance], bilby_function, [l_distance],
            dp=4
        )

    def test_change_of_cosmology_for_z_from_dL(self):
        from lalinference.bayespputils import calculate_redshift

        l_distance = np.random.randint(100, 5000, 20)
        lal_redshift = calculate_redshift(
            np.atleast_2d(l_distance).T
        ).T[0]
        redshift = z_from_dL_exact(
            l_distance, cosmology="Planck15_lal"
        )
        np.testing.assert_almost_equal(lal_redshift, redshift, 8)

    def test_dL_from_z(self):
        from bilby.gw.conversion import redshift_to_luminosity_distance

        redshift = np.random.randint(1, 5, 100)
        bilby_function = redshift_to_luminosity_distance
        pesummary_function = dL_from_z
        conversion_check(
            pesummary_function, [redshift], bilby_function, [redshift]
        )

    def test_comoving_distance_from_z(self):
        from bilby.gw.conversion import redshift_to_comoving_distance

        redshift = np.random.randint(1, 5, 100)
        bilby_function = redshift_to_comoving_distance
        pesummary_function = comoving_distance_from_z
        conversion_check(
            pesummary_function, [redshift], bilby_function, [redshift]
        )

    def test_m1_source_from_m1_z(self):
        from bilby.gw.conversion import generate_source_frame_parameters

        mass_1 = np.random.randint(5, 100, 100)
        mass_2 = np.random.randint(2, mass_1, 100)
        luminosity_distance = np.random.randint(100, 500, 100)
        sample = generate_source_frame_parameters(
            {"mass_1": mass_1, "mass_2": mass_2,
             "luminosity_distance": luminosity_distance}
        )
        source_frame = generate_source_frame_parameters(sample)
        assert np.testing.assert_almost_equal(
            m1_source_from_m1_z(mass_1, sample["redshift"]), sample["mass_1_source"],
            8
        ) is None

    def test_m1_m2_from_m1_source_m2_source_z(self):
        from bilby.gw.conversion import generate_source_frame_parameters

        mass_1_source = np.random.randint(5, 100, 100)
        mass_2_source = np.random.randint(2, mass_1_source, 100)
        redshift = np.random.randint(1, 10, 100)
        luminosity_distance = dL_from_z(redshift)

        # calculate mass_1 and mass_2 using pesummary
        mass_1 = m1_from_m1_source_z(mass_1_source, redshift)
        mass_2 = m2_from_m2_source_z(mass_2_source, redshift)
        # use calculated mass_1/mass_2 to calculate mass_1_source/mass_2_source using
        # bilby
        sample = generate_source_frame_parameters(
            {"mass_1": mass_1, "mass_2": mass_2,
             "luminosity_distance": luminosity_distance}
        )
        source_frame = generate_source_frame_parameters(sample)
        # confirm that bilby's mass_1_source/mass_2_source is the same as
        # mass_1_source/mass_2_source that pesummary used
        np.testing.assert_almost_equal(sample["mass_1_source"], mass_1_source, 6)
        np.testing.assert_almost_equal(sample["mass_2_source"], mass_2_source, 6)

    def test_m2_source_from_m2_z(self):
        from bilby.gw.conversion import generate_source_frame_parameters

        mass_1 = np.random.randint(5, 100, 100)
        mass_2 = np.random.randint(2, mass_1, 100)
        luminosity_distance = np.random.randint(100, 500, 100)
        sample = generate_source_frame_parameters(
            {"mass_1": mass_1, "mass_2": mass_2,
             "luminosity_distance": luminosity_distance}
        )
        source_frame = generate_source_frame_parameters(sample)
        assert np.testing.assert_almost_equal(
            m2_source_from_m2_z(mass_1, sample["redshift"]), sample["mass_1_source"],
            8
        ) is None

    def test_m_total_source_from_mtotal_z(self):
        from bilby.gw.conversion import generate_source_frame_parameters

        total_mass = np.random.randint(5, 100, 100)
        luminosity_distance = np.random.randint(100, 500, 100)
        sample = generate_source_frame_parameters(
            {"total_mass": total_mass, "luminosity_distance": luminosity_distance}
        )
        source_frame = generate_source_frame_parameters(sample)
        assert np.testing.assert_almost_equal(
            m_total_source_from_mtotal_z(total_mass, sample["redshift"]),
            sample["total_mass_source"], 8
        ) is None

    def test_m_total_from_mtotal_source_z(self):
        total_mass_source, redshift = 20., self.opts.redshift
        m_total = mtotal_from_mtotal_source_z(total_mass_source, redshift)
        assert np.round(m_total, 4) == self.opts.mtotal

    def test_mchirp_source_from_mchirp_z(self):
        from bilby.gw.conversion import generate_source_frame_parameters

        chirp_mass = np.random.randint(5, 100, 100)
        luminosity_distance = np.random.randint(100, 500, 100)
        sample = generate_source_frame_parameters(
            {"chirp_mass": chirp_mass,
             "luminosity_distance": luminosity_distance}
        )
        source_frame = generate_source_frame_parameters(sample)
        assert np.testing.assert_almost_equal(
            mchirp_source_from_mchirp_z(chirp_mass, sample["redshift"]),
            sample["chirp_mass_source"],
            8
        ) is None

    def test_mchirp_from_mchirp_source_z(self):
        mchirp_source, redshift = 20./3., self.opts.redshift
        mchirp = mchirp_from_mchirp_source_z(mchirp_source, redshift)
        assert np.round(mchirp, 4) == self.opts.mchirp

    def test_mchirp_from_m1_m2(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        pycbc_function = conversions.mchirp_from_mass1_mass2
        pesummary_function = mchirp_from_m1_m2
        conversion_check(
            pesummary_function, [mass1, mass2], pycbc_function, [mass1, mass2]
        )

    def test_m_total_from_m1_m2(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        pycbc_function = conversions.mtotal_from_mass1_mass2
        pesummary_function = m_total_from_m1_m2
        conversion_check(
            pesummary_function, [mass1, mass2], pycbc_function, [mass1, mass2]
        )

    def test_m1_from_mchirp_q(self):
        mchirp = np.random.randint(5, 100, 100)
        q = np.random.random(100)
        mchirp, q = self.opts.mchirp, self.opts.q
        pycbc_function = conversions.mass1_from_mchirp_q
        pesummary_function = m1_from_mchirp_q
        conversion_check(
            pesummary_function, [mchirp, q], pycbc_function, [mchirp, 1./q]
        )

    def test_m2_from_mchirp_q(self):
        mchirp = np.random.randint(5, 100, 100)
        q = np.random.random(100)
        pycbc_function = conversions.mass2_from_mchirp_q
        pesummary_function = m2_from_mchirp_q
        conversion_check(
            pesummary_function, [mchirp, q], pycbc_function, [mchirp, 1./q]
        )

    def test_m1_from_mtotal_q(self):
        mtotal = np.random.randint(5, 100, 100)
        q = np.random.random(100)
        pycbc_function = conversions.mass1_from_mtotal_q
        pesummary_function = m1_from_mtotal_q
        conversion_check(
            pesummary_function, [mtotal, q], pycbc_function, [mtotal, 1./q]
        )

    def test_m2_from_mtotal_q(self):
        mtotal = np.random.randint(5, 100, 100)
        q = np.random.random(100)
        pycbc_function = conversions.mass2_from_mtotal_q
        pesummary_function = m2_from_mtotal_q
        conversion_check(
            pesummary_function, [mtotal, q], pycbc_function, [mtotal, 1./q]
        )

    def test_eta_from_m1_m2(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        pycbc_function = conversions.eta_from_mass1_mass2
        pesummary_function = eta_from_m1_m2
        conversion_check(
            pesummary_function, [mass1, mass2], pycbc_function, [mass1, mass2]
        )

    def test_q_from_m1_m2(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        pycbc_function = conversions.invq_from_mass1_mass2
        pesummary_function = q_from_m1_m2
        conversion_check(
            pesummary_function, [mass1, mass2], pycbc_function, [mass1, mass2]
        )

    def test_q_from_eta(self):
        from bilby.gw.conversion import symmetric_mass_ratio_to_mass_ratio

        eta = np.random.uniform(0, 0.25, 100)
        bilby_function = symmetric_mass_ratio_to_mass_ratio
        pesummary_function = q_from_eta
        conversion_check(
            pesummary_function, [eta], bilby_function, [eta]
        )

    def test_mchirp_from_mtotal_q(self):
        mtotal, q = self.opts.mtotal, self.opts.q
        mchirp = mchirp_from_mtotal_q(mtotal, q)
        assert np.round(mchirp, 4) == 9.9906

    def test_chi_p(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        spin1_mag = np.random.random(100)
        spin1_ang = np.random.random(100)
        spin2_mag = np.random.random(100)
        spin2_ang = np.random.random(100)
        spin1x = spin1_mag * np.cos(spin1_ang)
        spin1y = spin1_mag * np.sin(spin1_ang)
        spin2x = spin2_mag * np.cos(spin2_ang)
        spin2y = spin2_mag * np.sin(spin2_ang)
        pycbc_function = conversions.chi_p
        pesummary_function = chi_p
        conversion_check(
            pesummary_function, [mass1, mass2, spin1x, spin1y, spin2x, spin2y],
            pycbc_function, [mass1, mass2, spin1x, spin1y, spin2x, spin2y]
        )

        from lalsimulation import SimPhenomUtilsChiP

        mass1, mass2 = self.opts.mass1, self.opts.mass2
        spin1x, spin1y = self.opts.spin1x, self.opts.spin1y
        spin1z, spin2x = self.opts.spin1z, self.opts.spin2x
        spin2y, spin2z = self.opts.spin2y, self.opts.spin2z
        chi_p_value = chi_p(mass1, mass2, spin1x, spin1y, spin2y, spin2y)
        assert chi_p_value == 0.75
        for i in range(100):
            mass_1 = np.random.randint(10, 100)
            mass_2 = np.random.randint(5, mass_1)
            spin1 = np.random.random(3)
            norm = np.sqrt(np.sum(np.square(spin1)))
            spin1 /= norm
            spin2 = np.random.random(3)
            norm = np.sqrt(np.sum(np.square(spin2)))
            spin2 /= norm
            chi_p_value = chi_p(
                mass_1, mass_2, spin1[0], spin1[1], spin2[0], spin2[1]
            )
            lal_value = SimPhenomUtilsChiP(
                mass_1, mass_2, spin1[0], spin1[1], spin2[0], spin2[1]
            )
            assert np.testing.assert_almost_equal(chi_p_value, lal_value, 9) is None

    def test_chi_eff(self):
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        spin1z = np.random.uniform(-1, 1, 100)
        spin2z = np.random.uniform(-1, 1, 100)
        pycbc_function = conversions.chi_eff
        pesummary_function = chi_eff
        conversion_check(
            pesummary_function, [mass1, mass2, spin1z, spin2z], pycbc_function,
            [mass1, mass2, spin1z, spin2z]
        )

    def test_phi_12_from_phi1_phi2(self):
        data = phi_12_from_phi1_phi2(0.2, 0.5)
        assert data == 0.3
        data = phi_12_from_phi1_phi2(0.5, 0.2)
        rounded_data = np.round(data, 2)
        assert rounded_data == 5.98
        data = phi_12_from_phi1_phi2(np.array([0.5, 0.2]), np.array([0.3, 0.7]))
        rounded_data = np.round(data, 2)
        assert all(i == j for i,j in zip(rounded_data, [6.08, 0.5]))

    def test_phi_from_spins(self):
        def cart2sph(x, y):
            return np.fmod(2 * np.pi + np.arctan2(y,x), 2 * np.pi)
        assert phi1_from_spins(0.5, 0.2) == cart2sph(0.5, 0.2)
        assert phi2_from_spins(0.1, 0.5) == cart2sph(0.1, 0.5)

    def test_spin_angles(self):
        from lalsimulation import SimInspiralTransformPrecessingWvf2PE

        mass1 = np.random.uniform(5., 100., 100)
        mass2 = np.random.uniform(2., mass1, 100)
        inc = np.random.uniform(0, np.pi, 100)
        spin1_mag = np.random.random(100)
        spin1_ang = np.random.random(100)
        spin2_mag = np.random.random(100)
        spin2_ang = np.random.random(100)
        spin1x = spin1_mag * np.cos(spin1_ang)
        spin1y = spin1_mag * np.sin(spin1_ang)
        spin2x = spin2_mag * np.cos(spin2_ang)
        spin2y = spin2_mag * np.sin(spin2_ang)
        spin1z = np.random.random(100) - (spin1x**2 + spin1y**2)**0.5
        spin2z = np.random.random(100) - (spin1x**2 + spin1y**2)**0.5
        f_ref = [20.0] * len(mass1)
        phase = [0.4] * len(mass1)
        lalsimulation_function = SimInspiralTransformPrecessingWvf2PE
        pesummary_function = spin_angles
        for ii in np.arange(len(mass1)):
            conversion_check(
                pesummary_function,
                [mass1[ii], mass2[ii], inc[ii], spin1x[ii], spin1y[ii],
                spin1z[ii], spin2x[ii], spin2y[ii], spin2z[ii], f_ref[ii],
                phase[ii]],
                lalsimulation_function,
                [inc[ii], spin1x[ii], spin1y[ii], spin1z[ii], spin2x[ii],
                spin2y[ii], spin2z[ii], mass1[ii], mass2[ii], f_ref[ii],
                phase[ii]]
        )

    def test_component_spins(self):
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        from lal import MSUN_SI

        mass1 = np.random.uniform(5., 100., 100)
        mass2 = np.random.uniform(2., mass1, 100)
        theta_jn = np.random.uniform(0, np.pi, 100)
        phi_jl = np.random.uniform(0, np.pi, 100)
        phi_12 = np.random.uniform(0, np.pi, 100)
        a_1 = np.random.uniform(0, 1, 100)
        a_2 = np.random.uniform(0, 1, 100)
        tilt_1 = np.random.uniform(0, np.pi, 100)
        tilt_2 = np.random.uniform(0, np.pi, 100)
        f_ref = [20.] * len(mass1)
        phase = [0.5] * len(mass2)
        
        bilby_function = bilby_to_lalsimulation_spins
        pesummary_function = component_spins
        for ii in np.arange(len(mass1)):
            conversion_check(
                pesummary_function,
                [theta_jn[ii], phi_jl[ii], tilt_1[ii], tilt_1[ii], phi_12[ii],
                a_1[ii], a_2[ii], mass1[ii], mass2[ii], f_ref[ii], phase[ii]],
                bilby_function,
                [theta_jn[ii], phi_jl[ii], tilt_1[ii], tilt_1[ii], phi_12[ii],
                a_1[ii], a_2[ii], mass1[ii]*MSUN_SI, mass2[ii]*MSUN_SI,
                f_ref[ii], phase[ii]]
        )

    def test_time_in_each_ifo(self):
        from pycbc.detector import Detector
        from lal import TimeDelayFromEarthCenter
        from lalsimulation import DetectorPrefixToLALDetector

        optimal_ra, optimal_dec = -0.2559168059473027, 0.81079526383
        time = time_in_each_ifo("H1", optimal_ra, optimal_dec, 0)
        light_time = 6371*10**3 / (3.0*10**8)
        assert -np.round(light_time, 4) == np.round(time, 4)

        ra = np.random.uniform(-np.pi, np.pi, 100)
        dec = np.random.uniform(0, 2*np.pi, 100)
        time = np.random.uniform(10000, 20000, 100)
        H1_time = time_in_each_ifo("H1", ra, dec, time)
        pycbc_time = time + Detector("H1").time_delay_from_earth_center(
            ra, dec, time
        )
        lal_time = time + np.array([TimeDelayFromEarthCenter(
            DetectorPrefixToLALDetector('H1').location, ra[ii], dec[ii],
            time[ii]) for ii in range(len(ra))
        ])
        difference = np.abs(lal_time - pycbc_time)
        dp = np.floor(np.abs(np.log10(np.max(difference))))
        assert np.testing.assert_almost_equal(H1_time, pycbc_time, dp) is None
        assert np.testing.assert_almost_equal(H1_time, lal_time, dp) is None
        

    def test_lambda_tilde_from_lambda1_lambda2(self):
        lambda1 = np.random.uniform(0, 5000, 100)
        lambda2 = np.random.uniform(0, 5000, 100)
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        pycbc_function = conversions.lambda_tilde
        pesummary_function = lambda_tilde_from_lambda1_lambda2
        conversion_check(
            pesummary_function, [lambda1, lambda2, mass1, mass2],
            pycbc_function, [mass1, mass2, lambda1, lambda2]
        )

    def test_delta_lambda_from_lambda1_lambda2(self):
        from bilby.gw.conversion import lambda_1_lambda_2_to_delta_lambda_tilde

        lambda1 = np.random.uniform(0, 5000, 100)
        lambda2 = np.random.uniform(0, 5000, 100)
        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        conversion_check(
            delta_lambda_from_lambda1_lambda2,
            [lambda1, lambda2, mass1, mass2],
            lambda_1_lambda_2_to_delta_lambda_tilde,
            [lambda1, lambda2, mass1, mass2]
        )

    def test_lambda1_from_lambda_tilde(self):
        from bilby.gw.conversion import lambda_tilde_to_lambda_1_lambda_2

        mass1 = np.random.randint(5, 100, 100)
        mass2 = np.random.randint(2, mass1, 100)
        lambda_tilde = np.random.uniform(-100, 100, 100)
        lambda_1 = lambda_tilde_to_lambda_1_lambda_2(
            lambda_tilde, mass1, mass2
        )[0]
        lambda1 = lambda1_from_lambda_tilde(lambda_tilde, mass1, mass2)
        assert np.testing.assert_almost_equal(lambda1, lambda_1) is None
        #assert np.round(lambda1, 4) == 192.8101

    def test_lambda2_from_lambda1(self):
        from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters

        mass1 = np.random.uniform(5, 50, 100)
        mass2 = np.random.uniform(2, mass1, 100)
        lambda_1 = np.random.uniform(0, 5000, 100)
        sample = {"mass_1": mass1, "mass_2": mass2, "lambda_1": lambda_1}
        convert = convert_to_lal_binary_neutron_star_parameters(sample)[0]
        lambda2 = lambda2_from_lambda1(lambda_1, mass1, mass2)
        diff = np.abs(lambda2 - convert["lambda_2"])
        ind = np.argmax(diff)
        assert np.testing.assert_almost_equal(lambda2, convert["lambda_2"], 5) is None

    def test_network_snr(self):
        snr_H1 = snr_L1 = snr_V1 = np.array([2., 3.])
        assert network_snr([snr_H1[0], snr_L1[0], snr_V1[0]]) == np.sqrt(3) * 2
        print(snr_H1)
        network = network_snr([snr_H1, snr_L1, snr_V1])
        print(network)
        assert network[0] == np.sqrt(3) * 2
        assert network[1] == np.sqrt(3) * 3

    def test_network_matched_filter_snr(self):
        """Samples taken from a lalinference result file
        """
        snr_mf_H1 = 7.950207935574794
        snr_mf_L1 = 19.232672412819483
        snr_mf_V1 = 3.666438738845737
        snr_opt_H1 = 9.668043620320788
        snr_opt_L1 = 19.0826463504282
        snr_opt_V1 = 3.5578582036515236
        network = network_matched_filter_snr(
            [snr_mf_H1, snr_mf_L1, snr_mf_V1],
            [snr_opt_H1, snr_opt_L1, snr_opt_V1]
        )
        np.testing.assert_almost_equal(network, 21.06984787727566)
        network = network_matched_filter_snr(
            [[snr_mf_H1] * 2, [snr_mf_L1] * 2, [snr_mf_V1] * 2],
            [[snr_opt_H1] * 2, [snr_opt_L1] * 2, [snr_opt_V1] * 2]
        )
        np.testing.assert_almost_equal(
            network, [21.06984787727566] * 2
        )
        snr_mf_H1 = 7.950207935574794 - 1.004962343498161 * 1j
        snr_mf_L1 = 19.232672412819483 - 0.4646531569951501 * 1j
        snr_mf_V1 = 3.666438738845737 - 0.08177741915398137 * 1j
        network = network_matched_filter_snr(
            [snr_mf_H1, snr_mf_L1, snr_mf_V1],
            [snr_opt_H1, snr_opt_L1, snr_opt_V1]
        )
        np.testing.assert_almost_equal(network, 21.06984787727566)

    def test_full_conversion(self):
        from pesummary.utils.array import Array
        from pesummary.gw.conversions import convert
        from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
        from pandas import DataFrame

        dictionary = {
            "mass_1": [10.],
            "mass_2": [2.],
            "a_1": [0.2],
            "a_2": [0.2],
            "cos_theta_jn": [0.5],
            "cos_tilt_1": [0.2],
            "cos_tilt_2": [0.2],
            "luminosity_distance": [0.2],
            "geocent_time": [0.2],
            "ra": [0.2],
            "dec": [0.2],
            "psi": [0.2],
            "phase": [0.5],
            "phi_12": [0.7],
            "phi_jl": [0.25],
            "H1_matched_filter_abs_snr": [10.0],
            "H1_matched_filter_snr_angle": [0.],
            "H1_matched_filter_snr": [10.0],
            "H1_optimal_snr": [10.0]
        }
        data = convert(dictionary)
        true_params = [
            'mass_1', 'mass_2', 'a_1', 'a_2', 'cos_theta_jn', 'cos_tilt_1',
            'cos_tilt_2', 'luminosity_distance', 'geocent_time', 'ra', 'dec',
            'psi', 'phase', 'phi_12', 'phi_jl', 'H1_matched_filter_abs_snr',
            'H1_matched_filter_snr_angle', 'H1_matched_filter_snr', 'theta_jn',
            'tilt_1', 'tilt_2', 'mass_ratio', 'inverted_mass_ratio',
            'total_mass', 'chirp_mass', 'symmetric_mass_ratio', 'iota',
            'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z',
            'phi_1', 'phi_2', 'chi_eff', 'chi_p', 'final_spin_non_evolved',
            'peak_luminosity_non_evolved', 'final_mass_non_evolved', 'redshift',
            'comoving_distance', 'mass_1_source', 'mass_2_source',
            'total_mass_source', 'chirp_mass_source',
            'final_mass_source_non_evolved', 'radiated_energy_non_evolved',
            'network_matched_filter_snr', 'cos_iota'
        ]
        assert all(i in data.parameters for i in true_params)
        assert len(data.parameters) == len(set(data.parameters))
        true = {
            'mass_1': Array(dictionary["mass_1"]),
            'mass_2': Array(dictionary["mass_2"]),
            'a_1': Array(dictionary["a_1"]),
            'a_2': Array(dictionary["a_2"]),
            'cos_theta_jn': Array(dictionary["cos_theta_jn"]),
            'cos_tilt_1': Array(dictionary["cos_tilt_1"]),
            'cos_tilt_2': Array(dictionary["cos_tilt_2"]),
            'luminosity_distance': Array(dictionary["luminosity_distance"]),
            'geocent_time': Array(dictionary["geocent_time"]),
            'ra': Array(dictionary["ra"]), 'dec': Array(dictionary["dec"]),
            'psi': Array(dictionary["psi"]), 'phase': Array(dictionary["phase"]),
            'phi_12': Array(dictionary["phi_12"]),
            'phi_jl': Array(dictionary["phi_jl"]),
            'H1_matched_filter_abs_snr': Array(dictionary["H1_matched_filter_abs_snr"]),
            'H1_matched_filter_snr_angle': Array(dictionary["H1_matched_filter_snr_angle"]),
            'H1_matched_filter_snr': Array(dictionary["H1_matched_filter_snr"]),
            'H1_optimal_snr': Array(dictionary["H1_optimal_snr"]),
            'theta_jn': Array(np.arccos(dictionary["cos_theta_jn"])),
            'tilt_1': Array(np.arccos(dictionary["cos_tilt_1"])),
            'tilt_2': Array(np.arccos(dictionary["cos_tilt_2"])),
            'mass_ratio': Array([dictionary["mass_2"][0] / dictionary["mass_1"][0]]),
            'inverted_mass_ratio': Array([dictionary["mass_1"][0] / dictionary["mass_2"][0]]),
            'total_mass': Array([dictionary["mass_1"][0] + dictionary["mass_2"][0]]),
            'chirp_mass': Array([3.67097772]),
            'symmetric_mass_ratio': Array([0.13888889]),
            'iota': Array([1.01719087]), 'spin_1x': Array([-0.18338713]),
            'spin_1y': Array([0.06905911]), 'spin_1z': Array([0.04]),
            'spin_2x': Array([-0.18475131]), 'spin_2y': Array([-0.06532192]),
            'spin_2z': Array([0.04]), 'phi_1': Array([2.78144141]),
            'phi_2': Array([3.48144141]), 'chi_eff': Array([0.04]),
            'chi_p': Array([0.19595918]),
            'final_spin_non_evolved': Array([0.46198316]),
            'peak_luminosity_non_evolved': Array([1.01239394]),
            'final_mass_non_evolved': Array([11.78221674]),
            'redshift': Array([4.5191183e-05]),
            'comoving_distance': Array([0.19999755]),
            'mass_1_source': Array([9.99954811]),
            'mass_2_source': Array([1.99990962]),
            'total_mass_source': Array([11.99945773]),
            'chirp_mass_source': Array([3.67081183]),
            'final_mass_source_non_evolved': Array([11.78168431]),
            'radiated_energy_non_evolved': Array([0.21777342]),
            'network_matched_filter_snr': Array([10.0]),
            'cos_iota': Array([0.52575756])
        }
        for key in true.keys():
            assert np.round(true[key][0], 8) == np.round(data[key][0], 8)
        convert = convert_to_lal_binary_black_hole_parameters(dictionary)[0]
        for key, item in convert.items():
            assert np.testing.assert_almost_equal(
                item, true[key], 8
            ) is None

    def test_remove_parameter(self):
        from pesummary.gw.conversions import convert

        dictionary = {
            "mass_1": np.random.uniform(5, 100, 100),
            "mass_ratio": [0.1] * 100
        }
        dictionary["mass_2"] = np.random.uniform(2, dictionary["mass_1"], 100)
        incorrect_mass_ratio = convert(dictionary)
        data = convert(dictionary, regenerate=["mass_ratio"])
        assert all(i != j for i, j in zip(
            incorrect_mass_ratio["mass_ratio"], data["mass_ratio"]
        ))
        assert np.testing.assert_almost_equal(
            data["mass_ratio"],
            q_from_m1_m2(dictionary["mass_1"], dictionary["mass_2"]), 8
       ) is None


class TestPrecessingSNR(object):
    """Test the precessing_snr conversion
    """
    def setup(self):
        """Setup the testing class
        """
        np.random.seed(1234)
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
        self.beta = opening_angle(
            self.mass_1, self.mass_2, self.phi_jl, self.tilt_1,
            self.tilt_2, self.phi_12, self.a_1, self.a_2,
            [20.] * self.n_samples, self.phase
        )
        self.spin_1z = self.a_1 * np.cos(self.tilt_1)
        self.spin_2z = self.a_2 * np.cos(self.tilt_2)
        self.psi_J = psi_J(self.psi_l, self.theta_jn, self.phi_jl, self.beta)

    def test_harmonic_overlap(self):
        """Test that the sum of 5 precessing harmonics matches exactly to the
        original precessing waveform.
        """
        from pycbc import pnutils
        from pycbc.psd import aLIGOZeroDetHighPower
        from pycbc.detector import Detector
        from pesummary.gw.conversions.snr import (
            _calculate_precessing_harmonics, _dphi,
            _make_waveform_from_precessing_harmonics
        )
        from pesummary.gw.waveform import fd_waveform

        for i in range(self.n_samples):
            duration = pnutils.get_imr_duration(
                self.mass_1[i], self.mass_2[i], self.spin_1z[i],
                self.spin_2z[i], self.f_low[i], "IMRPhenomD"
            )
            t_len = 2**np.ceil(np.log2(duration) + 1)
            df = 1./t_len
            flen = int(self.f_final[i] / df) + 1
            aLIGOpsd = aLIGOZeroDetHighPower(flen, df, self.f_low[i])
            psd = aLIGOpsd
            h = fd_waveform(
                {
                    "theta_jn": self.theta_jn[i], "phase": self.phase[i],
                    "phi_jl": self.phi_jl[i], "psi": self.psi_l[i],
                    "mass_1": self.mass_1[i], "mass_2": self.mass_2[i],
                    "tilt_1": self.tilt_1[i], "tilt_2": self.tilt_2[i],
                    "phi_12": self.phi_12[i], "a_1": self.a_1[i],
                    "a_2": self.a_2[i], "luminosity_distance": self.distance[i],
                    "ra": self.ra[i], "dec": self.dec[i],
                    "geocent_time": self.time[i]
               }, self.approx, df, self.f_low[i], self.f_final[i],
               f_ref=self.f_low[i], flen=flen, pycbc=True, project="L1"
            )
            harmonics = _calculate_precessing_harmonics(
                self.mass_1[i], self.mass_2[i], self.a_1[i],
                self.a_2[i], self.tilt_1[i], self.tilt_2[i],
                self.phi_12[i], self.beta[i], self.distance[i],
                approx=self.approx, f_final=self.f_final[i],
                flen=flen, f_ref=self.f_low[i], f_low=self.f_low[i],
                df=df, harmonics=[0, 1, 2, 3, 4]
            )
            dphi = _dphi(
                self.theta_jn[i], self.phi_jl[i], self.beta[i]
            )
            f_plus_j, f_cross_j = Detector("L1").antenna_pattern(
                self.ra[i], self.dec[i], self.psi_J[i], self.time[i]
            )
            h_all = _make_waveform_from_precessing_harmonics(
                harmonics, self.theta_jn[i], self.phi_jl[i],
                self.phase[i] - dphi, f_plus_j, f_cross_j
            )
            overlap = compute_the_overlap(
                h, h_all, psd, low_frequency_cutoff=self.f_low[i],
                high_frequency_cutoff=self.f_final[i], normalized=True
            )
            np.testing.assert_almost_equal(np.abs(overlap), 1.0)
            np.testing.assert_almost_equal(np.angle(overlap), 0.0)

    def test_precessing_snr(self):
        """Test the pesummary.gw.conversions.snr.precessing_snr function
        """
        # Use default PSD
        rho_p = precessing_snr(
            self.mass_1, self.mass_2, self.beta, self.psi_J, self.a_1, self.a_2,
            self.tilt_1, self.tilt_2, self.phi_12, self.theta_jn,
            self.ra, self.dec, self.time, self.phi_jl, self.distance,
            self.phase, f_low=self.f_low, spin_1z=self.spin_1z,
            spin_2z=self.spin_2z, multi_process=2, debug=False
        )
        assert len(rho_p) == len(self.mass_1)
        np.testing.assert_almost_equal(
            rho_p, [
                0.68377795, 4.44612704, 1.50235258, 16.36949527, 8.35617321,
                10.75820936, 5.56308683, 38.71224512, 19.0201054, 46.01320268,
                11.04007564, 22.14077344, 21.81204471, 0.52877289, 18.51382876,
                60.31991201, 20.90260283, 7.59837535, 28.78524904, 4.79727718
            ]
        )

class TestNRutils(object):

    def setup(self):
        self.mass_1 = 100
        self.mass_2 = 5
        self.total_mass =  m_total_from_m1_m2(self.mass_1, self.mass_2)
        self.eta = eta_from_m1_m2(self.mass_1, self.mass_2)
        self.spin_1z = 0.3
        self.spin_2z = 0.5
        self.chi_eff = chi_eff(
            self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
        )
        self.final_spin = bbh_final_spin_non_precessing_Healyetal(
            self.mass_1, self.mass_2, self.spin_1z, self.spin_2z,
            version="2016"
        )

    def test_bbh_peak_luminosity_non_precessing_Healyetal(self):
        from lalinference.imrtgr.nrutils import \
            bbh_peak_luminosity_non_precessing_Healyetal as lal_Healyetal

        assert np.round(
            bbh_peak_luminosity_non_precessing_Healyetal(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        ) == np.round(
            lal_Healyetal(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        )

    def test_bbh_peak_luminosity_non_precessing_T1600018(self):
        from lalinference.imrtgr.nrutils import \
            bbh_peak_luminosity_non_precessing_T1600018 as lal_T1600018

        assert np.round(
            bbh_peak_luminosity_non_precessing_T1600018(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        ) == np.round(
            lal_T1600018(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        )

    def test_bbh_peak_luminosity_non_precessing_UIB2016(self):
        from lalinference.imrtgr.nrutils import \
            bbh_peak_luminosity_non_precessing_UIB2016 as lal_UIB2016

        assert np.round(
            bbh_peak_luminosity_non_precessing_UIB2016(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        ) == np.round(
            lal_UIB2016(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z
            ), 8
        )

    def test_bbh_final_spin_non_precessing_Healyetal(self):
        from lalinference.imrtgr.nrutils import \
            bbh_final_spin_non_precessing_Healyetal as lal_Healyetal

        assert np.round(self.final_spin, 8) == np.round(
            lal_Healyetal(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z,
                version="2016"
            ), 8
        )

    def test_bbh_final_mass_non_precessing_Healyetal(self):
        from lalinference.imrtgr.nrutils import \
            bbh_final_mass_non_precessing_Healyetal as lal_Healyetal

        assert np.round(
            bbh_final_mass_non_precessing_Healyetal(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z,
                final_spin=self.final_spin, version="2016"
            ), 8
        ) == np.round(
            lal_Healyetal(
                self.mass_1, self.mass_2, self.spin_1z, self.spin_2z,
                chif=self.final_spin, version="2016"
            ), 8
        )


class TestConvert(object):
    """Test the pesummary.gw.conversions._Conversion class
    """
    def setup(self):
        """Setup the TestConvert class
        """
        self.dirs = [".outdir"]
        for dd in self.dirs:
            if not os.path.isdir(dd):
                os.mkdir(dd)

    def teardown(self):
        """Remove the files and directories created from this class
        """
        for dd in self.dirs:
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    @staticmethod
    def _convert(resume_file=None, **kwargs):
        """
        """
        np.random.seed(100)
        data = {
            "mass_1": np.random.uniform(10, 1000, 10000),
            "mass_2": np.random.uniform(2, 10, 10000),
            "spin_1z": np.random.uniform(0, 1, 10000),
            "spin_2z": np.random.uniform(0, 1, 10000)
        }
        converted = convert(data, resume_file=resume_file, **kwargs)
        return converted

    def test_from_checkpoint(self):
        """Check that when restarted from checkpoint, the output is the same
        """
        import time
        import multiprocessing
        from pesummary.io import read

        t0 = time.time()
        no_checkpoint = self._convert()
        t1 = time.time()
        # check that when interrupted and restarted, the results are the same
        process = multiprocessing.Process(
            target=self._convert, args=[".outdir/checkpoint.pickle"]
        )
        process.start()
        time.sleep(5)
        process.terminate()
        # check that not all samples have been made
        _checkpoint = read(".outdir/checkpoint.pickle", checkpoint=True)
        assert os.path.isfile(".outdir/checkpoint.pickle")
        assert not all(
            param in _checkpoint.parameters for param in no_checkpoint.keys()
        )
        # restart from checkpoint
        checkpoint = self._convert(resume_file=".outdir/checkpoint.pickle")
        for param, value in no_checkpoint.items():
            np.testing.assert_almost_equal(value, checkpoint[param])


def test_evolve_angles_backwards():
    """Check that the pesummary.gw.conversions.evolve.evolve_angles_backwards
    function works as expected
    """
    from pesummary.gw.conversions.evolve import evolve_spins
    from lal import MSUN_SI
    from packaging import version
    import lalsimulation
    input_data = [
        (
            1.3862687342652575e+32, 1.5853186050191907e+31, 0.8768912154180827,
            0.9635416612042661, 2.8861591668037119, 2.7423707262813442,
            4.750537251642867, 8.000000000000000, 2.8861301463160993, 2.7425208816155378,
            2.8861542118177956, 2.7426347054696230, 'v1'
        ),
        (
            4.0380177255695994e+31, 2.1111685497317552e+31, 0.9442047756726544,
            0.2197148251155545, 2.7060072810080551, 0.8920951236808333,
            1.7330264974887994, 14.0000000000000, 2.7082295817672812, 0.8821772303625787,
            2.7084623305332132, 0.8811491121799003, 'v1'
        ),
        (
            1.4778236544770486e+32, 2.6197742077777032e+31, 0.4650532384488123,
            0.4135203147241133, 2.5477872046486589, 1.3374887745402186,
            5.8300235171959054, 15.0000000000000, 2.5307614455226255, 1.3999636874375283,
            2.5310639813744329, 1.4020896531141123, 'v1'
        )
    ]
    for num, method in enumerate(["precession_averaged", "hybrid_orbit_averaged"]):
        for sample in input_data:
            _sample = np.array(sample[:8])
            _sample[:2] = _sample[:2] / MSUN_SI
            tilt_1_inf, tilt_2_inf, phi_12 = evolve_spins(
                *_sample, method=method, multi_process=1,
                evolve_limit="infinite_separation", version=sample[-1]
            )
            if num == 0:
                np.testing.assert_almost_equal(tilt_1_inf, sample[8], 5)
                np.testing.assert_almost_equal(tilt_2_inf, sample[9], 5)
            else:
                np.testing.assert_almost_equal(tilt_1_inf, sample[10], 5)
                np.testing.assert_almost_equal(tilt_2_inf, sample[11], 5)
