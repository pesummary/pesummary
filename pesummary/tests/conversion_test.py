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
import socket
import shutil

import numpy as np
import h5py

import deepdish

from pesummary.gw.file.conversions import *
from pesummary.gw.file.nrutils import *
from pycbc import conversions
import pytest


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
        from pesummary.utils.samples_dict import Array
        from pesummary.gw.file.conversions import _Conversion
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
        data = _Conversion(dictionary)
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
        from pesummary.gw.file.conversions import _Conversion

        dictionary = {
            "mass_1": np.random.uniform(5, 100, 100),
            "mass_ratio": [0.1] * 100
        }
        dictionary["mass_2"] = np.random.uniform(2, dictionary["mass_1"], 100)
        incorrect_mass_ratio = _Conversion(dictionary)
        data = _Conversion(dictionary, regenerate=["mass_ratio"])
        assert all(i != j for i, j in zip(
            incorrect_mass_ratio["mass_ratio"], data["mass_ratio"]
        ))
        assert np.testing.assert_almost_equal(
            data["mass_ratio"],
            q_from_m1_m2(dictionary["mass_1"], dictionary["mass_2"]), 8
       ) is None


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
