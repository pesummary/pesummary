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
import pytest


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
        l_distance = self.opts.l_distance
        redshift = z_from_dL_approx(l_distance)
        redshift_exact = z_from_dL_exact(l_distance)
        assert np.round(redshift, 4) == 0.1049
        assert np.round(redshift_exact, 4) == 0.1049

    def test_dL_from_z(self):
        redshift = self.opts.redshift
        l_distance = dL_from_z(redshift)
        assert np.round(l_distance, 4) == 2918.3419

    def test_comoving_distance_from_z(self):
        redshift = self.opts.redshift
        c_distance = comoving_distance_from_z(redshift)
        assert np.round(c_distance, 4) == 1945.5613 

    def test_m1_source_from_m1_z(self):
        mass_1, redshift = self.opts.mass1, self.opts.redshift
        mass_1_source = m1_source_from_m1_z(mass_1, redshift)
        assert np.round(mass_1_source, 4) == 6.6667 

    def test_m2_source_from_m2_z(self):
        mass_2, redshift = self.opts.mass2, self.opts.redshift
        mass_2_source = m2_source_from_m2_z(mass_2, redshift)
        assert np.round(mass_2_source, 4) == 3.3333 

    def test_m_total_source_from_mtotal_z(self):
        total_mass, redshift = self.opts.mtotal, self.opts.redshift
        m_total_source = m_total_source_from_mtotal_z(total_mass, redshift)
        assert m_total_source == 20.

    def test_m_total_from_mtotal_source_z(self):
        total_mass_source, redshift = 20., self.opts.redshift
        m_total = mtotal_from_mtotal_source_z(total_mass_source, redshift)
        assert np.round(m_total, 4) == self.opts.mtotal

    def test_mchirp_source_from_mchirp_z(self):
        mchirp, redshift = self.opts.mchirp, self.opts.redshift
        mchirp_source = mchirp_source_from_mchirp_z(mchirp, redshift)
        assert np.round(mchirp_source, 4) == 6.6667

    def test_mchirp_from_mchirp_source_z(self):
        mchirp_source, redshift = 20./3., self.opts.redshift
        mchirp = mchirp_from_mchirp_source_z(mchirp_source, redshift)
        assert np.round(mchirp, 4) == self.opts.mchirp

    def test_mchirp_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        mchirp = mchirp_from_m1_m2(mass1, mass2)
        assert np.round(mchirp, 4) == 6.0836

    def test_m_total_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        mtotal = m_total_from_m1_m2(mass1, mass2)
        assert mtotal == 15.

    def test_m1_from_mchirp_q(self):
        mchirp, q = self.opts.mchirp, self.opts.q
        mass1 = m1_from_mchirp_q(mchirp, q)
        assert np.round(mass1, 4) == 24.0225

    def test_m2_from_mchirp_q(self):
        mchirp, q = self.opts.mchirp, self.opts.q
        mass2 = m2_from_mchirp_q(mchirp, q)
        assert np.round(mass2, 4) == 6.0056

    def test_eta_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        eta = eta_from_m1_m2(mass1, mass2)
        assert np.round(eta, 4) == 0.2222

    def test_q_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        q = q_from_m1_m2(mass1, mass2)
        assert q == 1. / 2.

    def test_q_from_eta(self):
        eta = self.opts.eta
        q = q_from_eta(eta)
        assert np.round(q, 4) == 0.4498

    def test_mchirp_from_mtotal_q(self):
        mtotal, q = self.opts.mtotal, self.opts.q
        mchirp = mchirp_from_mtotal_q(mtotal, q)
        assert np.round(mchirp, 4) == 9.9906

    def test_chi_p(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        spin1x, spin1y = self.opts.spin1x, self.opts.spin1y
        spin1z, spin2x = self.opts.spin1z, self.opts.spin2x
        spin2y, spin2z = self.opts.spin2y, self.opts.spin2z
        chi_p_value = chi_p(mass1, mass2, spin1x, spin1y, spin2y, spin2y)
        assert chi_p_value == 0.75

    def test_chi_eff(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        spin1z, spin2z = self.opts.spin1z, self.opts.spin2z
        chi_eff_value = chi_eff(mass1, mass2, spin1z, spin2z)
        assert chi_eff_value == 0.5

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
        mass1, mass2 = [10., 10.], [5., 5.]
        inc, spin1x, spin1y = self.opts.iota, self.opts.spin1x, self.opts.spin1y
        spin1z, spin2x = self.opts.spin1z, self.opts.spin2x,
        spin2y, spin2z = self.opts.spin2y, self.opts.spin2z
        f_ref, phase = self.opts.f_ref, self.opts.phase
        data = spin_angles(mass1, mass2, [inc]*2, [spin1x]*2, [spin1y]*2,
                           [spin1z]*2, [spin2x]*2, [spin2y]*2, [spin2z]*2,
                           f_ref, phase)
        rounded = np.round(data, 4)
        expected = [0.5345, 2.797, 0.9828, 0.0, 0.0, 0.9014, 0.5]
        assert all(i == j for i,j in zip(rounded[0], expected))

    def test_component_spins(self):
        mass1, mass2 = [10., 10.], [5., 5.]
        thetajn, phijl = self.opts.theta_jn, self.opts.phi_jl
        t1, t2, phi12 = self.opts.tilt_1, self.opts.tilt_2, self.opts.phi_12
        a1, a2, f_ref = self.opts.a_1, self.opts.a_2, self.opts.f_ref
        phase = self.opts.phase
        data = component_spins(thetajn, phijl, t1, t2, phi12, a1, a2, mass1,
            mass2, f_ref, phase)
        rounded = np.round(data, 4)
        expected = [[0.4841,-0.2359,-0.0425, 0.4388,0.,0.,0.],
                    [0.5,0.,0.,0.5,0.,0.,0.]]
        for num, i in enumerate(rounded):
            assert i.tolist() == expected[num]

    def test_time_in_each_ifo(self):
        optimal_ra, optimal_dec = -0.2559168059473027, 0.81079526383
        time = time_in_each_ifo("H1", optimal_ra, optimal_dec, 0)
        light_time = 6371*10**3 / (3.0*10**8)
        assert -np.round(light_time, 4) == np.round(time, 4)

    def test_lambda_tilde_from_lambda1_lambda2(self):
        lambda_tilde = lambda_tilde_from_lambda1_lambda2(
            self.opts.lambda1, self.opts.lambda2, self.opts.mass1,
            self.opts.mass2)
        assert np.round(lambda_tilde, 4) == 630.5793

    def test_delta_lambda_from_lambda1_lambda2(self):
        delta_lambda = delta_lambda_from_lambda1_lambda2(
            self.opts.lambda1, self.opts.lambda2, self.opts.mass1,
            self.opts.mass2)
        assert np.round(delta_lambda, 4) == -150.1964

    def test_lambda1_from_lambda_tilde(self):
        lambda1 = lambda1_from_lambda_tilde(
            self.opts.lambda_tilde, self.opts.mass1, self.opts.mass2)
        assert np.round(lambda1, 4) == 192.8101

    def test_lambda2_from_lambda1(self):
        lambda2 = lambda2_from_lambda1(
            self.opts.lambda1, self.opts.mass1, self.opts.mass2)
        assert np.round(lambda2, 4) == 16000.0

    def test_network_snr(self):
        snr_H1 = snr_L1 = snr_V1 = np.array([2., 3.])
        assert network_snr([snr_H1[0], snr_L1[0], snr_V1[0]]) == np.sqrt(3) * 2
        print(snr_H1)
        network = network_snr([snr_H1, snr_L1, snr_V1])
        print(network)
        assert network[0] == np.sqrt(3) * 2
        assert network[1] == np.sqrt(3) * 3

    def test_full_conversion(self):
        from pesummary.utils.utils import Array
        from pesummary.gw.file.conversions import _Conversion

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
            "H1_matched_filter_snr": [10.0]
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
