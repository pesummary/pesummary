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

from pesummary.file import one_format
from pesummary.file.conversions import *

import pytest


class TestConversions(object):

    @classmethod
    def setup_class(cls):
        class Arguments(object):
            mass1 = 10.
            mass2 = 5.
            mtotal = 30.
            mchirp = 10.
            q = 4.
            eta = 0.214
            iota = 0.5
            spin1x = 0.75
            spin1y = 0.
            spin1z = 0.5
            spin2x = 0.
            spin2y = 0.
            spin2z = 0.5

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
        redshift = z_from_dL(l_distance)
        assert np.round(redshift, 4) == 0.1049

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
        assert q == 2.

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

    def test_one_format(self):
        path = "./tests/files/GW150914_result.h5"
        output = one_format.OneFormat(path, None)
        output.save()
        assert os.path.isfile("./tests/files/GW150914_result.h5_temp")
        path = "./tests/files/lalinference_example.h5"
        output = one_format.OneFormat(path, None)
        output.save()
        assert os.path.isfile("./tests/files/lalinference_example.h5_temp")

    def test_load_with_deepdish(self):
        path = "./tests/files/bilby_example.h5"
        f = deepdish.io.load(path)
        output = one_format.load_with_deepdish(f)
        params = sorted(output[0])
        samples = output[1]
        approximant = output[2]
        expected = [[1.0, 10.0, 10.], [2.0, 10.0, 20.], [3.0, 20.0, 30.], 
            [0., 0., 0.,]]
        assert all(i in ["H1_optimal_snr", "log_likelihood", "mass_1"] for i in params)
        assert all(all(i in expected[num] for i in k) for num, k in enumerate(samples))
        assert approximant == b"IMRPhenomPv2"

    def test_load_with_h5py(self):
        path = "./tests/files/GW150914_result.h5"
        f = h5py.File(path)
        output = one_format.load_with_h5py(f, "posterior")
        params = sorted(output[0])
        expected_params = ['a_1', 'a_2', 'dec', 'geocent_time', 'iota',
                           'log_likelihood', 'luminosity_distance', 'mass_1',
                           'mass_2', 'phase', 'phi_12', 'phi_jl', 'psi', 'ra',
                           'tilt_1', 'tilt_2']
        assert all(i == j for i,j in zip(params, sorted(expected_params)))
