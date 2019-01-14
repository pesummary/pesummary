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

from pesummary.one_format import data_format

import pytest


class TestOneFormat(object):

    def setup(self):
        directory = './.outdir'
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

    @pytest.mark.parametrize('parameters, data, approximant', [
        ([b"mass1", b"mass2", b"mtotal"],
         [[10., 10., 20.], [30., 10., 40.], [100., 20., 120.]],
         b"IMRPhenomPv2"),])
    def test_hdf5_creation(self, parameters, data, approximant):
        inj_par = parameters
        inj_data = [float("nan")]*len(parameters)
        name = ".outdir/test.h5"
        data_format._make_hdf5_file(name, np.array(data), np.array(parameters),
                                    np.array([approximant]),
                                    inj_par=np.array(inj_par),
                                    inj_data=np.array(inj_data))
        assert os.path.isfile("./.outdir/test.h5_temp") == True
        f = h5py.File("./.outdir/test.h5_temp")
        keys = sorted(list(f.keys()))
        assert keys == sorted(["parameter_names", "samples", "approximant",
                               "injection_parameters", "injection_data"])
        extracted_parameters = [i for i in f["parameter_names"]]
        assert extracted_parameters == parameters
        extracted_data = [i for i in f["samples"]]
        for num, i in enumerate(extracted_data):
            assert i.tolist() == data[num]
        extracted_approximant = f["approximant"][0]
        assert extracted_approximant == approximant


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
        redshift = data_format._z_from_dL(l_distance)
        assert np.round(redshift, 4) == 0.1049

    def test_dL_from_z(self):
        redshift = self.opts.redshift
        l_distance = data_format._dL_from_z(redshift)
        assert np.round(l_distance, 4) == 2918.3419

    def test_comoving_distance_from_z(self):
        redshift = self.opts.redshift
        c_distance = data_format._comoving_distance_from_z(redshift)
        assert np.round(c_distance, 4) == 1945.5613 

    def test_m1_source_from_m1_z(self):
        mass_1, redshift = self.opts.mass1, self.opts.redshift
        mass_1_source = data_format._m1_source_from_m1_z(mass_1, redshift)
        assert np.round(mass_1_source, 4) == 6.6667 

    def test_m2_source_from_m2_z(self):
        mass_2, redshift = self.opts.mass2, self.opts.redshift
        mass_2_source = data_format._m2_source_from_m2_z(mass_2, redshift)
        assert np.round(mass_2_source, 4) == 3.3333 

    def test_m_total_source_from_mtotal_z(self):
        total_mass, redshift = self.opts.mtotal, self.opts.redshift
        m_total_source = data_format._m_total_source_from_mtotal_z(total_mass,
                                                                   redshift)
        assert m_total_source == 20.

    def test_mchirp_source_from_mchirp_z(self):
        mchirp, redshift = self.opts.mchirp, self.opts.redshift
        mchirp_source = data_format._mchirp_source_from_mchirp_z(mchirp,
                                                                 redshift)
        assert np.round(mchirp_source, 4) == 6.6667 

    def test_mchirp_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        mchirp = data_format._mchirp_from_m1_m2(mass1, mass2)
        assert np.round(mchirp, 4) == 6.0836

    def test_m_total_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        mtotal = data_format._m_total_from_m1_m2(mass1, mass2)
        assert mtotal == 15.

    def test_m1_from_mchirp_q(self):
        mchirp, q = self.opts.mchirp, self.opts.q
        mass1 = data_format._m1_from_mchirp_q(mchirp, q)
        assert np.round(mass1, 4) == 24.0225

    def test_m2_from_mchirp_q(self):
        mchirp, q = self.opts.mchirp, self.opts.q
        mass2 = data_format._m2_from_mchirp_q(mchirp, q)
        assert np.round(mass2, 4) == 6.0056

    def test_eta_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        eta = data_format._eta_from_m1_m2(mass1, mass2)
        assert np.round(eta, 4) == 0.2222

    def test_q_from_m1_m2(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        q = data_format._q_from_m1_m2(mass1, mass2)
        assert q == 2.

    def test_q_from_eta(self):
        eta = self.opts.eta
        q = data_format._q_from_eta(eta)
        assert np.round(q, 4) == 0.4498

    def test_mchirp_from_mtotal_q(self):
        mtotal, q = self.opts.mtotal, self.opts.q
        mchirp = data_format._mchirp_from_mtotal_q(mtotal, q)
        assert np.round(mchirp, 4) == 9.9906

    def test_chi_p(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        spin1x, spin1y = self.opts.spin1x, self.opts.spin1y
        spin1z, spin2x = self.opts.spin1z, self.opts.spin2x
        spin2y, spin2z = self.opts.spin2y, self.opts.spin2z
        chi_p = data_format._chi_p(mass1, mass2, spin1x, spin1y, spin2y, spin2y)
        assert chi_p == 0.75

    def test_chi_eff(self):
        mass1, mass2 = self.opts.mass1, self.opts.mass2
        spin1z, spin2z = self.opts.spin1z, self.opts.spin2z
        chi_eff = data_format._chi_eff(mass1, mass2, spin1z, spin2z)
        assert chi_eff == 0.5

    def test_component_spins(self):
        mass1, mass2 = [10., 10.], [5., 5.]
        thetajn, phijl = self.opts.theta_jn, self.opts.phi_jl
        t1, t2, phi12 = self.opts.tilt_1, self.opts.tilt_2, self.opts.phi_12
        a1, a2, f_ref = self.opts.a_1, self.opts.a_2, self.opts.f_ref
        phase = self.opts.phase
        data = data_format._component_spins(thetajn, phijl, t1, t2, phi12, a1,
                                            a2, mass1, mass2, f_ref, phase)
        rounded = np.round(data, 4)
        expected = [[0.4841,-0.2359,-0.0425, 0.4388,0.,0.,0.],
                    [0.5,0.,0.,0.5,0.,0.,0.]]
        for num, i in enumerate(rounded):
            assert i.tolist() == expected[num]

    def test_all_parameters(self):
        parameters = ["mass_1", "mass_2", "iota", "phi_jl", "tilt_1", "tilt_2",
                      "phi_12", "a_1", "a_2", "phase"]
        data = [[self.opts.mass1, self.opts.mass2, self.opts.iota, self.opts.phi_jl[0],
                 self.opts.tilt_1[0], self.opts.tilt_2[0], self.opts.phi_12[0],
                 self.opts.a_1[0], self.opts.a_2[0], self.opts.phase[0]]]*2
        cbc_data = data_format.all_parameters(data, parameters)
        expected_params = ['mass_1', 'mass_2', 'iota', 'phi_jl', 'tilt_1', 'tilt_2',
                           'phi_12', 'a_1', 'a_2', 'phase', 'mass_ratio',
                           'reference_frequency', 'total_mass', 'chirp_mass',
                           'symmetric_mass_ratio', 'spin_1x', 'spin_1y', 'spin_1z',
                           'spin_2x', 'spin_2y', 'spin_2z', 'chi_p', 'chi_eff',
                           'cos_tilt_1', 'cos_tilt_2']
        expected_data = [[10.0, 5.0, 0.5, 0.3, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
                          2.0, 2.0, 20.0, 20.0, 15.0, 15.0, 6.0836, 6.0836,
                          0.2222, 0.2222, -0.2329, -0.0566, 0.4388, -0.0, -0.0,
                          0.0, -0.2329, -0.0566, 0.4388, -0.0, -0.0, 0.0,
                          16.1867, 3.9781, 16.1867, 3.9781, 0.8776, 0.8776, 1.0,
                          1.0]]*2
        cbc_params = cbc_data[1]
        cbc_data = np.round(cbc_data[0], 4).tolist()
        assert all(i==j for i,j in zip(expected_params, cbc_params))
        assert all(i==j for i,j in zip(expected_data[0], cbc_data[0]))

    def test_one_format(self):
        path = "./tests/files/GW150914_result.h5"
        output = data_format.one_format(path, None)
        assert os.path.isfile("./tests/files/GW150914_result.h5_temp")
        path = "./tests/files/lalinference_example.h5"
        output = data_format.one_format(path, None)
        assert os.path.isfile("./tests/files/lalinference_example.h5_temp")

    def test_load_with_deepdish(self):
        path = "./tests/files/bilby_example.h5"
        f = deepdish.io.load(path)
        output = data_format.load_with_deepdish(f)
        params = sorted(output[0])
        samples = output[1]
        approximant = output[2]
        expected = [[1.0, 10.0], [2.0, 10.0], [3.0, 20.0]]
        if output[0].index("log_likelihood") != params.index("log_likelihood"):
            expected = [[10.0, 1.0], [10.0, 2.0], [20.0, 3.0]] 
        assert all(i == j for i,j in zip(params, ["log_likelihood", "mass_1"]))
        assert all(all(i==j for i,j in zip(k,l)) for k,l in zip(samples, expected))
        assert approximant == "approx"

    def test_load_with_h5py(self):
        path = "./tests/files/GW150914_result.h5"
        f = h5py.File(path)
        output = data_format.load_with_h5py(f, "posterior")
        params = sorted(output[0])
        expected_params = ['a_1', 'a_2', 'dec', 'geocent_time', 'iota',
                           'log_likelihood', 'luminosity_distance', 'mass_1',
                           'mass_2', 'phase', 'phi_12', 'phi_jl', 'psi', 'ra',
                           'tilt_1', 'tilt_2']
        assert all(i == j for i,j in zip(params, sorted(expected_params)))
