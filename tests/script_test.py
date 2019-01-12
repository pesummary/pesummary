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
import subprocess
from subprocess import Popen

from glob import glob

import pytest

class TestMainScript(object):

    def setup(self):
        dirs = ['./.outdir_bilby', './.outdir_lalinference',
                './.outdir_full_cbc', './.outdir_comparison',
                './.outdir_addition']
        for i in dirs:
            try:
                os.mkdir(i)
            except:
                shutil.rmtree(i)
                os.mkdir(i)

    def test_bilby(self):
        arguments = "--webdir ./.outdir_bilby"
        arguments += " --samples ./tests/files/bilby_example.h5"
        arguments += " --baseurl https://./.outdir"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        dirs = sorted(glob("./.outdir_bilby/*"))
        html = sorted(glob("./.outdir_bilby/html/*"))
        expected_dirs = ['./.outdir_bilby/home.html', './.outdir_bilby/config',
                         './.outdir_bilby/css', './.outdir_bilby/html',
                         './.outdir_bilby/js', './.outdir_bilby/samples',
                         './.outdir_bilby/plots']
        expected_html = ['./.outdir_bilby/html/approx_config.html',
                         './.outdir_bilby/html/approx_multiple.html',
                         './.outdir_bilby/html/approx.html',
                         './.outdir_bilby/html/approx_mass_1.html',
                         './.outdir_bilby/html/approx_log_likelihood.html',
                         './.outdir_bilby/html/approx_corner.html']
        assert all(i == j for i,j in zip(sorted(expected_dirs), dirs))
        assert all(i == j for i,j in zip(sorted(expected_html), html))

    def test_lalinference(self):
        arguments = "--webdir ./.outdir_lalinference"
        arguments += " --samples ./tests/files/lalinference_example.h5"
        arguments += " --baseurl https://./.outdir"
        arguments += " --approximant IMRPhenomPv2"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        dirs = sorted(glob("./.outdir_lalinference/*"))
        html = sorted(glob("./.outdir_lalinference/html/*"))
        expected_dirs = ['./.outdir_lalinference/home.html',
                         './.outdir_lalinference/config',
                         './.outdir_lalinference/css',
                         './.outdir_lalinference/html',
                         './.outdir_lalinference/js',
                         './.outdir_lalinference/samples',
                         './.outdir_lalinference/plots']
        expected_html = ['./.outdir_lalinference/html/IMRPhenomPv2_corner.html',
                         './.outdir_lalinference/html/IMRPhenomPv2.html',
                         './.outdir_lalinference/html/IMRPhenomPv2_multiple.html',
                         './.outdir_lalinference/html/IMRPhenomPv2_log_likelihood.html',
                         './.outdir_lalinference/html/IMRPhenomPv2_mass_1.html',
                         './.outdir_lalinference/html/IMRPhenomPv2_config.html']
        assert all(i == j for i,j in zip(sorted(expected_dirs), dirs))
        assert all(i == j for i,j in zip(sorted(expected_html), html))

    def test_comparison(self):
        arguments = "--webdir ./.outdir_comparison"
        arguments += " --samples ./tests/files/bilby_example.h5 " + \
                     "./tests/files/lalinference_example.h5"
        arguments += " --baseurl https://./.outdir"
        arguments += " --approximant IMRPhenomPv2 IMRPhenomP"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        dirs = sorted(glob("./.outdir_comparison/*"))
        html = sorted(glob("./.outdir_comparison/html/*"))
        expected_dirs = ['./.outdir_comparison/home.html',
                         './.outdir_comparison/config',
                         './.outdir_comparison/css',
                         './.outdir_comparison/html',
                         './.outdir_comparison/js',
                         './.outdir_comparison/samples',
                         './.outdir_comparison/plots']
        expected_html = ['./.outdir_comparison/html/IMRPhenomP_corner.html',
                         './.outdir_comparison/html/Comparison_log_likelihood.html',
                         './.outdir_comparison/html/IMRPhenomPv2_corner.html',
                         './.outdir_comparison/html/IMRPhenomP.html',
                         './.outdir_comparison/html/IMRPhenomPv2.html',
                         './.outdir_comparison/html/IMRPhenomP_config.html',
                         './.outdir_comparison/html/IMRPhenomPv2_multiple.html',
                         './.outdir_comparison/html/Comparison.html',
                         './.outdir_comparison/html/Comparison_multiple.html',
                         './.outdir_comparison/html/IMRPhenomPv2_log_likelihood.html',
                         './.outdir_comparison/html/IMRPhenomP_mass_1.html',
                         './.outdir_comparison/html/Comparison_mass_1.html',
                         './.outdir_comparison/html/IMRPhenomPv2_mass_1.html',
                         './.outdir_comparison/html/IMRPhenomP_multiple.html',
                         './.outdir_comparison/html/IMRPhenomP_log_likelihood.html',
                         './.outdir_comparison/html/IMRPhenomPv2_config.html']
        assert all(i == j for i,j in zip(dirs, sorted(expected_dirs)))
        assert all(i == j for i,j in zip(html, sorted(expected_html)))

    def test_addition(self):
        arguments = "--webdir ./.outdir_addition"
        arguments += " --samples ./tests/files/bilby_example.h5"
        arguments += " --baseurl https://./.outdir"
        arguments += " --approximant IMRPhenomPv2"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        arguments = "--add_to_existing"
        arguments += " --existing_webdir ./.outdir_addition"
        arguments += " --samples ./tests/files/lalinference_example.h5"
        arguments += " --baseurl https://./.outdir"
        arguments += " --approximant IMRPhenomP"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        dirs = sorted(glob("./.outdir_addition/*"))
        html = sorted(glob("./.outdir_addition/html/*"))
        expected_dirs = ['./.outdir_addition/config', './.outdir_addition/css',
                         './.outdir_addition/home.html',
                         './.outdir_addition/html', './.outdir_addition/js',
                         './.outdir_addition/plots',
                         './.outdir_addition/samples']
        expected_html = ['./.outdir_addition/html/Comparison.html',
                         './.outdir_addition/html/Comparison_log_likelihood.html',
                         './.outdir_addition/html/Comparison_mass_1.html',
                         './.outdir_addition/html/Comparison_multiple.html',
                         './.outdir_addition/html/IMRPhenomP.html',
                         './.outdir_addition/html/IMRPhenomP_config.html',
                         './.outdir_addition/html/IMRPhenomP_corner.html',
                         './.outdir_addition/html/IMRPhenomP_log_likelihood.html',
                         './.outdir_addition/html/IMRPhenomP_mass_1.html',
                         './.outdir_addition/html/IMRPhenomP_multiple.html',
                         './.outdir_addition/html/IMRPhenomPv2.html',
                         './.outdir_addition/html/IMRPhenomPv2_config.html',
                         './.outdir_addition/html/IMRPhenomPv2_corner.html',
                         './.outdir_addition/html/IMRPhenomPv2_log_likelihood.html',
                         './.outdir_addition/html/IMRPhenomPv2_mass_1.html',
                         './.outdir_addition/html/IMRPhenomPv2_multiple.html']
        assert all(i == j for i,j in zip(sorted(expected_dirs), dirs)) 
        assert all(i == j for i,j in zip(sorted(expected_html), html))

    def test_full_cbc(self):
        arguments = "--webdir ./.outdir_full_cbc"
        arguments += " --samples ./tests/files/GW150914_result.h5"
        arguments += " --baseurl https://./.outdir"
        arguments += " --approximant IMRPhenomPv2"
        ess = Popen("summarypages.py %s" %(arguments), shell=True)
        ess.wait()
        if ess.returncode == 0:
            assert 0==0
        else:
            assert 1==0
        dirs = sorted(glob("./.outdir_full_cbc/*"))
        html = sorted(glob("./.outdir_full_cbc/html/*")) 
        expected_dirs = ['./.outdir_full_cbc/home.html',
                         './.outdir_full_cbc/config',
                         './.outdir_full_cbc/css',
                         './.outdir_full_cbc/html',
                         './.outdir_full_cbc/js',
                         './.outdir_full_cbc/samples',
                         './.outdir_full_cbc/plots']
        expected_html = ['./.outdir_full_cbc/html/IMRPhenomPv2_dec.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_luminosity_distance.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_phase.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_1z.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_ra.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_phi_jl.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_2x.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_corner.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_phi_12.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_mass_2.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_symmetric_mass_ratio.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_tilt_2.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_multiple.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_chi_p.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_1y.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_mass_ratio.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_log_likelihood.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_total_mass.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_a_2.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_psi.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_mass_1.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_2z.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_chi_eff.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_geocent_time.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_1x.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_cos_tilt_2.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_chirp_mass.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_tilt_1.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_iota.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_a_1.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_spin_2y.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_cos_tilt_1.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_config.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_redshift.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_comoving_distance.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_mass_1_source.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_mass_2_source.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_total_mass_source.html',
                         './.outdir_full_cbc/html/IMRPhenomPv2_chirp_mass_source.html']
        assert all(i == j for i,j in zip(sorted(expected_dirs), dirs)) 
        assert all(i == j for i,j in zip(sorted(expected_html), html))
