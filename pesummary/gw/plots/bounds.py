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

import numpy as np


default_bounds = {"luminosity_distance": {"low": 0.},
                  "geocent_time": {"low": 0.},
                  "dec": {"low": -np.pi / 2, "high": np.pi / 2},
                  "ra": {"low": 0., "high": 2 * np.pi},
                  "a_1": {"low": 0., "high": 1.},
                  "a_2": {"low": 0., "high": 1.},
                  "phi_jl": {"low": 0., "high": 2 * np.pi},
                  "phase": {"low": 0., "high": 2 * np.pi},
                  "psi": {"low": 0., "high": 2 * np.pi},
                  "iota": {"low": 0., "high": np.pi},
                  "tilt_1": {"low": 0., "high": np.pi},
                  "tilt_2": {"low": 0., "high": np.pi},
                  "phi_12": {"low": 0., "high": 2 * np.pi},
                  "mass_2": {"low": 0., "high": "mass_1"},
                  "mass_1": {"low": 0},
                  "total_mass": {"low": 0.},
                  "chirp_mass": {"low": 0.},
                  "H1_time": {"low": 0.},
                  "L1_time": {"low": 0.},
                  "V1_time": {"low": 0.},
                  "E1_time": {"low": 0.},
                  "spin_1x": {"low": 0., "high": 1.},
                  "spin_1y": {"low": 0., "high": 1.},
                  "spin_1z": {"low": -1., "high": 1.},
                  "spin_2x": {"low": 0., "high": 1.},
                  "spin_2y": {"low": 0., "high": 1.},
                  "spin_2z": {"low": -1., "high": 1.},
                  "chi_p": {"low": 0., "high": 1.},
                  "chi_eff": {"low": -1., "high": 1.},
                  "mass_ratio": {"low": 0., "high": 1.},
                  "symmetric_mass_ratio": {"low": 0., "high": 0.25},
                  "phi_1": {"low": 0., "high": 2 * np.pi},
                  "phi_2": {"low": 0., "high": 2 * np.pi},
                  "cos_tilt_1": {"low": -1., "high": 1.},
                  "cos_tilt_2": {"low": -1., "high": 1.},
                  "redshift": {"low": 0.},
                  "comoving_distance": {"low": 0.},
                  "mass_1_source": {"low": 0.},
                  "mass_2_source": {"low": 0., "high": "mass_1_source"},
                  "chirp_mass_source": {"low": 0.},
                  "total_mass_source": {"low": 0.},
                  "cos_iota": {"low": -1., "high": 1.},
                  "theta_jn": {"low": 0., "high": np.pi},
                  "lambda_1": {"low": 0.},
                  "lambda_2": {"low": 0.},
                  "lambda_tilde": {"low": 0.},
                  "delta_lambda": {}}
