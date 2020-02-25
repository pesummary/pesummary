# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
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

import scipy.optimize as so
import numpy as np


def _rescale_peak_luminosity(peak_luminosity):
    """Concert peak luminosity from geometric units to multiples of 10^56 egs/s
    """
    import lal

    LumPl_ergs_per_sec = lal.LUMPL_SI * 1e-49
    return LumPl_ergs_per_sec * peak_luminosity


def _RIT_expression(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z, coeffs
):
    """Function to return the even-in-interchange-of-particles expression used
    in the RIT fits
    """
    delta_m = (mass_1 - mass_2) / total_mass
    S = (mass_1 * mass_1 * spin_1z + mass_2 * mass_2 * spin_2z) / total_mass**2
    delta = (mass_2 * spin_2z - mass_1 * spin_1z) / total_mass

    S2 = S * S
    delta_m2 = delta_m * delta_m
    deltadm = delta * delta_m
    delta2 = delta * delta

    expression = 16. * symmetric_mass_ratio * symmetric_mass_ratio * (
        coeffs[0] + coeffs[1] * S + coeffs[2] * deltadm + coeffs[3] * S2
        + coeffs[4] * delta2 + coeffs[5] * delta_m2 + coeffs[6] * deltadm * S
        + coeffs[7] * S * delta2 + coeffs[8] * S2 * S + coeffs[9] * S * delta_m2
        + coeffs[10] * deltadm * S2 + coeffs[11] * delta2 * deltadm
        + coeffs[12] * delta2 * delta2 + coeffs[13] * S2 * S2
        + coeffs[14] * delta2 * S2 + coeffs[15] * delta_m2 * delta_m2
        + coeffs[16] * deltadm * delta_m2 + coeffs[17] * delta2 * delta_m2
        + coeffs[18] * S2 * delta_m2
    )
    return expression


def bbh_peak_luminosity_non_precessing_Healyetal(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z
):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the fit from Healy and Lousto: arXiv:1610.09713
    """
    coeffs = [
        0.00102101737, 0.000897428902, -9.77467189e-05, 0.000920883841,
        1.86982704e-05, -0.000391316975, -0.000120214144, 0.000148102239,
        0.00137901473, -0.000493730555, 0.000884792724, 3.29254224e-07,
        1.70170729e-05, 0.00151484008, -0.000148456828, 0.000136659593,
        0.000160343115, -6.18530577e-05, -0.00103602521
    ]

    peak_luminosity = _RIT_expression(
        mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z,
        coeffs
    )
    return _rescale_peak_luminosity(peak_luminosity)


def bbh_peak_luminosity_non_precessing_T1600018(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, chi_eff, spin_1z, spin_2z
):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the fit by Sascha Husa, Xisco Jimenez Forteza, David Keitel
    [LIGO-T1500598] using 5th order in chieff
    """
    symmetric_mass_ratio2 = symmetric_mass_ratio * symmetric_mass_ratio
    symmetric_mass_ratio4 = symmetric_mass_ratio2 * symmetric_mass_ratio2

    chidiff = spin_1z - spin_2z
    chidiff2 = chidiff * chidiff
    dm2 = 1. - 4. * symmetric_mass_ratio

    chi_eff2 = chi_eff * chi_eff
    chi_eff3 = chi_eff2 * chi_eff
    chi_eff4 = chi_eff3 * chi_eff
    chi_eff5 = chi_eff4 * chi_eff

    peak_luminosity = (
        0.012851338846828302 + 0.007822265919928252 * chi_eff
        + 0.010221856361035788 * chi_eff2 + 0.015805535732661396 * chi_eff3
        + 0.0011356206806770043 * chi_eff4 - 0.009868152529667197 * chi_eff5
    ) * symmetric_mass_ratio2 + (
        0.05681786589129071 - 0.0017473702709303457 * chi_eff
        - 0.10150706091341818 * chi_eff2 - 0.2349153289253309 * chi_eff3
        + 0.015657737820040145 * chi_eff4 + 0.19556893194885075 * chi_eff5
    ) * symmetric_mass_ratio4 + (
        0.026161288241420833 * dm2**0.541825641769908
        * symmetric_mass_ratio**3.1629576945611757 * chidiff
        + 0.0007771032100485481 * dm2**0.4499151697918658
        * symmetric_mass_ratio**1.7800346166040835 * chidiff2
    )
    return _rescale_peak_luminosity(peak_luminosity)


def bbh_peak_luminosity_non_precessing_UIB2016(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z
):
    """Return the peak luminosity (in units of 10^56 ergs/s) of an aligned-spin
    BBH using the git by David Keitel, Xisco Jimenez Forteza, Sascha Husa,
    Lionel London et al. arxiv:1612.09566v1
    """
    a0 = 0.8742169580717333
    a1 = -2.111792574893241
    a2 = 35.214103272783646
    a3 = -244.94930678226913
    a4 = 877.1061892200927
    a5 = -1172.549896493467
    b1 = 0.9800204548606681
    b2 = -0.1779843936224084
    b4 = 1.7859209418791981
    d10 = 3.789116271213293
    d20 = 0.40214125006660567
    d30 = 4.273116678713487
    f10 = 1.6281049269810424
    f11 = -3.6322940180721037
    f20 = 31.710537408279116
    f21 = -273.84758785648336
    f30 = -0.23470852321351202
    f31 = 6.961626779884965
    f40 = 0.21139341988062182
    f41 = 1.5255885529750841
    f60 = 3.0901740789623453
    f61 = -16.66465705511997
    f70 = 0.8362061463375388
    f71 = 0.

    symmetric_mass_ratio2 = symmetric_mass_ratio * symmetric_mass_ratio
    symmetric_mass_ratio3 = symmetric_mass_ratio2 * symmetric_mass_ratio
    symmetric_mass_ratio4 = symmetric_mass_ratio2 * symmetric_mass_ratio2
    symmetric_mass_ratio5 = symmetric_mass_ratio3 * symmetric_mass_ratio2

    mass_12 = mass_1**2
    mass_22 = mass_2**2
    S1 = spin_1z * mass_12 / total_mass**2
    S2 = spin_2z * mass_22 / total_mass**2
    Stot = S1 + S2
    Shat = (spin_1z * mass_12 + spin_2z * mass_22) / (mass_12 + mass_22)
    Shat2 = Shat * Shat
    Shat3 = Shat2 * Shat
    Shat4 = Shat2 * Shat2
    chidiff = spin_1z - spin_2z
    chidiff2 = chidiff * chidiff
    sqrt1m4eta = (1. - 4. * symmetric_mass_ratio)**0.5

    peak_luminosity = (
        a0 + a1 * symmetric_mass_ratio + a2 * symmetric_mass_ratio2
        + a3 * symmetric_mass_ratio3 + a4 * symmetric_mass_ratio4
        + a5 * symmetric_mass_ratio5 + (
            0.465 * b1 * Shat * (
                f10 + f11 * symmetric_mass_ratio + (
                    16. - 16. * f10 - 4. * f11
                ) * symmetric_mass_ratio2
            ) + 0.107 * b2 * Shat2 * (
                f20 + f21 * symmetric_mass_ratio + (
                    16. - 16. * f20 - 4. * f21
                ) * symmetric_mass_ratio2
            ) + Shat3 * (
                f30 + f31 * symmetric_mass_ratio + (
                    -16. * f30 - 4. * f31
                ) * symmetric_mass_ratio2
            ) + Shat4 * (
                f40 + f41 * symmetric_mass_ratio + (
                    -16. * f40 - 4. * f41
                ) * symmetric_mass_ratio2
            )
        ) / (
            1. - 0.328 * b4 * Shat * (
                f60 + f61 * symmetric_mass_ratio + (
                    16. - 16. * f60 - 4. * f61
                ) * symmetric_mass_ratio2
            ) + Shat2 * (
                f70 + f71 * symmetric_mass_ratio + (
                    -16. * f70 - 4. * f71
                ) * symmetric_mass_ratio2
            )
        ) + symmetric_mass_ratio3 * (
            (d10 + d30 * Shat) * sqrt1m4eta * chidiff + d20 * chidiff2
        )
    )
    L0 = 0.016379197203103536
    peak_luminosity = peak_luminosity * symmetric_mass_ratio2 * L0
    return _rescale_peak_luminosity(peak_luminosity)


def isco_radius(a):
    """Return the ISCO radius of a Kerr BH as a function of the final spin
    using equations 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)
    """
    z1 = 1. + (1. - a**2.)**(1. / 3) * ((1. + a)**(1. / 3) + (1. - a)**(1. / 3))
    z2 = np.sqrt(3. * a**2 + z1**2)
    a_sign = np.sign(a)
    return 3 + z2 - np.sqrt((3. - z1) * (3. + z1 + 2. * z2)) * a_sign


def _final_spin_diff_Healyetal(
    final_spin, mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z,
    spin_2z
):
    """The final spin with the Healy et al. fits is determined by minimizing
    this function
    """
    S = (mass_1 * mass_1 * spin_1z + mass_2 * mass_2 * spin_2z) / total_mass**2
    delta_m = (mass_1 - mass_2) / total_mass
    r_isco = isco_radius(final_spin)
    J_isco = (3 * np.sqrt(r_isco) - 2 * final_spin) * 2. / np.sqrt(3 * r_isco)
    L0 = 0.686732132
    L1 = 0.613284976
    L2a = -0.148530075
    L2b = -0.113826318
    L2c = -0.00323995784
    L2d = 0.798011319
    L3a = -0.0687823713
    L3b = 0.00129103641
    L3c = -0.0780143929
    L3d = 1.55728564
    L4a = -0.00571010557
    L4b = 0.005919799
    L4c = -0.00170575554
    L4d = -0.0588819084
    L4e = -0.0101866693
    L4f = 0.964444768
    L4g = -0.11088507
    L4h = -0.00682082169
    L4i = -0.0816482139

    delta_m4 = delta_m**4
    delta_m6 = delta_m**6

    final_spin_new = _RIT_expression(
        mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z,
        [
            L0, L1, L2a, L2b, L2c, L2d, L3a, L3b, L3c, L3d, L4a, L4b, L4c, L4d,
            L4e, L4f, L4g, L4h, L4i
        ]
    ) + (
        S * (1. + 8. * symmetric_mass_ratio) * delta_m4
        + symmetric_mass_ratio * J_isco * delta_m6
    )
    return abs(final_spin - final_spin_new)


def bbh_final_spin_non_precessing_Healyetal(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z
):
    """Return the final spin of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Healy and Lousto: arXiv:1610.09713
    """
    return_float = False
    if isinstance(mass_1, (float, int)):
        return_float = True
        mass_1 = np.array([mass_1])
        mass_2 = np.array([mass_2])
        total_mass = np.array([total_mass])
        symmetric_mass_ratio = np.array([symmetric_mass_ratio])
        spin_1z = np.array([spin_1z])
        spin_2z = np.array([spin_2z])

    final_spin = []
    for i in range(len(mass_1)):
        fs = so.leastsq(
            _final_spin_diff_Healyetal, 0., args=(
                mass_1[i], mass_2[i], total_mass[i], symmetric_mass_ratio[i],
                spin_1z[i], spin_2z[i]
            )
        )
        final_spin.append(fs[0][0])

    if return_float:
        return final_spin[0]
    return np.array(final_spin)


def bbh_final_mass_non_precessing_Healyetal(
    mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z,
    final_spin
):
    """Return the final mass of the BH resulting from the merger of a BBH for an
    aligned-spin system using the fit from Healy and Lousto: arXiv:1610.09713
    """
    r_isco = isco_radius(final_spin)
    M0 = 0.951659087
    K1 = -0.0511301363
    K2a = -0.00569897591
    K2b = -0.0580644933
    K2c = -0.00186732281
    K2d = 1.99570464
    K3a = 0.00499137602
    K3b = -0.00923776244
    K3c = -0.120577082
    K3d = 0.0164168385
    K4a = -0.0607207285
    K4b = -0.00179828653
    K4c = 0.000654388173
    K4d = -0.156625642
    K4e = 0.0103033606
    K4f = 2.97872857
    K4g = 0.00790433045
    K4h = 0.000631241195
    K4i = 0.0844776942

    delta_m = (mass_1 - mass_2) / total_mass
    delta_m6 = delta_m**6

    E_isco = (
        1. - 2. / r_isco + final_spin / r_isco**1.5
    ) / np.sqrt(1. - 3. / r_isco + 2. * final_spin / r_isco**1.5)

    final_mass = _RIT_expression(
        mass_1, mass_2, total_mass, symmetric_mass_ratio, spin_1z, spin_2z,
        [
            M0, K1, K2a, K2b, K2c, K2d, K3a, K3b, K3c, K3d, K4a, K4b, K4c, K4d,
            K4e, K4f, K4g, K4h, K4i
        ]
    ) + (1 + symmetric_mass_ratio * (E_isco + 11.)) * delta_m6
    return final_mass * total_mass


def bbh_final_mass_non_spinning_Panetal(total_mass, symmetric_mass_ratio):
    """Return the final mass of the BH resulting from the merger of a non
    spinning BBH using the fit from Pan et al: Phys Rev D 84, 124052 (2011).
    """
    return total_mass * (
        1. + (np.sqrt(8. / 9.) - 1.) * symmetric_mass_ratio
        - 0.4333 * (symmetric_mass_ratio**2.) - (
            0.4392 * (symmetric_mass_ratio**3)
        )
    )


def bbh_final_spin_non_spinning_Panetal(symmetric_mass_ratio):
    """Return the final spin of the BH resulting from the merger of a non
    spinning BBH using the fit from Pan et al: Phys Rev D 84, 124052 (2011)
    """
    return (
        np.sqrt(12.) * symmetric_mass_ratio - 3.871 * (symmetric_mass_ratio**2.)
        + 4.028 * (symmetric_mass_ratio**3)
    )
