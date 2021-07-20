# Licensed under an MIT style license -- see LICENSE.md

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_IFOS = ["H1", "L1", "V1", "K1", "E1"]
tidal_params = ["lambda_1", "lambda_2", "delta_lambda", "lambda_tilde"]


lalinference_map = {
    "logl": "log_likelihood",
    "logprior": "log_prior",
    "matched_filter_snr": "network_matched_filter_snr",
    "optimal_snr": "network_optimal_snr",
    "phi12": "phi_12",
    "q": "mass_ratio",
    "time": "geocent_time",
    "dist": "luminosity_distance",
    "mc": "chirp_mass",
    "a1": "a_1",
    "a2": "a_2",
    "tilt1": "tilt_1",
    "tilt2": "tilt_2",
    "m1": "mass_1",
    "m2": "mass_2",
    "eta": "symmetric_mass_ratio",
    "mtotal": "total_mass",
    "h1_end_time": "H1_time",
    "l1_end_time": "L1_time",
    "v1_end_time": "V1_time",
    "a1z": "spin_1z",
    "a2z": "spin_2z",
    "m1_source": "mass_1_source",
    "m2_source": "mass_2_source",
    "mtotal_source": "total_mass_source",
    "mc_source": "chirp_mass_source",
    "phi1": "phi_1",
    "phi2": "phi_2",
    "costilt1": "cos_tilt_1",
    "costilt2": "cos_tilt_2",
    "costheta_jn": "cos_theta_jn",
    "cosiota": "cos_iota",
    "lambda1": "lambda_1",
    "lambda2": "lambda_2",
    "lambdaT": "lambda_tilde",
    "dLambdaT": "delta_lambda",
    "logp1": "log_pressure",
    "gamma1": "gamma_1",
    "gamma2": "gamma_2",
    "gamma3": "gamma_3",
    "SDgamma0": "spectral_decomposition_gamma_0",
    "SDgamma1": "spectral_decomposition_gamma_1",
    "SDgamma2": "spectral_decomposition_gamma_2",
    "SDgamma3": "spectral_decomposition_gamma_3",
    "sdgamma0": "spectral_decomposition_gamma_0",
    "sdgamma1": "spectral_decomposition_gamma_1",
    "sdgamma2": "spectral_decomposition_gamma_2",
    "sdgamma3": "spectral_decomposition_gamma_3",
    "mf_evol_avg": "final_mass",
    "mf_nonevol": "final_mass_non_evolved",
    "mf_source_evol_avg": "final_mass_source",
    "mf_source_nonevol": "final_mass_source_non_evolved",
    "af_nonevol": "final_spin_non_evolved",
    "af_evol_avg": "final_spin",
    "l_peak_evol_avg": "peak_luminosity",
    "l_peak_nonevol": "peak_luminosity_non_evolved",
    "e_rad_nonevol": "radiated_energy_non_evolved",
    "e_rad_evol_avg": "radiated_energy",
    "beta": "beta"
}


for detector in _IFOS:
    lalinference_map["{}_cplx_snr_amp".format(detector.lower())] = (
        "{}_matched_filter_abs_snr".format(detector)
    )
    lalinference_map["{}_cplx_snr_arg".format(detector.lower())] = (
        "{}_matched_filter_snr_angle".format(detector)
    )
    lalinference_map["{}_optimal_snr".format(detector.lower())] = (
        "{}_optimal_snr".format(detector)
    )


bilby_map = {
    "chirp_mass": "chirp_mass",
    "mass_ratio": "mass_ratio",
    "a_1": "a_1",
    "a_2": "a_2",
    "tilt_1": "tilt_1",
    "tilt_2": "tilt_2",
    "phi_12": "phi_12",
    "phi_jl": "phi_jl",
    "dec": "dec",
    "ra": "ra",
    "theta_jn": "theta_jn",
    "psi": "psi",
    "luminosity_distance": "luminosity_distance",
    "phase": "phase",
    "geocent_time": "geocent_time",
    "log_likelihood": "log_likelihood",
    "log_prior": "log_prior",
    "reference_frequency": "reference_frequency",
    "total_mass": "total_mass",
    "mass_1": "mass_1",
    "mass_2": "mass_2",
    "symmetric_mass_ratio": "symmetric_mass_ratio",
    "iota": "iota",
    "spin_1x": "spin_1x",
    "spin_1y": "spin_1y",
    "spin_1z": "spin_1z",
    "spin_2x": "spin_2x",
    "spin_2y": "spin_2y",
    "spin_2z": "spin_2z",
    "phi_1": "phi_1",
    "phi_2": "phi_2",
    "chi_eff": "chi_eff",
    "chi_p": "chi_p",
    "redshift": "redshift",
    "mass_1_source": "mass_1_source",
    "mass_2_source": "mass_2_source",
    "chirp_mass_source": "chirp_mass_source",
    "total_mass_source": "total_mass_source",
    "lambda_1": "lambda_1",
    "lambda_2": "lambda_2",
    "lambda_tilde": "lambda_tilde",
    "cos_iota": "cos_iota",
    "cos_theta_jn": "cos_theta_jn",
}


for detector in _IFOS:
    bilby_map["{}_matched_filter_snr_abs".format(detector)] = (
        "{}_matched_filter_snr_abs".format(detector)
    )
    bilby_map["{}_matched_filter_snr_angle".format(detector)] = (
        "{}_matched_filter_snr_angle".format(detector)
    )
    bilby_map["{}_optimal_snr".format(detector)] = (
        "{}_optimal_snr".format(detector)
    )


pycbc_map = {
    "mchirp": "chirp_mass",
    "coa_phase": "phase",
    "loglikelihood": "log_likelihood",
}


pesummary_map = {
    "network_precessing_snr": "network_precessing_snr",
    "chirp_mass_source": "chirp_mass_source",
    "delta_lambda": "delta_lambda",
    "viewing_angle": "viewing_angle",
    "tilt_1_infinity": "tilt_1_infinity",
    "spin_1z_infinity": "spin_1z_infinity",
    "spin_1z_infinity_only_prec_avg": "spin_1z_infinity_only_prec_avg",
    "tilt_2_infinity": "tilt_2_infinity",
    "spin_2z_infinity": "spin_2z_infinity",
    "spin_2z_infinity_only_prec_avg": "spin_2z_infinity_only_prec_avg",
    "tilt_1_infinity_only_prec_avg": "tilt_1_infinity_only_prec_avg",
    "tilt_2_infinity_only_prec_avg": "tilt_2_infinity_only_prec_avg",
    "chi_eff_infinity": "chi_eff_infinity",
    "chi_eff_infinity_only_prec_avg": "chi_eff_infinity_only_prec_avg",
    "chi_p_infinity": "chi_p_infinity",
    "chi_p_infinity_only_prec_avg": "chi_p_infinity_only_prec_avg",
    "cos_tilt_1_infinity": "cos_tilt_1_infinity",
    "cos_tilt_2_infinity": "cos_tilt_2_infinity",
    "cos_tilt_1_infinity_only_prec_avg": "cos_tilt_1_infinity_only_prec_avg",
    "cos_tilt_2_infinity_only_prec_avg": "cos_tilt_2_infinity_only_prec_avg",
    "spin_1z": "spin_1z",
    "spin_2z": "spin_2z",
    "chi_p_2spin": "chi_p_2spin",
    "peak_luminosity": "peak_luminosity",
    "peak_luminosity_non_evolved": "peak_luminosity_non_evolved",
    "final_mass": "final_mass",
    "final_mass_non_evolved": "final_mass_non_evolved",
    "final_spin": "final_spin",
    "final_spin_non_evolved": "final_spin_non_evolved",
    "radiated_energy": "radiated_energy",
    "radiated_energy_non_evolved": "radiated_energy_non_evolved",
    "weights": "weights",
    "psi_J": "psi_J",
    "polarization_J": "psi_J",
    "opening_angle": "beta",
    "beta0": "beta",
    "rho_p": "network_precessing_snr",
    "final_kick": "final_kick",
    "tidal_disruption_frequency": "tidal_disruption_frequency",
    "tidal_disruption_frequency_ratio": "tidal_disruption_frequency_ratio",
    "220_quasinormal_mode_frequency": "220_quasinormal_mode_frequency",
    "baryonic_torus_mass": "baryonic_torus_mass",
    "baryonic_torus_mass_source": "baryonic_torus_mass_source",
    "compactness_1": "compactness_1",
    "compactness_2": "compactness_2",
    "baryonic_mass_1": "baryonic_mass_1",
    "baryonic_mass_1_source": "baryonic_mass_1_source",
    "baryonic_mass_2": "baryonic_mass_2",
    "baryonic_mass_2_source": "baryonic_mass_2_source"
}


for detector in _IFOS:
    pesummary_map["{}_matched_filter_snr".format(detector)] = (
        "{}_matched_filter_snr".format(detector)
    )
    pesummary_map["{}_matched_filter_snr_abs".format(detector)] = (
        "{}_matched_filter_snr_abs".format(detector)
    )
    pesummary_map["{}_matched_filter_snr_angle".format(detector)] = (
        "{}_matched_filter_snr_angle".format(detector)
    )
    pesummary_map["{}_optimal_snr".format(detector)] = (
        "{}_optimal_snr".format(detector)
    )


other_map = {
    "logL": "log_likelihood",
    "lnL": "log_likelihood",
    "loglr": "log_likelihood",
    "tilt_spin1": "tilt_1",
    "theta_1l": "tilt_1",
    "tilt_spin2": "tilt_2",
    "theta_2l": "tilt_2",
    "chirpmass_source": "chirp_mass_source",
    "chirp_mass_source": "chirp_mass_source",
    "mass1": "mass_1",
    "m1_detector_frame_Msun": "mass_1",
    "m2_detector_frame_Msun": "mass_2",
    "mass2": "mass_2",
    "rightascension": "ra",
    "right_ascension": "ra",
    "longitude": "ra",
    "declination": "dec",
    "latitude": "dec",
    "incl": "iota",
    "inclination": "iota",
    "phi_1l": "phi_1",
    "phi_2l": "phi_2",
    "polarisation": "psi",
    "polarization": "psi",
    "phijl": "phi_jl",
    "a_spin1": "a_1",
    "spin1": "a_1",
    "spin1_a": "a_1",
    "a1x": "spin_1x",
    "a1y": "spin_1y",
    "spin1x": "spin_1x",
    "spin1y": "spin_1y",
    "spin1z": "spin_1z",
    "a_spin2": "a_2",
    "spin2": "a_2",
    "spin2_a": "a_2",
    "a2x": "spin_2x",
    "a2y": "spin_2y",
    "spin2x": "spin_2x",
    "spin2y": "spin_2y",
    "spin2z": "spin_2z",
    "theta1": "tilt_1",
    "theta2": "tilt_2",
    "phiorb": "phase",
    "phi0": "phase",
    "distance": "luminosity_distance",
    "luminosity_distance_Mpc": "luminosity_distance",
    "chirpmass": "chirp_mass",
    "tc": "geocent_time",
    "geocent_end_time": "geocent_time",
    "fref": "reference_frequency",
    "time_maxl": "marginalized_geocent_time",
    "tref": "marginalized_geocent_time",
    "phase_maxl": "marginalized_phase",
    "distance_maxl": "marginalized_distance",
    "spin1_azimuthal": "a_1_azimuthal",
    "spin1_polar": "a_1_polar",
    "spin2_azimuthal": "a_2_azimuthal",
    "spin2_polar": "a_2_polar",
    "delta_lambda_tilde": "delta_lambda",
    "logPrior": "log_prior",
    "weight": "weights",
    "delta_lambda": "delta_lambda",
    "peak_luminosity": "peak_luminosity",
    "final_mass": "final_mass",
    "final_spin": "final_spin",
    "weights": "weights",
    "inverted_mass_ratio": "inverted_mass_ratio",
    "mf": "final_mass",
    "mf_evol": "final_mass",
    "mf_source_evol": "final_mass_source",
    "af": "final_spin",
    "af_evol": "final_spin",
    "l_peak": "peak_luminosity",
    "l_peak_evol": "peak_luminosity",
    "e_rad_evol": "radiated_energy",
}


for detector in _IFOS:
    other_map["{}_cplx_snr_arg".format(detector)] = (
        "{}_matched_filter_snr_angle".format(detector)
    )
    other_map["{}_cplx_snr_amp".format(detector)] = (
        "{}_matched_filter_abs_snr".format(detector)
    )
    other_map["{}_matched_filter_abs_snr".format(detector)] = (
        "{}_matched_filter_abs_snr".format(detector)
    )
    other_map["{}_matched_filter_snr_amp".format(detector)] = (
        "{}_matched_filter_abs_snr".format(detector)
    )
    other_map["{}_matched_filter_snr".format(detector.lower())] = (
        "{}_matched_filter_snr".format(detector)
    )
    other_map["{}_matched_filter_snr".format(detector)] = (
        "{}_matched_filter_snr".format(detector)
    )
    other_map["{}_matched_filter_snr_abs".format(detector)] = (
        "{}_matched_filter_snr_abs".format(detector)
    )
    other_map["{}_matched_filter_snr_angle".format(detector)] = (
        "{}_matched_filter_snr_angle".format(detector)
    )


standard_names = {}
standard_names.update(lalinference_map)
standard_names.update(bilby_map)
standard_names.update(pycbc_map)
standard_names.update(other_map)

descriptive_names = {
    "log_likelihood": (
        "the logarithm of the likelihood"
    ),
    "tilt_1": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the primary spin, S1"
    ),
    "tilt_2": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the secondary spin, S2"
    ),
    "tilt_1_infinity_only_prec_avg": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the primary spin, S1, defined at infinite binary separation computed "
        "using only the precession-averaged approximation"
    ),
    "tilt_2_infinity_only_prec_avg": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the secondary spin, S2, defined at infinite binary separation computed "
        "using only the precession-averaged approximation"
    ),
    "tilt_1_infinity": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the primary spin, S1, defined at infinite binary separation"
    ),
    "tilt_2_infinity": (
        "the zenith angle between the Newtonian orbital angular momentum, L, and "
        "the secondary spin, S2, defined at infinite binary separation"
    ),
    "cos_tilt_1": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the primary spin, S1"
    ),
    "cos_tilt_2": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the secondary spin, S2"
    ),
    "cos_tilt_1_infinity": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the primary spin, S1, defined at infinite binary separation"
    ),
    "cos_tilt_2_infinity": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the secondary spin, S2, defined at infinite binary separation"
    ),
    "cos_tilt_1_infinity_only_prec_avg": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the primary spin, S1, defined at infinite binary separation "
        "computed using only the precession-averaged approximation"
    ),
    "cos_tilt_2_infinity_only_prec_avg": (
        "the cosine of the zenith angle between the Newtonian orbital angular momentum "
        "momentum, L, and the secondary spin, S2, defined at infinite binary separation "
        "computed using only the precession-averaged approximation"
    ),
    "beta": (
        "the zenith angle between the total orbital angular momentum, L, and "
        "the total angular momentum J. For a non-precessing system, beta is "
        "zero by definition"
    ),
    "redshift": (
        "the redshift depending on specified cosmology"
    ),
    "network_optimal_snr": (
        "the optimal signal to noise ratio in the gravitational wave detector "
        "network"
    ),
    "network_matched_filter_snr": (
        "the matched filter signal to noise ratio in the gravitational wave "
        "detector network"
    ),
    "chirp_mass_source": (
        "the source-frame chirp mass"
    ),
    "symmetric_mass_ratio": (
        "a definition of mass ratio which is independent of the identity of "
        "the primary/secondary object"
    ),
    "mass_1": (
        "the detector-frame (redshifted) mass of the heavier object"
    ),
    "mass_2": (
        "the detector-frame (redshifted) mass of the lighter object"
    ),
    "ra": (
        "the right ascension of the source"
    ),
    "dec": (
        "the declination of the source"
    ),
    "iota": (
        "the angle between the total orbital angular momentum, L, and the "
        "line of sight, N"
    ),
    "cos_iota": (
        "the cosine of the angle between the total orbital angular momentum, L "
        ", and the line of sight, N"
    ),
    "mass_2_source": (
        "the source mass of the lighter object in the binary"
    ),
    "mass_1_source": (
        "the source mass of the heavier object in the binary"
    ),
    "phi_1": (
        "the azimuthal angle of the spin vector of the primary object"
    ),
    "phi_2": (
        "the azimuthal angle of the spin vector of the secondary object"
    ),
    "psi": (
        "the polarization angle of the source"
    ),
    "phi_12": (
        "the difference between the azimuthal angles of the individual spin "
        "vectors of the primary and secondary object's"
    ),
    "phi_jl": (
        "the difference between total and orbital angular momentum azimuthal "
        "angles"
    ),
    "a_1": (
        "the dimensionless spin magnitude of the primary object"
    ),
    "spin_1x": (
        "the x-component of the primary object's spin in Euclidean coordinates"
    ),
    "spin_1y": (
        "the y-component of the primary object's spin in Euclidean coordinates"
    ),
    "spin_1z": (
        "the z-component of the primary object's spin in Euclidean coordinates"
    ),
    "spin_1z_infinity": (
        "the z-component of the primary object's spin in Euclidean coordinates "
        "defined at infinite binary separation"
    ),
    "spin_1z_infinity_only_prec_avg": (
        "the z-component of the primary object's spin in Euclidean coordinates "
        "defined at infinite binary separation computed using only the "
        "precession-averaged approximation"
    ),
    "a_2": (
        "the dimensionless spin magnitude of the secondary object"
    ),
    "spin_2x": (
        "the x-component of the secondary object's spin in Euclidean "
        "coordinates"
    ),
    "spin_2y": (
        "the y-component of the secondary object's spin in Euclidean "
        "coordinates"
    ),
    "spin_2z": (
        "the z-component of the secondary object's spin in Euclidean "
        "coordinates"
    ),
    "spin_2z_infinity": (
        "the z-component of the secondary object's spin in Euclidean coordinates "
        "defined at infinite binary separation"
    ),
    "spin_2z_infinity_only_prec_avg": (
        "the z-component of the secondary object's spin in Euclidean coordinates "
        "defined at infinite binary separation computed using only the "
        "precession-averaged approximation"
    ),
    "chi_p": (
        "the effective precession spin parameter"
    ),
    "chi_p_infinity": (
        "the effective precession spin parameter defined at infinite binary separation"
    ),
    "chi_p_infinity_only_prec_avg": (
        "the effective precession spin parameter defined at infinite binary separation "
        "computed using only the precession-averaged approximation"
    ),
    "chi_p_2spin": (
        "a modified effective precession spin parameter accounting for "
        "precessing spin information from both compact objects."
    ),
    "phase": (
        "the binary phase defined at a given reference frequency"
    ),
    "luminosity_distance": (
        "the luminosity distance of the source"
    ),
    "chirp_mass": (
        "the detector-frame chirp mass"
    ),
    "chi_eff": (
        "the effective inspiral spin parameter"
    ),
    "chi_eff_infinity": (
        "the effective inspiral spin parameter defined at infinite binary separation"
    ),
    "chi_eff_infinity_only_prec_avg": (
        "the effective inspiral spin parameter defined at infinite binary separation "
        "computed using only the precession-averaged approximation"
    ),
    "total_mass_source": (
        "the source-frame combined mass of the primary and secondary masses "
    ),
    "total_mass": (
        "the detector-frame combined mass of the primary and secondary masses "
    ),
    "mass_ratio": (
        "the ratio of the binary component masses. We use the convention that "
        "the mass ratio is always less than 1"
    ),
    "inverted_mass_ratio": (
        "The inverted ratio of the binary component masses. Note that normal "
        "convention is mass ratio less than 1, but here the inverted mass ratio "
        "is always bigger than 1"
    ),
    "geocent_time": (
        "the GPS merger time at the geocenter"
    ),
    "theta_jn": (
        "the angle between the total angular momentum, J, and the line of "
        "sight, N"
    ),
    "cos_theta_jn": (
        "the cosine of the angle between the total angular momentum, J, and "
        "the line of sight, N"
    ),
    "reference_frequency": (
        "the frequency at which the frequency dependent parameters are defined"
    ),
    "a_1_azimuthal": (
        "the azimuthal spin angle of the primary object"
    ),
    "a_1_polar": (
        "the polar spin angle of the primary object"
    ),
    "a_2_azimuthal": (
        "the azimuthal spin angle of the secondary object"
    ),
    "a_2_polar": (
        "the polar spin angle of the secondary object"
    ),
    "lambda_1": (
        "the dimensionless tidal deformability of the primary object"
    ),
    "lambda_2": (
        "the dimensionless tidal deformability of the secondary object"
    ),
    "lambda_tilde": (
        "the combined dimensionless tidal deformability"
    ),
    "delta_lambda": (
        "the relative difference in the combined tidal deformability"
    ),
    "log_pressure": (
        "the base 10 logarithm of the pressure in Pa at the reference density "
        "of 10^17.7 kg/m^3"
    ),
    "gamma_1": (
        "the adiabatic index for densities below 10^17.7 kg/m^3"
    ),
    "gamma_2": (
        "the adiabatic index for densities from 10^17.7 kg/m^3 to 10^18 kg/m^3"
    ),
    "gamma_3": (
        "the adiabatic index for densities above 10^18 kg/m^3"
    ),
    "spectral_decomposition_gamma_0": (
        "the 0th expansion coefficient of the spectrally decomposed adiabatic "
        "index of the EOS"
    ),
    "spectral_decomposition_gamma_1": (
        "the 1st expansion coefficient of the spectrally decomposed adiabatic "
        "index of the EOS"
    ),
    "spectral_decomposition_gamma_2": (
        "the 2nd expansion coefficient of the spectrally decomposed adiabatic "
        "index of the EOS"
    ),
    "spectral_decomposition_gamma_3": (
        "the 3rd expansion coefficient of the spectrally decomposed adiabatic "
        "index of the EOS"
    ),
    "peak_luminosity": (
        "the peak gravitational wave luminosity estimated using the spins "
        "evolved to the ISCO frequency"
    ),
    "peak_luminosity_non_evolved": (
        "the peak gravitational wave luminosity estimated using the spins "
        "defined at the reference frequency"
    ),
    "final_mass": (
        "the detector-frame remnant mass estimated using the spins evolved to "
        "the ISCO frequency"
    ),
    "final_mass_source": (
        "the source-frame remnant mass estimated using the spins evolved to "
        "the ISCO frequency"
    ),
    "final_mass_non_evolved": (
        "the detector-frame remnant mass estimated using the spins defined at "
        "the reference frequency"
    ),
    "final_mass_source_non_evolved": (
        "the source-frame remnant mass estimated using the spins defined at "
        "the reference frequency"
    ),
    "final_spin": (
        "the spin of the remnant object estimated using the spins evolved to "
        "the ISCO frequency"
    ),
    "final_spin_non_evolved": (
        "the spin of the remnant object estimated using the spins defined at "
        "the reference frequency"
    ),
    "radiated_energy": (
        "the energy radiated in gravitational waves. Defined as the difference "
        "between the source total and source remnant mass. The source remnant "
        "mass was estimated using the spins evolved at the ISCO frequency"
    ),
    "radiated_energy_non_evolved": (
        "the energy radiated in gravitational waves. Defined as the difference "
        "between the source total and source remant mass. The source remnant "
        "mass was estimated using the spins defined at the reference frequency"
    ),
    "tidal_disruption_frequency": (
        "the gravitational wave detector-frame frequency at which tidal forces "
        "dominate over the self-gravity forces, invoking mass shedding"
    ),
    "tidal_disruption_frequency_ratio": (
        "the ratio of the tidal disruption and the 220 quasinormal mode "
        "frequency of the system. In NSBH models this ratio describes whether the "
        "system is disruptive or non-disruptive. If the ratio is less than 1, the "
        "system is characterised as either mildly disruptive or disruptive. If the ratio "
        "is greater than 1, the system is characterised as non-disruptive meaning "
        "the secondary object remains intact as it plunges into the primary."
    ),
    "220_quasinormal_mode_frequency": (
        "the detector-frame 220 quasinormal mode (QNM) frequency of the "
        "remnant object"
    ),
    "baryonic_torus_mass": (
        "the detector-frame (redshifted) baryonic mass of the torus formed "
        "around the primary object. If the baryonic torus mass is 0, the system "
        "is characterised as either mildly disruptive or non-disruptive."
    ),
    "baryonic_torus_mass_source": (
        "the source-frame baryonic mass of the torus formed around the primary "
        "object"
    ),
    "compactness_1": "the compactness of the primary object",
    "compactness_2": "the compactness of the secondary object",
    "baryonic_mass_1": (
        "the detector-frame (redshifted) baryonic mass of the primary object"
    ),
    "baryonic_mass_1_source": (
        "the source-frame baryonic mass of the primary object"
    ),
    "baryonic_mass_2": (
        "the detector-frame (redshifted) baryonic mass of the secondary object"
    ),
    "baryonic_mass_2_source": (
        "the source-frame baryonic mass of the secondary object"
    ),
}

for detector in _IFOS:
    descriptive_names["{}_optimal_snr".format(detector)] = (
        "the optimal signal to noise ratio in the %s gravitational wave "
        "detector" % (detector)
    )
    descriptive_names["{}_matched_filter_snr".format(detector)] = (
        "the real component of the complex matched filter signal to noise "
        "ratio in the %s gravitational wave detector" % (detector)
    )
    descriptive_names["{}_matched_filter_abs_snr".format(detector)] = (
        "the absolute value of the complex matched filter signal to noise "
        "ratio in the %s gravitational wave detector" % (detector)
    )
    descriptive_names["{}_matched_filter_snr_abs".format(detector)] = (
        "the absolute value of the complex matched filter signal to noise "
        "ratio in the %s gravitational wave detector" % (detector)
    )
    descriptive_names["{}_matched_filter_snr_angle".format(detector)] = (
        "the angle of the complex component of the matched filter signal to "
        "noise ratio in the %s gravitational wave detector" % (detector)
    )
    descriptive_names["{}_time".format(detector)] = (
        "the GPS merger time at the %s gravitational wave detector" % (detector)
    )
