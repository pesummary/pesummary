import numpy as np
import pkg_resources
import os

# matplotlib style file
_path = pkg_resources.resource_filename("pesummary", "conf")
style_file = os.path.join(_path, "matplotlib_rcparams.sty")

# Overwrite message
overwrite = "Overwriting {} from {} to {}"

# The palette to be used to distinguish result files
palette = "colorblind"

# Include the prior on the posterior plots
include_prior = False

# The user that submitted the job
user = "albert.einstein"

# The number of samples to disregard as burnin
burnin = 0

# The method to use to remove the samples as burnin
burnin_method = "burnin_by_step_number"

# delimiter to use when saving files to dat with np.savetxt
delimiter = "\t"

# Plot 1d kdes rather than 1d histograms
kde_plot = False

# color for non-comparison plots
color = 'b'

# color cycle for different mcmc chains
colorcycle = "brgkmc"

# color for injection lines
injection_color = 'orange'

# color for prior histograms
prior_color = 'k'

# Produce public facing summarypages
public = False

# Number of cores to run on
multi_process = 1

# Standard meta_data names
log_evidence = "ln_evidence"
evidence = "evidence"
log_evidence_error = "ln_evidence_error"
log_bayes_factor = "ln_bayes_factor"
bayes_factor = "bayes_factor"
log_noise_evidence = "ln_noise_evidence"
log_prior_volume = "ln_prior_volume"

# corner.corner default kwargs
corner_kwargs = dict(
    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color='#0072C1',
    truth_color='tab:orange', quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False, plot_datapoints=True, fill_contours=True,
    max_n_ticks=3
)

# Parameters to use for GW corner plot
gw_corner_parameters = [
    "luminosity_distance", "dec", "a_2", "a_1", "geocent_time", "phi_jl",
    "psi", "ra", "phase", "mass_2", "mass_1", "phi_12", "tilt_2", "iota",
    "tilt_1", "chi_p", "chirp_mass", "mass_ratio", "symmetric_mass_ratio",
    "total_mass", "chi_eff", "redshift", "mass_1_source", "mass_2_source",
    "total_mass_source", "chirp_mass_source", "lambda_1", "lambda_2",
    "delta_lambda", "lambda_tilde"
]

# Parameters to use for GW source frame corner plot
gw_source_frame_corner_parameters = [
    "luminosity_distance", "mass_1_source", "mass_2_source",
    "total_mass_source", "chirp_mass_source", "redshift"
]

# List of precessing angles
precessing_angles = [
    "tilt_1", "tilt_2", "phi_12", "phi_jl"
]
# Parameters to use for GW extrinsic corner plot
gw_extrinsic_corner_parameters = ["luminosity_distance", "psi", "ra", "dec"]

# Cosmology to use when calculating redshift
cosmology = "Planck15"
