import numpy as np
import pkg_resources
import os

# matplotlib style file
_path = pkg_resources.resource_filename("pesummary", "conf")
style_file = os.path.join(_path, "matplotlib_rcparams.sty")

# checkpoint file
checkpoint_dir = lambda webdir: os.path.join(webdir, "checkpoint")
resume_file = "pesummary_resume.pickle"

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

# Minimum length of h5 dataset for compression. Compressing small datasets can
# lead to an increased file size
compression_min_length = 1

# Plot 1d kdes rather than 1d histograms
kde_plot = False

# color for non-comparison plots
color = 'b'

# color cycle for different mcmc chains
colorcycle = "brgkmc"

# color cycle for different cmaps
cmapcycle = ["YlOrBr", "Blues", "Purples", "Greens", "PuRd", "inferno"]

# color for injection lines
injection_color = 'orange'

# color for prior histograms
prior_color = 'k'

# Produce public facing summarypages
public = False

# Number of cores to run on
multi_process = 1

# Default f_low to use for GW specific conversions
default_flow = 20.0

# Default f_final to use for GW specific conversions
default_f_final = 1024.0

# Default delta_f to use for GW specific conversions
default_delta_f = 1. / 256

# Standard meta_data names
log_evidence = "ln_evidence"
evidence = "evidence"
log_evidence_error = "ln_evidence_error"
log_bayes_factor = "ln_bayes_factor"
bayes_factor = "bayes_factor"
log_noise_evidence = "ln_noise_evidence"
log_prior_volume = "ln_prior_volume"

# corner.corner colors
corner_colors = ['#0072C1', '#b30909', '#8809b3', '#b37a09']

# corner.corner default kwargs
corner_kwargs = dict(
    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color=corner_colors[0],
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
    "delta_lambda", "lambda_tilde", "log_likelihood"
]

# Parameters to use for GW source frame corner plot
gw_source_frame_corner_parameters = [
    "luminosity_distance", "mass_1_source", "mass_2_source",
    "total_mass_source", "chirp_mass_source", "redshift"
]

# List of precessing angles
precessing_angles = [
    "cos_tilt_1", "cos_tilt_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"
]
# Parameters to use for GW extrinsic corner plot
gw_extrinsic_corner_parameters = ["luminosity_distance", "psi", "ra", "dec"]

# Cosmology to use when calculating redshift
cosmology = "Planck15"

# Analytic PSD to use for conversions when no PSD file is provided
psd = "aLIGOZeroDetHighPower"

# GraceDB service url to use
gracedb_server = "https://gracedb.ligo.org/api/"

# Information required for reproducing a GW analysis
gw_reproducibility = ["config", "psd"]

# Additional 1d histogram pages that combine multiple GW marginalized posterior
# distributions
additional_1d_pages = {
    "precession": ["chi_p", "chi_p_2spin", "network_precessing_snr", "beta"]
}
