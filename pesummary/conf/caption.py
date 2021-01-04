# Caption for 1d histogram
caption_1d_histogram = (
    "Plot showing the marginalized posterior distribution for {}. The vertical "
    "dashed lines show the 90% credible interval. The title gives the median "
    "and 90% confidence interval."
)

# Caption for bootstrapped 1d histogram
caption_1d_histogram_bootstrap = (
    "Plot showing {} posterior realizations for {}. {} samples were randomly "
    "drawn for each test."
)

# Caption for 2d contour plot
caption_2d_contour = (
    "Plot showing the 2d posterior distribution for {} and {}. The contours "
    "show the 50% and 90% credible intervals."
)

# Caption for autocorrelation plot
caption_autocorrelation = (
    "Plot showing the autocorrelation function for {}. This plot is commonly "
    "used to check for randomness in the data. An autocorrelation of 1/-1 "
    "means that the samples are correlated and 0 means no correlation. The "
    "autocorrelation function at a lag of N gives the correlation between "
    "the 0th sample and the Nth sample."
)

# Caption for sample evolution plot
caption_sample_evolution = (
    "Scatter plot showing the evolution of the collected samples for {}."
)

# Caption for sample evolution plot colored by second variable
caption_sample_evolution_colored = (
    caption_sample_evolution[:-1] + " colored by {}."
)

# Caption for preliminary skymap
caption_skymap_preliminary = (
    "Plot showing the most likely position of the source. This map has been "
    "produced by creating a 2d histogram from the samples of 'ra' and 'dec'. "
    "This is a valid approximation near the equator but breaks down near the "
    "poles. The 50% and 90% credible intervals are approximate. For true "
    "50% and 90% credible intervals a 'ligo.skymap' skymap should be "
    "generated."
)

# Caption for ligo.skymap skymap
caption_skymap = (
    "Plot showing the most likely position of the source that generated the "
    "gravitational wave. We give the 50% and 90% credible intervals. The "
    "black region corresponds to the most likely position and light orange "
    "least likely."
)

# Caption for strain plot
caption_strain = (
    "Plot showing the comparison between the gravitational wave strain "
    "measured at the detectors (grey) and the maximum likelihood waveform "
    "(orange) for all available detectors. A 30Hz high and 300Hz low pass "
    "filter have been applied to the gravitational wave strain."
)

# Caption for frequency domain waveform
caption_frequency_waveform = (
    "Plot showing the frequency domain waveform generated from the maximum "
    "likelihood samples."
)

# Caption for comparison time domain waveform plot
caption_time_waveform = (
    "Plot showing the time domain waveform generated from the maximum "
    "likelihood samples."
)

# Caption for psd plot
caption_psd = (
    "Plot showing the instrument noise for each detector near the time of "
    "detection; the power spectral density is expressed in terms of "
    "equivalent gravitational-wave strain amplitude."
)

# Caption for calibration plot
caption_calibration = (
    "Plot showing the calibration posteriors (solid lines) and calibration "
    "priors (solid band) for amplitude (top) and phase (bottom)."
)

# Caption for default mass1 mass2 classification plot
caption_default_classification_mass_1_mass_2 = (
    "Scatter plot showing the individual samples with their classifications "
    "over the mass_1 and mass_2 parameter space. The samples have not been "
    "reweighted. Green regions correspond to BBHs, blue BNS, orange NSBH and "
    "red MassGap."
)

# Caption for population mass1 mass2 classification plot
caption_population_classification_mass_1_mass_2 = (
    "Scatter plot showing the individual samples with their classifications "
    "over the mass_1 and mass_2 parameter space. The samples have been "
    "reweighted to a population prior. Green regions correspond to BBHs, blue "
    "BNS, orange NSBH and red MassGap."
)

# Caption for default classification bar plot
caption_default_classification_bar = (
    "Bar plot showing the most likely classification of the binary based on "
    "the samples."
)

# Caption for population classification bar plot
caption_population_classification_bar = (
    "Bar plot showing the most likely classification of the binary based on "
    "the samples which have been reweighted to a population prior."
)

# Caption for 2d contour plots
caption_2d_contour = (
    "2d contour plot showing the bounded posterior distributions for {} and "
    "{}. Each contour shows the 90% credible interval and the posterior "
    "distributions are normalized."
)

# Caption for violin plots
caption_violin = (
    "Violin plot showing the posterior distributions for {}. The horizontal "
    "black lines show the 90% credible intervals."
)

# Caption for spin disk plots
caption_spin_disk = (
    "Posterior probability distributions for the dimensionless component spins "
    "relative to the normal to the orbital plane, L, marginalized over the  "
    "azimuthal angles. The bins are constructed linearly in spin magnitude and "
    "the cosine of the tilt angles and are assigned equal prior probability."
)
