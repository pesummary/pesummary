from pesummary.gw.fetch import fetch_open_samples
from pesummary.utils.utils import draw_conditioned_prior_samples
import matplotlib.pyplot as plt
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
import numpy as np

np.random.seed(21)

# First lets download and read the publicly available data
f = fetch_open_samples(
    "GW190412", catalog="GWTC-2", unpack=True, path="GW190412.h5"
)
posterior = f.samples_dict
prior = f.priors["samples"]["C01:SEOBNRv4PHM"]
unconditioned = prior["chi_p"]
# Next let us generate the prior samples conditioned on the chi_eff posterior
# distribution
parameters_to_condition_on = ["chi_eff"]
boundaries = {
    "low": {"chi_p": 0.0, "chi_eff": -1.0},
    "high": {"chi_p": 1.0, "chi_eff": 1.0}
}
N_bins = 100
conditioned = draw_conditioned_prior_samples(
    posterior["PublicationSamples"], prior, parameters_to_condition_on,
    boundaries["low"], boundaries["high"], N_bins
)

# Next let us plot the data
fig = plt.figure()
xsmooth = np.linspace(0.01, 0.99, 100)
samples = [
    posterior["C01:SEOBNRv4PHM"]["chi_p"], posterior["C01:IMRPhenomPv3HM"]["chi_p"],
    unconditioned, conditioned["chi_p"]
]
colors = ["#e69f00", "#0072b2", "grey", "k"]
labels = ["EOBNR PHM", "Phenom PHM", "global", "restricted"]
linestyles = ["-", "-", ":", ":"]
for _samples, c, l, label in zip(samples, colors, linestyles, labels):
    kde = bounded_1d_kde(
        _samples, method="Transform", xlow=0.0, xhigh=1.0, smooth=2.0
    )
    plt.plot(*kde(xsmooth), label=label, linestyle=l, color=c)

plt.xlabel(r"$\chi_{p}$", fontsize=16)
plt.ylabel(r"Probability Density", fontsize=16)
plt.legend()
plt.show()
