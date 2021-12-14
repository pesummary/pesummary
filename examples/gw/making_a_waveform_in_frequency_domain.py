from pesummary.gw.fetch import fetch_open_samples
import matplotlib.pyplot as plt
import requests

# First we download and read the publically available posterior samples
f = fetch_open_samples(
    "GW190814", catalog="GWTC-2", unpack=True, path="GW190814.h5"
)
samples = f.samples_dict
Phenom = samples["C01:IMRPhenomPv3HM"]

# Next we generate the plus and cross polarizations in the frequency domain for
# the maximum likelihood sample
approximant = "IMRPhenomPv3HM"
delta_f = 1. / 256
f_low = 20.
f_high = 1024.
wvfs = Phenom.maxL_fd_waveform(approximant, delta_f, f_low, f_high, f_ref=f_low)

# Alternatively, we may generate the plus and cross polarizations in the
# frequency domain for a specific sample with the ind kwarg
ind = 100
wvfs = Phenom.fd_waveform(approximant, delta_f, f_low, f_high, f_ref=f_low, ind=ind)

# It is often useful to know what the strain is at a given gravitational wave
# detector. This involves calculating the antenna response function. If we
# wanted to generate the maximum likelihood strain projected onto the LIGO
# Livingtson detector, we may simply run
ht = Phenom.maxL_fd_waveform(approximant, delta_f, f_low, f_high, f_ref=f_low, project="L1")

# In all cases, a `gwpy.frequencyseries.FrequencySeries` object is returned.
# This can therefore easily be plotted with
fig = plt.figure()
plt.plot(ht.frequencies, ht)
plt.savefig("GW190814_IMRPhenomPv3HM_fd.png")
