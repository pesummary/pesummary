from pesummary.io import read
import matplotlib.pyplot as plt
import requests

# First we download publically available posterior samples
data = requests.get(
    "https://dcc.ligo.org/public/0168/P2000183/008/GW190814_posterior_samples.h5"
)
with open("GW190814_posterior_samples.h5", "wb") as f:
    f.write(data.content)

# Next we read in the posterior samples and select the samples we wish to plot
f = read("GW190814_posterior_samples.h5")
samples = f.samples_dict
EOB = samples["C01:SEOBNRv4PHM"]

# Next we generate the plus and cross polarizations in the time domain for the
# maximum likelihood sample
approximant = "SEOBNRv4PHM"
delta_t = 1. / 4096
f_low = 20.
wvfs = EOB.maxL_td_waveform(approximant, delta_t, f_low, f_ref=f_low)

# Alternatively, we may generate the plus and cross polarizations in the time
# domain for a specific sample with the ind kwarg
ind = 100
wvfs = EOB.td_waveform(approximant, delta_t, f_low, f_ref=f_low, ind=ind)

# It is often useful to know what the strain is at a given gravitational wave
# detector. This involves calculating the antenna response function. If we
# wanted to generate the maximum likelihood strain projected onto the LIGO
# Livingtson detector, we may simply run
ht = EOB.maxL_td_waveform(approximant, delta_t, f_low, f_ref=f_low, project="L1")

# In all cases, a `gwpy.timeseries.TimeSeries` object is returned. This can
# therefore easily be plotted with
fig = plt.figure()
plt.plot(ht.times, ht)
plt.savefig("GW190814_SEOBNRv4PHM_td.png")
