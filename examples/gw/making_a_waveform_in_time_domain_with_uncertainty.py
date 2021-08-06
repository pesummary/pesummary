from pesummary.gw.fetch import fetch_open_samples
import matplotlib.pyplot as plt
import requests

f = fetch_open_samples("GW190814", unpack=True, path="GW190814.h5")
samples = f.samples_dict
EOB = samples["C01:SEOBNRv4PHM"]
_ = EOB.downsample(1000)
approximant = "SEOBNRv4PHM"
delta_t = 1. / 4096
f_low = 20.
ht, upper, lower, bound_times = EOB.maxL_td_waveform(
    approximant, delta_t, f_low, f_ref=f_low, project="L1", level=[0.68, 0.95],
    multi_process=4
)
fig = plt.figure()
plt.plot(ht.times, ht)
plt.fill_between(bound_times, upper[0], lower[0], color='r', alpha=0.4)
plt.fill_between(bound_times, upper[1], lower[1], color='r', alpha=0.2)
plt.xlim(1249852257.01 - 0.1, 1249852257.01 + 0.04)
plt.savefig("GW190814_SEOBNRv4PHM_td_with_uncertainty.png")
