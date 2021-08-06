from pesummary.gw.file.strain import StrainData
from pesummary.gw.fetch import fetch_open_samples
import requests

# First we download the GW190814 posterior samples and generate the maximum
# likelihood waveform in the time domain
data = fetch_open_samples("GW190814", unpack=True, path="GW190814.h5")
samples = data.samples_dict["C01:SEOBNRv4PHM"]
maxL = samples.maxL_td_waveform("SEOBNRv4PHM", 1. / 4096, 20., project="L1")

# # Next we fetch the LIGO Livingston data around the time of GW190814
L1_data = StrainData.fetch_open_data('L1', 1249852257.01 - 20, 1249852257.01 + 5)

# Next we plot the data
fig = L1_data.plot(
    type="td", merger_time=1249852257.01, window=(-0.1, 0.04),
    template={"L1": maxL}, bandpass_frequencies=[50., 300.]
)
fig.savefig("GW190814.png")
fig.close()
