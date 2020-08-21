# let us extract the information from the GW specific pesummary metafile
from pesummary.io import read
import requests

data = requests.get(
    "https://dcc.ligo.org/public/0168/P2000183/008/GW190814_posterior_samples.h5"
)
with open("GW190814_posterior_samples.h5", "wb") as f:
    f.write(data.content)

f = read("GW190814_posterior_samples.h5", package="gw")
config_data = f.config
samples = f.samples_dict
parameters = f.parameters
labels = f.labels
priors = f.priors
injection_values = f.injection_parameters

label_index = 0
injection_data = injection_values[label_index]

calibration_envelopes = f.calibration
psds = f.psd
approximants = f.approximant

# If you would prefer to have the psds and calibration envelopes stored
# as dat/txt files, you can easily convert them back to their original form
# by using the `save_to_file` function

psds[labels[0]]["H1"].save_to_file("IFO0_psd.dat")
calibration_envelopes[labels[0]]["H1"].save_to_file("calibration_H1.txt")

# We can also save the config_data as a valid configuration file by running

f.write_config_to_file(labels[0], outdir="./")
