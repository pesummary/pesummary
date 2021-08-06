# let us extract the information from the GW specific pesummary metafile
from pesummary.gw.fetch import fetch_open_samples

f = fetch_open_samples("GW190814", unpack=True, path="GW190814.h5")
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

label = "C01:IMRPhenomD"
psds[label]["H1"].save_to_file("IFO0_psd.dat")
calibration_envelopes[label]["H1"].save_to_file("calibration_H1.txt")

# We can also save the config_data as a valid configuration file by running

f.write_config_to_file(label, outdir="./")
