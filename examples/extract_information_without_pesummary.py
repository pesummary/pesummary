# Below we show how to extract information stored in the PESummary metafile
# using the core python libraries

import json
import h5py
import numpy as np

# If the metafile is of JSON format, we require the `json` package
with open("posterior_samples.json", "r") as f:
    data = json.load(f)

# Otherwise, we need to use the `h5py` package
# data = h5py.File("posterior_sample.h5", "r")

labels = list(data["posterior_samples"].keys())

# Let us assume that you wish to extract the data for the first label in the
# list

label = labels[0]
dictionary = data[label]

# Now let us assume that you wish to extract the posterior samples for the
# first label in the list
posterior_samples = dictionary["posterior_samples"]
parameters = posterior_samples.dtype.names
samples = posterior_samples

# Now lets show how to extract the configuration file used in the analysis
config = dictionary["config_file"]
if config == {}:
    print("No configuration file has been stored for {}".format(labels[0]))

for i in config.keys():
    print("[{}]".format(i))
    for key, item in config[i].items():
        print("{}={}".format(key, item))
    print("\n")

# Now lets show how to extract the calibration envelope used in the analysis
calibration_envelope = dictionary["priors"]["calibration"]
if calibration_envelope == {}:
    print("No calibration envelope has been stored for {}".format(labels[0]))

IFOs = list(calibration_envelope.keys())
calibration_envelope_data = {
    i: np.array(
        [tuple(j) for j in calibration_envelope[i]], dtype=[
            ("Frequency", "f"),
            ("Median Mag", "f"),
            ("Phase (Rad)", "f"),
            ("-1 Sigma Mag", "f"),
            ("-1 Sigma Phase", "f"),
            ("+1 Sigma Mag", "f"),
            ("+1 Sigma Phase", "f")
        ]
    ) for i in IFOs
}

# Now lets show how to extract the psd used in the analysis
psd = dictionary["psds"]
if psd == {}:
    print("No psd data has been stored for {}".format(labels[0]))

IFOs = list(psd.keys())
psd_data = {
    i: np.array(
        [tuple(j) for j in psd[i]], dtype=[
            ("Frequency", "f"),
            ("Strain", "f")
        ]
    ) for i in IFOs
}
