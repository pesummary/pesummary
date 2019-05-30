# Let us assume you have run PESummary for both a standard results file and
# a gravitational wave specific results file. For clarity we will call the
# pesummary metafile produced from the standard results file `core.json` and the
# gravitational wave specific results file `gw.json`

# First let us extract the information from the `core.json` file
from pesummary.core.file.existing import ExistingFile

f = ExistingFile("./core.json")
config_data = f.existing_config
samples = f.existing_samples
parameters = f.existing_parameters
labels = f.existing_labels
injection_values = f.existing_injection
injection_data = {i: j for i, j in zip(parameters, injection_values)}

# If you want to, you are able to write the config_data stored in the metafile
# to a valid configuration file. This is done by using the
# `ExistingFile.write_config_to_file` function. This function takes 1 argument
# and 1 optional argument. Run `help(ExistingFile.write_config_to_file)` to
# learn about these arguments

f.write_config_to_file(labels[0], outdir="./outdir")

# Now let us extract the information from the `gw.json` file
from pesummary.gw.file.existing import GWExistingFile

f = GWExistingFile("./gw.json")
config_data = f.existing_config
samples = f.existing_samples
parameters = f.existing_parameters
labels = f.existing_labels
injection_values = f.existing_injection
injection_data = {i: j for i, j in zip(parameters, injection_values)}

calibration_envelopes = f.existing_calibration
psds = f.existing_psds
approximants = f.existing_approximants

# As GWExistingFile is inherited from ExistingFile, all the same functions
# can be used for the GWExistingFile. For instance, you are able to save the
# config_data as a valid configuration file by running

f.write_config_to_file(labels[0], outdir="./outdir")
