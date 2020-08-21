# Let us assume you have run PESummary for both a standard results file and
# a gravitational wave specific results file. For clarity we will call the
# pesummary metafile produced from the standard results file `core.json` and the
# gravitational wave specific results file `gw.json`

# First let us extract the information from the `core.json` file
from pesummary.core.file.read import read

f = read("./core.json")
config_data = f.config
samples = f.samples_dict
parameters = f.parameters
labels = f.labels
priors = f.priors
injection_values = f.injection_parameters
injection_data = {i: j for i, j in zip(parameters, injection_values)}

# If you want to, you are able to write the config_data stored in the metafile
# to a valid configuration file. This is done by using the
# `read.write_config_to_file` function. This function takes 1 argument
# and 1 optional argument. Run `help(read.write_config_to_file)` to
# learn about these arguments

f.write_config_to_file(labels[0], outdir="./outdir")
