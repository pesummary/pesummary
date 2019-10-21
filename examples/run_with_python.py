#!/usr/bin/env python

"""
This is an example script to run PESummary from within a python script for
non-gravitational wave data.
"""
from pesummary.core import command_line, inputs, finish
from pesummary.core.file import meta_file
from pesummary.cli import summarypages, summaryplots
import pesummary.cli
import json
import numpy as np

# You can either pass PESummary an existing results file or you can generate
# one here and just pass it directly to PESummary. We will use the latter in
# this example and just generate random numbers for 3 parameters
data = {"posterior": {"amplitude": list(np.random.random(100)),
                      "phase": list(np.random.random(100)),
                      "sigma": list(np.random.random(100))}}

with open("example.json", "w") as f:
    json.dump(data, f, indent=4, sort_keys=True)

# First set up your parser
parser = command_line.command_line()

# Pass PESummary a results file and a directory for where you would like
# to store the data
opts = parser.parse_args(["--samples", "example.json",
                          "--webdir", "./outdir"])

# Do low level checks of the inputs
args = inputs.Input(opts)

# Generate all plots for the given results file
summaryplots.PlotGeneration(args)

# Generate the webpages for the given results file
summarypages.WebpageGeneration(args)

# Generate the metafile
meta_file.MetaFile(args)

# Clean up the current working directory
finish.FinishingTouches(args)
