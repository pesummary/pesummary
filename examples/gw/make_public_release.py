#! /usr/bin/env python

from pesummary.gw.notebook import make_public_notebook
from pesummary.gw.fetch import fetch_open_samples

file_name = fetch_open_samples(
    "GW190814", catalog="GWTC-2", read_file=False, delete_on_exit=False,
    outdir=".", unpack=True, path="GW190814.h5"
)
make_public_notebook(
    file_name, (
        "GW190814: Gravitational Waves from the Coalescence of a 23 Msun Black "
        "Hole with a 2.6 Msun Compact Object"
    ), default_analysis="combined", default_parameter="mass_2_source"
)
