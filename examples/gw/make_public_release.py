#! /usr/bin/env python

from pesummary.gw.notebook import make_public_notebook
import requests

data = requests.get(
    "https://dcc.ligo.org/public/0168/P2000183/008/GW190814_posterior_samples.h5"
)
with open("GW190814_posterior_samples.h5", "wb") as f:
    f.write(data.content)

make_public_notebook(
    "GW190814_posterior_samples.h5", (
        "GW190814: Gravitational Waves from the Coalescence of a 23 Msun Black "
        "Hole with a 2.6 Msun Compact Object"
    ), default_analysis="combined", default_parameter="mass_2_source"
)
