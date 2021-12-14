# You are able to use PESummary to generate latex tables and/or latex macros
# for a given result file. This can be done by first loading in the result
# file and then using the `to_latex_table` and the `generate_latex_macros`
# functions.

from pesummary.gw.fetch import fetch_open_samples

f = fetch_open_samples(
    "GW190814", catalog="GWTC-2", unpack=True, path="GW190814.h5"
)

# Then make a dictionary which maps the parameter to a description that you
# wish to use in the latex table
parameter_dict = {
    "mass_1": "Detector-frame primary mass $m_{1}/M_{\odot}$",
    "mass_2": "Detector-frame secondary mass $m_{2}/M_{\odot}$",
}

# As a single PESummary metafile can contain many runs, we need to specify which
# analysis we want included in the latex table. This can be done by the
# following:

f.to_latex_table(
    parameter_dict=parameter_dict, labels=f.labels[0],
    save_to_file="pesummary_latex_table.tex"
)

# If we wanted to include more than one run in the table, this can be done
# by simply passing a list of labels that you wish to include

f.to_latex_table(
    parameter_dict=parameter_dict, labels=f.labels[:2],
    save_to_file="pesummary_latex_table_multiple.tex"
)

# To generate latex macros, we need to generate a similar dictionary, but this
# time we want to map the parameter to a given latex macro name
parameter_dict = {
    "mass_1": "detector_primary",
    "mass_2": "detector_secondary"
}

# If you want to generate macros for more than one run, this can be done by
# again passing a list of labels that you wish to include

f.generate_latex_macros(
    parameter_dict=parameter_dict, labels=f.labels[:2],
    save_to_file="pesummary_latex_macros.tex"
)
