# You are able to use PESummary to generate latex tables and/or latex macros
# for a given result file. This can be done by first loading in the result
# file and then using the `to_latex_table` and the `generate_latex_macros`
# functions.

from pesummary.gw.file.read import read

# First load in the result file
f = read("result_file.dat")

# Then make a dictionary which maps the parameter to a description that you
# wish to use in the latex table
parameter_dict = {
    "mass_1": "Detector-frame primary mass $m_{1}/M_{\odot}$",
    "mass_2": "Detector-frame secondary mass $m_{2}/M_{\odot}$",
}

# Now generate the latex table and save it to a file
f.to_latex_table(
    parameter_dict=parameter_dict, save_to_file="latex_table.tex"
)

# If we had a PESummary metafile, then because a single PESummary metafile
# can contain many runs, we need to specify which analysis we want included
# in the latex table. This can be done by the following:

f.to_latex_table(
    parameter_dict=parameter_dict, labels="example",
    save_to_file="pesummary_latex_table.tex"
)

# If we wanted to include more than one run in the table, this can be done
# by simply passing a list of labels that you wish to include

f.to_latex_table(
    parameter_dict=parameter_dict, labels=["example1", "example2"],
    save_to_file="pesummary_latex_table.tex"
)

# To generate latex macros, we need to generate a similar dictionary, but this
# time we want to map the parameter to a given latex macro name
parameter_dict = {
    "mass_1": "detector_primary",
    "mass_2": "detector_secondary"
}

# Now generate the latex macros and save it to a file
f.generate_latex_macros(
    parameter_dict=parameter_dict, save_to_file="latex_macros.tex"
)

# If you want to generate macros for more than one run, this can be done by
# again passing a list of labels that you wish to include

f.generate_latex_macros(
    parameter_dict=parameter_dict, labels=["example1", "example2"],
    save_to_file="pesummary_latex_macros.tex"
)