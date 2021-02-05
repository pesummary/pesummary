# Licensed under an MIT style license -- see LICENSE.md

from pesummary import __version__
from pesummary.io import read
from pesummary.core.notebook import (
    NoteBook, imports, pesummary_read, posterior_samples,
    samples_dict_plot
)
from .notebook import psd_plot

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def make_public_notebook(
    pesummary_file, publication_title, dcc_link=".", event="",
    default_analysis=None, default_parameter="mass_1",
    corner_parameters=["mass_1", "mass_2", "luminosity_distance", "iota"],
    filename="posterior_samples.ipynb", outdir="./", comparison_analysis=None
):
    """Make a jupyter notebook showing how to use the PESummary result file

    Parameters
    ----------
    """
    nb = NoteBook()
    f = read(pesummary_file)
    if default_analysis is None:
        default_analysis = f.labels[0]
    elif default_analysis not in f.labels:
        raise ValueError(
            "The analysis '{}' does not exist in '{}'. The available analyses "
            "are {}".format(
                default_analysis, pesummary_file, ", ".join(f.labels)
            )
        )
    cell = (
        "# Sample release{}\nThis notebook serves as a basic introduction to "
        "loading and viewing data\nreleased in associaton with the publication "
        "titled {}{}\n\nThe released data file can be read in using the "
        "PESummary or h5py libraries. For general instructions on how to "
        "manipulate the data file and/or read this data file with h5py, see the "
        "[PESummary docs](https://lscsoft.docs.ligo.org/pesummary)".format(
            event, publication_title, dcc_link
        )
    )
    nb.add_cell(cell, markdown=True)
    text, cell = imports(
        module_imports=["pesummary", "pesummary.io:read"],
        extra_lines=["print(pesummary.__version__)"]
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)
    text, cell = pesummary_read(
        pesummary_file, read_variable="data",
        text=(
            "As part of this sample release, we are releasing the posterior "
            "samples generated from {} different analyses. The samples for "
            "each analysis is stored in the data file. This data file "
            "can be read in using the 'pesummary' read function".format(
                len(f.labels)
            )
        )
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)
    text, cell = posterior_samples(
        "data", metafile=True, default_analysis=default_analysis,
        print_parameters=True, samples_variable="posterior_samples"
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)
    cell = "## {} analysis".format(default_analysis)
    nb.add_cell(cell, markdown=True)
    text, cell = samples_dict_plot(
        "posterior_samples", plot_kwargs={"type": "'hist'", "kde": True},
        plot_args=["'{}'".format(default_parameter)], text=(
            "'pesummary' allows for the user to easily make plots. As an "
            "example, we show the posterior distribution for '{}' plotted "
            "as a KDE.".format(default_parameter)
        ), extra_lines=["fig.set_size_inches(12, 8)", "fig.show()"]
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)

    samples = f.samples_dict[default_analysis]
    spin_params = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
    if all(param in samples.keys() for param in spin_params):
        text, cell = samples_dict_plot(
            "posterior_samples", plot_kwargs={
                "type": "'spin_disk'", "colorbar": True, "annotate": True,
                "show_label": True, "cmap": "'Blues'"
            }, extra_lines=["fig.show()"], text=(
                "We may also easily generate a spin disk, showing the most "
                "probable direction of the spin vectors"
            )
        )
        nb.add_cell(text, markdown=True)
        nb.add_cell(cell, code=True)

    text, cell = samples_dict_plot(
        "posterior_samples", plot_kwargs={
            "type": "'corner'", "parameters": corner_parameters
        }, text=(
            "Corner plots are very useful for spotting degeneracies between "
            "parameters. A corner plot can easily be generated using "
            "'pesummary'"
        ), extra_lines=["fig.show()"]
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)

    if len(f.labels) > 1:
        cell = "## Comparing multiple analyses"
        nb.add_cell(cell, markdown=True)
        text, cell = samples_dict_plot(
            "samples_dict", plot_args=["'{}'".format(default_parameter)],
            plot_kwargs={"type": "'hist'", "kde": True}, text=(
                "As the 'pesummary' file is able to store multiple analyses "
                "in a single file, we are able to easily generate a comparison "
                "plot showing the posterior distribution for '{}' for each "
                "analysis".format(default_parameter)
            ), extra_lines=["fig.set_size_inches(12, 8)", "fig.show()"]
        )
        nb.add_cell(text, markdown=True)
        nb.add_cell(cell, code=True)
        text, cell = samples_dict_plot(
            "samples_dict", plot_args=["'{}'".format(default_parameter)],
            plot_kwargs={"type": "'violin'"}, text=(
                "A comparison histogram is not the only way to display this "
                "data. We may also generate a violin plot showing the "
                "posterior distribution for each analysis"
            ), extra_lines=["fig.show()"]
        )
        nb.add_cell(text, markdown=True)
        nb.add_cell(cell, code=True)
        text, cell = samples_dict_plot(
            "samples_dict", plot_args=["{}".format(corner_parameters[:2])],
            plot_kwargs={"type": "'reverse_triangle'", "grid": False}, text=(
                "'pesummary' also allows for the user to generate a "
                "triangle plot with ease"
            ), extra_lines=["fig[0].show()"]
        )
        nb.add_cell(text, markdown=True)
        nb.add_cell(cell, code=True)
        text, cell = samples_dict_plot(
            "samples_dict", extra_lines=["fig.show()"], plot_kwargs={
                "type": "'corner'", "parameters": corner_parameters
            }, text=(
                "It is also useful to see how degeneracies between certain "
                "parameters change for different analysis. This can be "
                "investigated by generating a comparison corner plot"
            )
        )
        nb.add_cell(text, markdown=True)
        nb.add_cell(cell, code=True)
    cell = "## PSD data"
    nb.add_cell(cell, markdown=True)
    text, cell = psd_plot(
        "data", default_analysis, plot_kwargs={"fmin": 30},
        extra_lines=["fig.show()"], text=(
            "The 'pesummary' file also stores the PSD that was used for "
            "each analysis. This can be extracted and plotted"
        )
    )
    nb.add_cell(text, markdown=True)
    nb.add_cell(cell, code=True)
    nb.write(filename=filename, outdir=outdir)
