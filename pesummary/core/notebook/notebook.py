# Licensed under an MIT style license -- see LICENSE.md

try:
    import nbformat as nbf
except ImportError:
    raise ImportError("'nbformat' is required for this module")
import os

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class NoteBook(object):
    """Class to handle the creation of a jupyter notebook

    Attributes
    ----------
    cells: dict
        dictionary containing the cell contents keyed by their cell number
    """
    def __init__(self):
        self.cells = {}

    def add_cell(self, string, markdown=False, code=False):
        """Add cell to the cell dictionary

        Parameters
        ----------
        string: str
            string containing the cell contents
        markdown: Bool, optional
            if True, treat the cell as a markdown cell. Default False
        code: Bool, optional
            if True, treat the cell as a code cell. Default False
        """
        cell_number = len(self.cells)
        if markdown and not code:
            self.cells[cell_number] = nbf.v4.new_markdown_cell(string)
        elif code and not markdown:
            self.cells[cell_number] = nbf.v4.new_code_cell(string)
        else:
            raise ValueError("String must either be markdown or code not both")

    def write(self, filename="posterior_samples.ipynb", outdir="./"):
        """Save the cells to file

        Parameters
        ----------
        cells: dict
            dictionary containing the cell information. Keyed by their cell
            number
        filename: str, optional
            name of the file you wish to save the notebook to. Default
            posterior_samples.ipynb
        outdir: str, optional
            the directory to save the file. Default ./
        """
        nb = nbf.v4.new_notebook()
        nb['cells'] = [self.cells[num] for num in range(len(self.cells))]
        nbf.write(nb, os.path.join(outdir, filename))


def imports(
    magic="%matplotlib inline", comment=None,
    module_imports=["pesummary", "pesummary.io:read"],
    text="First we import the key python modules", extra_lines=[]
):
    """Return a string containing the key imports

    Parameters
    ----------
    comment: str, optional
        comment to add at the top of the string. Default None
    magic: str, optional
        magic statement to include in the string. Default '%matplotlib inline'
    module_imports: list, optional
        list of modules you wish to import. If a colon is in the import string
        this is interpreted as, 'from %s import %s' % tuple(import.split(':'))'
    text: str, optional
        Markdown text explaining the import cell below. Default
        'First we import the key python modules'
    extra_lines: list, optional
        optional lines to add to the end of the string
    """
    string = ""
    if comment is not None:
        string += comment + "\n"
    if magic is not None:
        string += magic + "\n"
    for _import in module_imports:
        if ":" not in _import:
            string += "import {}\n".format(_import)
        else:
            string += "from %s import %s\n" % tuple(_import.split(":"))
    string += "\n".join(extra_lines)
    if text is not None:
        return [text, string]
    return [string]


def pesummary_read(
    path, text="We now load in the file using the 'pesummary' read function",
    read_variable="data"
):
    """Return a string containing the function to read a file

    Parameters
    ----------
    path: str
        path to the result file you wish to load
    text: str, optional
        Markdown text explaining the import cell below. Default
        'We now load in the file using the 'pesummary' read function'
    read_variable: str, optional
        name of the variable to assign to the read result file. Default 'data'
    """
    string = "file_name = '{}'\n{} = read(file_name)".format(path, read_variable)
    if text is not None:
        return [text, string]
    return [string]


def posterior_samples(
    read_variable, metafile=False, default_analysis=None,
    print_parameters=True, samples_variable="posterior_samples", text=(
        "The posterior samples can be extracted through the `samples_dict` "
        "property. These posterior samples are stored in a custom "
        "table structure"
    )
):
    """Return a string showing how to extract the posterior samples from a
    result file

    Parameters
    ----------
    read_variable: str
        name of the read object
    metafile: Bool, optional
        if True, the result file is a pesummary meta file
    default_analysis: str, optional
        default analysis to use when `metafile=True`
    print_parameters: Bool, optional
        if True, print the parameters stored in the table
    samples_variable: str, optional
        name of the SamplesDict class
    text: str, optional
        Markdown text explaining how to extract posterior samples from the file
    """
    string = "samples_dict = {}.samples_dict\n".format(read_variable)
    if metafile:
        if default_analysis is None:
            raise ValueError("Please provide a default analysis")
        string += "{} = samples_dict['{}']\n".format(
            samples_variable, default_analysis
        )
    if print_parameters:
        string += "parameters = list({}.keys())\n".format(samples_variable)
        string += "print(parameters)"
    if text is not None:
        return [text, string]
    return [string]


def samples_dict_plot(
    samples_variable, plot_args=[], plot_kwargs={}, extra_lines=[],
    text="As an example, we now plot the posterior samples"
):
    """Return a string containing the function to generate a plot

    Parameters
    ----------
    samples_variable: str
        name of the SamplesDict class
    plot_args: list, optional
        arguments for the `.plot()` method
    plot_kwargs: dict, optional
        kwargs for the `.plot()` method.
    extra_lines: list, optional
        additional lines to add to the end of the string
    text: str, optional
        Markdown text explaining the plot
    """
    args = ", ".join(plot_args)
    kwargs = ", ".join([f"{key}={item}" for key, item in plot_kwargs.items()])
    string = "fig = {}.plot({})\n".format(
        samples_variable, "%s, %s" % (args, kwargs) if len(args) else kwargs
    )
    string += "\n".join(extra_lines)
    if text is not None:
        return [text, string]
    return [string]
