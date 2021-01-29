# Licensed under an MIT style license -- see LICENSE.md

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def psd_plot(
    read_variable, default_analysis=None, plot_kwargs={}, extra_lines=[],
    text="As an example, we now plot the PSDs stored in the file"
):
    """Return a string containing the function to generate a plot showing the
    stored PSDs

    Parameters
    ----------
    read_variable: str
        name of the read object
    default_analysis: str, optional
        The analysis PSD that you wish to plot
    plot_kwargs: dict, optional
        kwargs for the `.plot()` method.
    extra_lines: list, optional
        additional lines to add to the end of the string
    text: str, optional
        Markdown text explaining the plot
    """
    if default_analysis is None:
        raise ValueError("Please provide a default analysis to use")
    kwargs = ", ".join([f"{key}={item}" for key, item in plot_kwargs.items()])
    string = "psd = {}.psd['{}']\n".format(read_variable, default_analysis)
    string += "fig = psd.plot({})\n".format(kwargs)
    string += "\n".join(extra_lines)
    if text is not None:
        return [text, string]
    return [string]
