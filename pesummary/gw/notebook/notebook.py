# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


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
