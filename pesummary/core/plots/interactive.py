# Copyright (C) 2019  Charlie Hoy <charlie.hoy@ligo.org>
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

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly


def write_to_html(fig, filename):
    """Write a plotly.graph.objects.go.Figure to a html file

    Parameters
    ----------
    fig: plotly.graph.objects.go.Figure object
        figure containing the plot that you wish to save to html
    filename: str
        name of the file that you wish to write the figure to
    """
    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    data = "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
    data += (
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/"
        "MathJax.js?config=TeX-MML-AM_SVG'></script>"
    )
    with open(filename, "w") as f:
        data += div
        f.write(data)


def corner(
    data, labels, dimensions={'width': 900, 'height': 900}, show_diagonal=False,
    colors={'selected': 'rgba(248,148,6,1)', 'not_selected': 'rgba(0,0,0,1)'},
    show_upper_half=False, write_to_html_file="interactive_corner.html"
):
    """Build an interactive corner plot

    Parameters
    ----------
    data: list, np.ndarray
        The samples you wish to produce a corner plot for. This should be a 2
        dimensional array where the zeroth axis is the list of samples and
        the next axis is are the dimensions of the space
    labels: list, np.ndarray
        A list of names for each dimension
    dimensions: dict
        A dictionary giving the width and height of the figure.
    show_diagonal: Bool
        Whether or not to show the diagonal scatter plots
    colors: dict
        A dictionary of colors for the individual samples. The dictionary should
        have keys 'selected' and 'not_selected' to indicate the colors to be
        used when the markers are selected and not selected respectively
    show_upper_half: Bool
        Whether or not to show the upper half of scatter plots
    write_to_html_file: str
        Name of the html file you wish to write the figure to
    """
    data_structure = [
        dict(label=label, values=value) for label, value in zip(
            labels, data
        )
    ]
    fig = go.Figure(
        data=go.Splom(
            dimensions=data_structure,
            marker=dict(
                color=colors["not_selected"], showscale=False,
                line_color='white', line_width=0.5,
                size=3
            ),
            selected=dict(marker=dict(color=colors["selected"])),
            diagonal_visible=show_diagonal,
            showupperhalf=show_upper_half,
        )
    )
    fig.update_layout(
        dragmode='select',
        width=dimensions["width"],
        height=dimensions["height"],
        hovermode='closest',
        font=dict(
            size=10
        )
    )
    if write_to_html_file is not None:
        write_to_html(fig, write_to_html_file)
        return
    return fig
