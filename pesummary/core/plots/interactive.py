# Licensed under an MIT style license -- see LICENSE.md

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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


def histogram2d(
    x, y, xlabel='x', ylabel='y', contour=False, contour_color='Blues',
    marker_color='rgba(248,148,6,1)', dimensions={'width': 900, 'height': 900},
    write_to_html_file="interactive_2d_histogram.html", showgrid=False,
    showlegend=False
):
    """Build an interactive 2d histogram plot

    Parameters
    ----------
    x: np.ndarray
        An array containing the x coordinates of the points to be histogrammed
    y: np.ndarray
        An array containing the y coordinates of the points to be histogrammed
    xlabel: str
        The label for the x coordinates
    ylabel: str
        The label for the y coordinates
    contour: Bool
        Whether or not to show contours on the scatter plot
    contour_color: str
        Name of the matplotlib palette to use for contour colors
    marker_color: str
        Color to use for the markers
    dimensions: dict
        A dictionary giving the width and height of the figure.
    write_to_html_file: str
        Name of the html file you wish to write the figure to
    showgrid: Bool
        Whether or not to show a grid on the plot
    showlegend: Bool
        Whether or not to add a legend to the plot
    """
    fig = go.Figure()
    if contour:
        fig.add_trace(
            go.Histogram2dContour(
                x=x, y=y, colorscale=contour_color, reversescale=True,
                xaxis='x', yaxis='y', histnorm="probability density"
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, xaxis='x', yaxis='y', mode='markers',
            marker=dict(color=marker_color, size=3)
        )
    )
    fig.add_trace(
        go.Histogram(
            y=y, xaxis='x2', marker=dict(color=marker_color),
            histnorm="probability density"
        )
    )
    fig.add_trace(
        go.Histogram(
            x=x, yaxis='y2', marker=dict(color=marker_color),
            histnorm="probability density"
        )
    )

    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False, domain=[0, 0.85], showgrid=showgrid
        ),
        yaxis=dict(
            zeroline=False, domain=[0, 0.85], showgrid=showgrid
        ),
        xaxis2=dict(
            zeroline=False, domain=[0.85, 1], showgrid=showgrid
        ),
        yaxis2=dict(
            zeroline=False, domain=[0.85, 1], showgrid=showgrid
        ),
        height=dimensions["height"],
        width=dimensions["width"],
        bargap=0,
        hovermode='closest',
        showlegend=showlegend,
        font=dict(
            size=10
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    if write_to_html_file is not None:
        write_to_html(fig, write_to_html_file)
        return
    return fig


def ridgeline(
    data, labels, xlabel='x', palette='colorblind', colors=None, width=3,
    write_to_html_file="interactive_ridgeline.html", showlegend=False,
    dimensions={'width': 1100, 'height': 700}
):
    """Build an interactive ridgeline plot

    Parameters
    ----------
    data: list, np.ndarray
        The samples you wish to produce a ridgline plot for. This should be a 2
        dimensional array where the zeroth axis is the list of samples and
        the next axis is are the dimensions of the space
    labels: list
        List of labels corresponding to each set of samples
    xlabel: str
        The label for the x coordinates
    palette: str
        Name of the seaborn colorpalette to use for the different posterior
        distributions
    colors: list
        List of colors to use for the different posterior distributions
    width: float
        Width of the violin plots
    write_to_html_file: str
        Name of the html file you wish to write the figure to
    showlegend: Bool
        Whether or not to add a legend to the plot
    dimensions: dict
        A dictionary giving the width and height of the figure
    """
    fig = go.Figure()
    if colors is None:
        import seaborn

        colors = seaborn.color_palette(
            palette=palette, n_colors=len(data)
        ).as_hex()

    for dd, label, color in zip(data, labels, colors):
        fig.add_trace(go.Violin(x=dd, line_color=color, name=label))

    fig.update_traces(
        orientation='h', side='positive', width=width, points=False
    )
    fig.update_layout(
        xaxis_showgrid=False, xaxis_zeroline=False, xaxis_title=xlabel,
        width=dimensions["width"], height=dimensions["height"],
        font=dict(size=18), showlegend=showlegend
    )
    if write_to_html_file is not None:
        write_to_html(fig, write_to_html_file)
        return
    return fig


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
