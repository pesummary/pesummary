# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
#                     Edward Fauchon-Jones <edward.fauchon-jones@ligo.org>
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

import pesummary
from pesummary.webpage import tables
from pesummary.webpage.base import Base

import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import time

BOOTSTRAP = """<!DOCTYPE html>
<html lang='en'>
    <title>title</title>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'>
    stylesheet elements
</head>
<body style='background-color:#F8F8F8; margin-top:5em'>
"""

HOME_SCRIPTS = """    <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>
    <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>
    <script src='./js/combine_corner.js'></script>
    <script src='./js/grab.js'></script>
    <script src='./js/modal.js'></script>
    <script src='./js/multi_dropbar.js'></script>
    <script src='./js/multiple_posteriors.js'></script>
    <script src='./js/search.js'></script>
    <script src='./js/side_bar.js'></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="./css/navbar.css">
"""

OTHER_SCRIPTS = """    <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>
    <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>
    <script src='../js/combine_corner.js'></script>
    <script src='../js/grab.js'></script>
    <script src='../js/modal.js'></script>
    <script src='../js/multi_dropbar.js'></script>
    <script src='../js/multiple_posteriors.js'></script>
    <script src='../js/search.js'></script>
    <script src='../js/side_bar.js'></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="../css/navbar.css">
"""


def make_html(web_dir, title="Summary Pages", pages=None, stylesheets=[],
              label=None):
    """Make the initial html page.

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    title: str, optional
        header title of html page
    pages: list, optional
        list of pages that you would like to be created
    stylesheets: list, optional
        list of stylesheets to including in the html page. It is assumed all
        stylesheets are located in the `css` in the root of the summary pages.
        This should be provided as basenames without extension that is assumed
        to be `.css`.
    """
    for i in pages:
        if i != "home":
            f = open("{}/html/{}.html".format(web_dir, i), "w")
        else:
            f = open("{}/{}.html".format(web_dir, i), "w")
        stylesheet_elements = ''.join([
            "  <link rel='stylesheet' href='../css/{0:s}.css'>\n".format(s)
            for s in stylesheets])
        bootstrap = BOOTSTRAP.split("\n")
        bootstrap[1] = "  <title>{}</title>".format(title)
        bootstrap[-4] = stylesheet_elements
        bootstrap = [j + "\n" for j in bootstrap]
        f.writelines(bootstrap)
        if i != "home":
            scripts = OTHER_SCRIPTS.split("\n")
        else:
            scripts = HOME_SCRIPTS.split("\n")
        scripts = [j + "\n" for j in scripts]
        f.writelines(scripts)


def open_html(web_dir, base_url, html_page, label=None):
    """Open html page ready so you can manipulate the contents

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    base_url: str
        url to the location where you would like the html file to be saved
    page: str
        name of the html page that you would like to edit
    label: str
        the label that prepends your page name
    """
    try:
        if html_page[-5:] == ".html":
            html_page = html_page[:-5]
    except Exception:
        pass
    if html_page == "home.html" or html_page == "home":
        f = open(web_dir + "/home.html", "a")
    else:
        if label is not None:
            f = open(web_dir + "/html/{}_".format(label) + html_page
                     + ".html", "a")
        else:
            f = open(web_dir + "/html/" + html_page + ".html", "a")
    return page(f, web_dir, base_url, label)


class page(Base):
    """Class to generate and manipulate an html page.
    """
    def __init__(self, html_file, web_dir, base_url, label):
        self.html_file = html_file
        self.web_dir = web_dir
        self.base_url = base_url
        self.label = label
        self.content = []

    def _header(self, approximant):
        """
        """
        self.add_content("<h7 hidden>{}</h7>".format(self.label))
        self.add_content("<h7 hidden>{}</h7>".format(approximant))

    def _footer(self, user, rundir):
        """
        """
        command = ""
        for i in sys.argv:
            command += " "
            if i[0] == "-":
                command += "\n"
            command += "{}".format(i)
        self.make_div(
            _class='jumbotron', _style='margin-bottom:0; line-height: 0.5;'
            + 'background-color:#E0E0E0')
        self.add_content(
            "<p>This page was produced by {} at {} on {} on behalf "
            "of the Parameter Estimation group\n".format(
                user, time.strftime("%H:%M"), time.strftime("%B %d %Y")), indent=2)
        self.add_content("<p>Run directories found at {}</p>\n".format(rundir), indent=2)
        self.add_content("<p>This code was generated with the following command line call:</p>", indent=2)
        self.add_content("<p> </p>", indent=2)
        self.make_div(
            _class='container', _style='background-color:#FFFFFF; '
            'box-shadow: 0 0 5px grey; line-height: 1.5')
        styles = self.make_code_block(language='shell', contents=command)
        with open('{0:s}/css/command_line.css'.format(self.web_dir), 'w') as g:
            g.write(styles)
        self.end_div()
        self.make_div(_style="text-align:center")
        self.add_content(
            "<a href='https://git.ligo.org/lscsoft/pesummary'>"
            "View PESummary v%s on git.ligo.org</a> | "
            "<a href='https://git.ligo.org/lscsoft/pesummary/issues'>"
            "Report an issue</a> | <a href='https://docs.ligo.org/lscsoft/"
            "pesummary/summarypage.html'> Help on using this webpage</a>" % (
                pesummary.__version__), indent=2)
        self.end_div()
        self.end_div()

    def _setup_navbar(self, background_colour):
        if background_colour == "navbar-dark" or background_colour is None:
            self.add_content("<nav class='navbar navbar-expand-sm navbar-dark "
                             "bg-dark fixed-top'>\n")
        else:
            self.add_content("<nav class='navbar navbar-expand-sm fixed-top "
                             "navbar-custom' style='background-color: %s'>" % (
                                 background_colour))
        self.add_content("<a class='navbar-brand' href='#' style='color: white'"
                         ">Navbar</a>\n", indent=2)
        self.add_content("<button class='navbar-toggler' type='button' "
                         "data-toggle='collapse' data-target='#collapsibleNavbar'>\n", indent=2)
        self.add_content("<span class='navbar-toggler-icon'></span>\n", indent=4)
        self.add_content("</button>\n", indent=2)

    def make_header(self, approximant="IMRPhenomPv2"):
        """Make header for document in bootstrap format.

        Parameters
        ----------
        title: str, optional
            header title of html page
        approximant: str, optional
            the approximant that you are analysing
        """
        self._header(approximant)

    def make_footer(self, user=None, rundir=None):
        """Make footer for document in bootstrap format.
        """
        self._footer(user, rundir)

    def make_navbar(self, links=None, samples_path="./samples", search=True,
                    histogram_download=None,
                    background_color="navbar-dark",
                    hdf5=False):
        """Make a navigation bar in boostrap format.

        Parameters
        ----------
        links: list, optional
            list giving links that you want your navbar to include. If a
            dropdown option is required, give a 2d list showing the main link
            followed by dropdown links. For instance, if you wanted to have
            links corner plots and a dropdown link named 1d_histograms with
            options mass1, mass2, mchirp, then we would give,

                links=[corner, [1d_histograms, [mass1, mass2, mchirp]]]

        samples_path: str, optional
            path to the location of the meta file
        search: bool, optional
            if True, search bar will be given in navbar
        histogram_download: str, optional
            path to the location of the data associated with the histogram
        hdf5: Bool, optional
            true if a hdf5 file format is chosen for the meta file
        """
        self._setup_navbar(background_color)
        if links is None:
            raise Exception("Please specify links for use with navbar\n")
        self.add_content("<div class='collapse navbar-collapse' id='collapsibleNavbar'>\n", indent=4)
        self.add_content("<ul class='navbar-nav'>\n", indent=6)
        for i in links:
            if type(i) == list:
                self.add_content("<li class='nav-item'>\n", indent=8)
                self.add_content("<li class='nav-item dropdown'>\n", indent=10)
                self.add_content("<a class='nav-link dropdown-toggle' "
                                 "href='#' id='navbarDropdown' role='button' "
                                 "data-toggle='dropdown' aria-haspopup='true' "
                                 "aria-expanded='false'>\n", indent=12)
                self.add_content("{}\n".format(i[0]), indent=12)
                self.add_content("</a>\n", indent=12)
                self.add_content("<ul class='dropdown-menu' aria-labelledby='dropdown1'>\n", indent=12)
                for j in i:
                    if type(j) == list:
                        if len(j) > 1:
                            if type(j[1]) == list:
                                self.add_content("<li class='dropdown-item dropdown'>\n", indent=14)
                                self.add_content("<a class='dropdown-toggle' id='{}' "
                                                 "data-toggle='dropdown' "
                                                 "aria-haspopup='true' "
                                                 "aria-expanded='false'>{}</a>\n".format(j[0], j[0]), indent=16)
                                self.add_content("<ul class='dropdown-menu' "
                                                 "aria-labelledby='{}'>\n".format(j[0]), indent=16)
                                for k in j[1]:
                                    if type(k) == dict:
                                        key = list(k.keys())[0]
                                        self.add_content(
                                            "<li class='dropdown-item' href='#' "
                                            "onclick='grab_html(\"{}\", label=\"{}\")'>"
                                            "<a>{}</a></li>\n".format(key, k[key], key), indent=18)
                                    else:
                                        self.add_content(
                                            "<li class='dropdown-item' href='#' "
                                            "onclick='grab_html(\"{}\")'>"
                                            "<a>{}</a></li>\n".format(k, k), indent=18)

                                self.add_content("</ul>", indent=16)
                                self.add_content("</li>", indent=14)
                            else:
                                for k in j:
                                    if type(k) == dict:
                                        key = list(k.keys())[0]
                                        self.add_content(
                                            "<li class='dropdown-item' href='#' "
                                            "onclick='grab_html(\"{}\", label=\"{}\")'>"
                                            "<a>{}</a></li>\n".format(key, k[key], key), indent=14)

                                    else:
                                        self.add_content(
                                            "<li class='dropdown-item' href='#' "
                                            "onclick='grab_html(\"{}\")'>"
                                            "<a>{}</a></li>\n".format(k, k), indent=14)

                        else:
                            if type(j[0]) == dict:
                                key = list(j[0].keys())[0]
                                self.add_content(
                                    "<li class='dropdown-item' href='#' "
                                    "onclick='grab_html(\"{}\", label=\"{}\")'>"
                                    "<a>{}</a></li>\n".format(key, j[0][key], key), indent=14)

                            else:
                                self.add_content(
                                    "<li class='dropdown-item' href='#' "
                                    "onclick='grab_html(\"{}\")'>"
                                    "<a>{}</a></li>\n".format(j[0], j[0]), indent=14)

                self.add_content("</ul>\n", indent=12)
                self.add_content("</li>\n", indent=10)
            else:
                self.add_content("<li class='nav-item'>\n", indent=8)
                if i == "home":
                    self.add_content("<a class='nav-link' "
                                     "href='#' onclick='grab_html(\"{}\")'"
                                     ">{}</a>\n".format(i, i), indent=10)
                else:
                    if type(i) == dict:
                        key = list(i.keys())[0]
                        self.add_content(
                            "<a class='nav-link' "
                            "href='#' onclick='grab_html(\"{}\", label=\"{}\")'"
                            ">{}</a>\n".format(key, i[key], key), indent=10)

                    else:
                        self.add_content(
                            "<a class='nav-link' "
                            "href='#' onclick='grab_html(\"{}\")'"
                            ">{}</a>\n".format(i, i), indent=10)

                self.add_content("</li>\n", indent=8)
        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)
        if histogram_download:
            self.add_content("<a href='%s' download>" % (histogram_download),
                             indent=4)
            self.add_content(
                "<button type='submit' style='margin-right: 15px; cursor:pointer'> "
                "<i class='fa fa-download'></i> Histogram Data</button>", indent=6)
            self.add_content("</a>", indent=4)
        if hdf5:
            path = '%s/posterior_samples.h5' % (samples_path)
        else:
            path = '%s/posterior_samples.json' % (samples_path)
        self.add_content("<a href='%s' download>" % (path), indent=4)
        self.add_content(
            "<button type='submit' style='margin-right: 15px; cursor:pointer'> "
            "<i class='fa fa-download'></i> Results File</button>", indent=6)
        self.add_content("</a>", indent=4)
        if search:
            self.add_content("<input type='text' placeholder='search' id='search'>\n", indent=4)
            self.add_content("<button type='submit' onclick='myFunction()'>Search</button>\n", indent=4)
        self.add_content("</nav>\n")

    def make_table(self, headings=None, contents=None, heading_span=1,
                   colors=None, accordian_header="Summary Table", **kwargs):
        """Generate a table in bootstrap format.

        Parameters
        ----------
        headings: list, optional
            list of headings
        contents: list, optional
            nd list giving the contents of the table.
        heading_span: int, optional
            width of the header cell. By default it will span a single column
        colors: list, optional
            list of colors for the table columns
        """
        label = accordian_header.replace(" ", "_")
        self.make_container(style=kwargs.get("style", None))
        self.make_div(indent=2, _class='row justify-content-center')
        self.make_div(
            indent=4, _class='accordian', _style='width: 100%',
            _id='accordian%s' % (label))
        self.make_div(indent=6, _class='card')
        self.make_div(
            indent=8, _class='card-header', _style='background-color: #E0E0E0',
            _id='table')
        self.add_content("<h5 class='mb-0'>", indent=10)
        self.make_div(indent=12, _class='row justify-content-center')
        self.add_content(
            "<button class='btn btn-link collapsed' type='button' "
            "data-toggle='collapse' data-target='#collapsetable%s' "
            "aria-expanded='false' aria-controls='collapsetable'>" % (label), indent=14)
        self.add_content(accordian_header)
        self.add_content("</button>")
        self.end_div(indent=12)
        self.end_div(indent=10)
        self.add_content(
            "<div id='collapsetable%s' class='collapse' "
            "aria-labelledby='table' data-parent='#accordian%s'>" % (label, label),
            indent=12)
        self.make_div(_class='card-body', indent=14)
        self.make_div(_class='row justify-content-center', indent=16)
        self.make_div(_class='container', indent=18)
        if type(contents) == list:
            self.make_div(indent=20, _class='table-responsive')
            if heading_span > 1:
                self.add_content("<table class='table table-sm'>\n", indent=22)
            else:
                self.add_content("<table class='table table-striped table-sm'>\n", indent=24)
            self.add_content("<thead>\n", indent=26)
            self.add_content("<tr>\n", indent=28)
            for i in headings:
                self.add_content("<th colspan='{}'>{}</th>\n".format(heading_span, i), indent=30)
            self.add_content("</tr>\n", indent=28)
            self.add_content("<tbody>\n", indent=26)

            for num, i in enumerate(contents):
                self.add_content("<tr>\n", indent=28)
                if heading_span == 2:
                    for j, col in zip(i, ["#ffffff"] + colors * (len(i) - 1)):
                        self.add_content(
                            "<td style='background-color: {}'>{}</td>\n".format(col, j), indent=30)
                    self.add_content("</tr>", indent=28)
                else:
                    for j in i:
                        self.add_content("<td>{}</td>\n".format(j), indent=30)
                    self.add_content("</tr>\n", indent=28)
            self.add_content("</tbody>\n", indent=26)
            self.add_content("</table>\n", indent=24)
            self.end_div(indent=20)
        elif type(contents) == dict:
            self.make_div(indent=20, _class='table-responsive')
            if heading_span > 1:
                self.add_content("<table class='table table-sm'>\n", indent=22)
            else:
                self.add_content("<table class='table table-striped table-sm'>\n", indent=24)
            self.add_content("<thead>\n", indent=26)
            for j in contents.keys():
                self.add_content("<tr bgcolor='#F0F0F0'>\n", indent=28)
                self.add_content(
                    "<th colspan='{}' class='text-center'>"
                    "{}</th>\n".format(len(contents[j][0]), j), indent=30)
                self.add_content("</tr>\n", indent=28)
                self.add_content("<tr>\n", indent=28)
                for i in headings:
                    self.add_content(
                        "<th colspan='{}' class='text-center'>"
                        "{}</th>\n".format(heading_span, i), indent=30)
                self.add_content("</tr>\n", indent=28)

                for num, i in enumerate(contents[j]):
                    self.add_content("<tr>\n", indent=28)
                    for j in i:
                        self.add_content("<td>{}</td>\n".format(j), indent=30)
                    self.add_content("</tr>\n", indent=28)
                self.add_content("</tbody>\n", indent=26)
            self.add_content("</table>\n", indent=24)
            self.end_div(indent=20)

        self.end_div(indent=18)
        self.end_div(indent=16)
        self.end_div(indent=14)
        self.end_div(indent=12)
        self.end_div(indent=10)
        self.end_div(indent=8)
        self.end_div(indent=6)
        self.end_div(indent=4)
        self.end_div(indent=2)
        self.end_div()

    def make_code_block(self, language=None, contents=None):
        """Generate a code block hightlighted using pigments.

        Parameters
        ----------
        language: str, optional
            The lanaguge of the configuration file to use for syntax
            highlighting.
        contents: str, optional
            String containing the contents of the config file.

        Returns
        -------
        style: str
            The styles used to highlight the rendered contents.
        """
        lexer = get_lexer_by_name(language)
        formatter = HtmlFormatter(style='manni')
        render = highlight(contents, lexer, formatter)
        self.add_content(render)
        styles = formatter.get_style_defs('.highlight')
        styles += ".highlight {margin: 20px; padding: 20px;}"
        return styles

    def make_table_of_images(self, contents=None, rows=None, columns=None,
                             code="modal"):
        """Generate a table of images in bootstrap format.

        Parameters
        ----------
        headings: list, optional
            list of headings
        contents: list, optional
            nd list giving the contents of the table.
        carousel: bool, optional
            if True, the images will be configured to work operate as part of
            a carousel
        width: float, optional
            width of the images in the table
        container: bool, optional
            if True, the table of images is placed inside a container
        """
        table = tables.table_of_images(contents, rows, columns, self.html_file,
                                       code=code)
        table.make()

    def insert_image(self, path, justify="center", code=None):
        """Generate an image in bootstrap format.

        Parameters
        ----------
        path: str, optional
            path to the image that you would like inserted
        justify: str, optional
            justifies the image to either the left, right or center
        """
        self.make_container()
        _id = path.split("/")[-1][:-4]
        string = "<img src='{}' id='{}' alt='No image available' ".format(path, _id) + \
                 "style='align-items:center; width:850px; cursor: pointer'"
        if justify == "center":
            string += " class='mx-auto d-block'"
        elif justify == "left":
            string = string[:-1] + " float:left;'"
        elif justify == "right":
            string = string[:-1] + " float:right;'"
        if code:
            string += " onclick='{}(\"{}\")'".format(code, _id)
        string += ">\n"
        self.add_content(string, indent=2)
        self.end_container()

    def make_accordian(self, headings=None, content=None):
        """Generate an accordian in bootstrap format with images as content.

        Parameters
        ----------
        headings: list, optional
            list of headings that you want your accordian to have
        content: nd list, optional
            n dimensional list where n is the number of rows. The content
            of each list should be the path to the location of the image
        """
        self.make_div(_class='row justify-content-center')
        self.add_content("<div class='accordian' id='accordian' style='width:70%'>\n", indent=2)
        for num, i in enumerate(headings):
            self.add_content("<div class='card' style='border: 0px solid black'>\n", indent=4)
            self.add_content("<div class='card-header' id='{}'>\n".format(i), indent=6)
            self.add_content("<h5 class='mb-0'>\n", indent=8)
            self.add_content(
                "<button class='btn btn-link collapsed' type='button' data-toggle='collapse' "
                "data-target='#collapse{}' aria-expanded='false' ".format(i)
                + "aria-controls='collapse{}'>\n".format(i), indent=10)
            self.add_content("{}\n".format(i), indent=12)
            self.add_content("</button>\n", indent=10)
            self.add_content("</h5>\n", indent=8)
            self.add_content("</div>\n", indent=6)
            self.add_content(
                "<div id='collapse{}' class='collapse' ".format(i)
                + "aria-labelledby='{}' data-parent='#accordian'>\n".format(i), indent=6)
            self.add_content("<div class='card-body'>\n", indent=8)
            self.add_content("<img src='{}' ".format(content[num])
                             + "alt='No image available' style='width:700px;' "
                             + "class='mx-auto d-block'>\n", indent=10)
            self.add_content("</div>\n", indent=8)
            self.add_content("</div>\n", indent=6)
            self.add_content("</div>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def make_search_bar(self, sidebar=None, popular_options=None, label=None,
                        code="combine"):
        """Generate a search bar to combine the corner plots
        javascript.

        Parameters
        ----------
        sidebar: list, optional
            a list of parameters that you would like included in the side bar
        popular_options: list, optional
            a list of popular options for your search bar
        """
        ids = "canvas" if code == "combine" else code
        self.add_content("<link rel='stylesheet' href='../css/side_bar.css'>\n")
        self.add_content("<div class='w3-sidebar w3-bar-block w3-border-right sidenav' "
                         "style='display:none' id='mySidebar'>\n")
        self.add_content("<button onclick='side_bar_close()' class='close'>&times;</button>\n", indent=2)
        if sidebar:
            for i in sidebar:
                self.add_content("<input type='checkbox' name='type' "
                                 "value='{}' id='{}' style='text-align: center; margin: 0 5px 0;'"
                                 ">{}<br>\n".format(i, i, i, i), indent=2)
        self.add_content("</div>")
        self.add_content("<div class='row justify-content-center'>")
        self.add_content(
            "<p style='margin-top:2.5em'> Input the parameter names that you "
            "would like to compare</p>", indent=2)
        self.add_content("</div>")
        self.add_content("<div class='row justify-content-center'>\n")
        self.add_content("<input type='text' placeholder='search' id='corner_search'>\n", indent=2)
        self.add_content(
            "<button type='submit' onclick='{}(undefined, label=\"{}\")' "
            "style='cursor: pointer'>Submit</button>\n".format(code, label), indent=2)
        self.add_content(
            "<button class='w3-button w3-teal w3-xlarge' "
            "onclick='side_bar_open()' style='cursor: pointer'>&#8801</button>\n", indent=2)
        self.add_content("</div>\n")
        self.add_content("<div class='row justify-content-center'>\n")
        if popular_options:
            for i in popular_options:
                if type(i) == dict and list(i.keys()) == ["all"]:
                    self.add_content("<button type='button' class='btn btn-info' "
                                     "onclick='{}(\"{}\", label=\"{}\")' "
                                     "style='margin-left:0.25em; margin-right:0.25em; "
                                     "margin-top: 1.0em; cursor: pointer'>all</button>\n".format(
                                         code, i["all"], label), indent=2)
                else:
                    self.add_content("<button type='button' class='btn btn-info' "
                                     "onclick='{}(\"{}\", label=\"{}\")' "
                                     "style='margin-left:0.25em; margin-right:0.25em; "
                                     "margin-top: 1.0em; cursor: pointer'>{}</button>\n".format(
                                         code, i, label, i), indent=2)
        self.add_content("</div>")
        self.add_content("<div class='container' style='margin-top:5em; margin-bottom:5em;"
                         "background-color:#FFFFFF; box-shadow: 0 0 5px grey;'>\n")
        self.add_content("<div class='row justify-content-center' id='corner_plot'>\n", indent=2)
        self.add_content("<canvas id='{}' width='600' height='600'></canvas>\n".format(ids), indent=4)
        if code == "combine":
            self.add_content("<img src='' id='mirror'/>", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def make_modal_carousel(self, images=None):
        """Make a pop up window that appears on top of the home page showing
        images in a carousel.

        Parameters
        ----------
        images: list
            list of image locations that you would like included in the
            carousel
        """
        self.add_content("<div class='modal fade bs-example-modal-lg' tabindex='-1' "
                         "role='dialog' aria-labelledby='myLargeModalLabel' "
                         "aria-hidden='true' id='myModel' style='margin-top: 200px;'>\n")
        self.add_content("<div class='modal-dialog modal-lg' style='width:90%'>\n", indent=2)
        self.add_content("<div class='modal-content'>\n", indent=4)
        self.add_content("<div id='demo' class='carousel slide' data-ride='carousel'"
                         " data-interval='false'>\n", indent=6)
        self.add_content("<ul class='carousel-indicators'>\n", indent=8)
        for num, i in enumerate(images):
            if num == 0:
                self.add_content("<li data-target='#demo' data-slide-to-'%s' "
                                 "class='active'></li>\n" % (num), indent=10)
            self.add_content("<li data-target='#demo' data-slide-to-'%s'>"
                             "</li>\n" % (num), indent=10)
        self.add_content("<li data-target='#demo' data-slide-to='0' "
                         "class='active'></li>\n", indent=10)
        self.add_content("<li data-target='#demo' data-slide-to='1'></li>\n", indent=10)
        self.add_content("<li data-target='#demo' data-slide-to='2'></li>\n", indent=10)
        self.add_content("</ul>\n", indent=8)
        self.add_content("<div class='carousel-inner'>\n", indent=8)
        for num, i in enumerate(images):
            if num == 0:
                self.add_content("<div class='carousel-item active'>\n", indent=10)
            else:
                self.add_content("<div class='carousel-item'>\n", indent=10)
            self.add_content("<img src={} style='align-items:center;' "
                             "class='mx-auto d-block'>\n".format(i), indent=12)
            self.add_content("</div>\n", indent=10)

        self.add_content("</div>\n", indent=8)
        self.add_content("<a class='carousel-control-prev' href='#demo' "
                         "data-slide='prev'>\n", indent=8)
        self.add_content("<span class='carousel-control-prev-icon'></span>\n", indent=10)
        self.add_content("</a>\n", indent=8)
        self.add_content("<a class='carousel-control-next' href='#demo' "
                         "data-slide='next'>\n", indent=8)
        self.add_content("<span class='carousel-control-next-icon'></span>\n", indent=10)
        self.add_content("</a>\n", indent=8)
        self.add_content("</div>\n", indent=6)
        self.add_content("</div>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")
