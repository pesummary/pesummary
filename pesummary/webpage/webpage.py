# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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
from pesummary.utils import utils
import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

def make_html(web_dir, title="Summary Pages", pages=None, stylesheets=[]):
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
            i = "html/" + i
        f = open("{}/{}.html".format(web_dir, i), "w")
        doc_type = "<!DOCTYPE html>\n"
        stylesheet_elements = ''.join([
            "  <link rel='stylesheet' href='../css/{0:s}.css'>\n".format(s)
            for s in stylesheets])
        bootstrap = "<html lang='en'>\n" + \
                    "  <title>{}</title>\n".format(title) + \
                    "  <meta charset='utf-8'>\n" + \
                    "  <meta name='viewport' content='width=device-width, initial-scale=1'>\n" + \
                    "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'>\n" + \
                    "  <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>\n" + \
                    "  <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>\n" + \
                    "  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>\n" + \
                    stylesheet_elements + \
                    "</head>\n" + "<body style='background-color:#F8F8F8'>\n"
        f.writelines([doc_type, bootstrap])

def open_html(web_dir, base_url, html_page):
    """Open html page ready so you can manipulate the contents

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    base_url: str
        url to the location where you would like the html file to be saved
    page: str
        name of the html page that you would like to edit
    """
    try:
        if html_page[-5:] == ".html":
            html_page = html_page[:-5]
    except:
        pass
    if html_page == "home.html" or html_page == "home":
        f = open(web_dir+"/home.html", "a")
    else:
        f = open(web_dir+"/html/"+html_page+".html", "a")
    return page(f, web_dir, base_url)

class page():
    """Class to generate and manipulate an html page.
    """
    def __init__(self, html_file, web_dir, base_url):
        self.html_file = html_file
        self.web_dir = web_dir
        self.base_url = base_url
        self.content = []

    def close(self):
        """Close the opened html file.
        """
        self.html_file.close()

    def add_content(self, content, indent=0):
        """Add content to the html page

        Parameters
        ----------
        content: str, optional
            string that you want to add to html page
        indent: int, optional
            the indent of the line
        """
        self.html_file.write(" "*indent + content)

    def _header(self, title, colour, approximant):
        """
        """
        self.add_content("<div class='jumbotron text-center' style='background-color: {}; margin-bottom:0'>\n".format(colour))
        self.add_content("  <h1 id={}>{}</h1>\n".format(approximant, title))
        self.add_content("<h4><span class='badge badge-info'>Code Version: %s"
                         "</span></h4>\n" %(pesummary.__version__), indent=2)
        self.add_content("</div>\n")

    def _footer(self, user, rundir):
        """
        """
        command= ""
        for i in sys.argv:
            command+=" {}".format(i)
        self.add_content("<div class='jumbotron text-center' style='margin-bottom:0'>\n")
        self.add_content("<p>Simulation run by {}. Run directories found at {}</p>\n".format(user, rundir), indent=2)
        self.add_content("<p>Command line: {}</p>\n".format(command), indent=2)
        self.add_content("</div>\n")

    def _setup_navbar(self):
        self.add_content("<script src='{}/js/variables.js'></script>\n".format(self.base_url))
        self.add_content("<script src='{}/js/grab.js'></script>\n".format(self.base_url))
        self.add_content("<script src='{}/js/multi_dropbar.js'></script>\n".format(self.base_url))
        self.add_content("<nav class='navbar navbar-expand-sm navbar-dark bg-dark'>\n")
        self.add_content("<a class='navbar-brand' href='#'>Navbar</a>\n", indent=2)
        self.add_content("<button class='navbar-toggler' type='button' "
                         "data-toggle='collapse' data-target='#collapsibleNavbar'>\n", indent=2)
        self.add_content("<span class='navbar-toggler-icon'></span>\n", indent=4)
        self.add_content("</button>\n", indent=2)

    def make_header(self, title="Parameter Estimation Summary Pages", background_colour="#eee",
                    approximant="IMRPhenomPv2"):
        """Make header for document in bootstrap format.

        Parameters
        ----------
        title: str, optional
            header title of html page
        approximant: str, optional
            the approximant that you are analysing
        """
        self._header(title, background_colour, approximant)

    def make_footer(self, user=None, rundir=None):
        """Make footer for document in bootstrap format.
        """
        self._footer(user, rundir)

    def make_navbar(self, links=None, search=True):
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
        search: bool, optional
            if True, search bar will be given in navbar
        """
        self._setup_navbar()
        if links == None:
            raise Exception ("Please specify links for use with navbar\n")
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
                                    self.add_content("<li class='dropdown-item' href='#' "
                                                     "onclick='grab_html(\"{}\")'>"
                                                     "<a>{}</a></li>\n".format(k, k), indent=18)
                                self.add_content("</ul>", indent=16)
                                self.add_content("</li>", indent=14)
                            else:
                                for k in j:
                                    self.add_content("<li class='dropdown-item' href='#' "
                                                     "onclick='grab_html(\"{}\")'>"
                                                     "<a>{}</a></li>\n".format(k, k), indent=14)
                        else:
                            self.add_content("<li class='dropdown-item' href='#' "
                                             "onclick='grab_html(\"{}\")'>"
                                             "<a>{}</a></li>\n".format(j[0], j[0]), indent=14)
                self.add_content("</ul>\n", indent=12)
                self.add_content("</li>\n", indent=10)  
            else:
                self.add_content("<li class='nav-item'>\n", indent=8)
                if i == "home":
                    self.add_content("<a class='nav-link' "
                                     "href='{}/{}.html'>{}</a>\n".format(self.base_url, i, i), indent=10)
                else:
                    self.add_content("<a class='nav-link' "
                                     "href='#' onclick='grab_html(\"{}\")'"
                                     ">{}</a>\n".format(i, i), indent=10)
                self.add_content("</li>\n", indent=8)
        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)

        if search:
            self.add_content("<input type='text' placeholder='search' id='search'>\n", indent=4)
            self.add_content("<script type='text/javascript' src='js/search.js'></script>\n", indent=4)
            self.add_content("<button type='submit' onclick='myFunction()'>Search</button>\n", indent=4)
        self.add_content("</nav>\n")

    def make_table(self, headings=None, contents=None, heading_span=1, colors=None):
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
        self.add_content("<div class='container' style='margin-top:5em'>\n")
        self.add_content("<div class='table-responsive'>\n", indent=2)
        if heading_span > 1:
            self.add_content("<table class='table table-sm'>\n", indent=4)
        else:
            self.add_content("<table class='table table-striped table-sm'>\n", indent=4)
        self.add_content("<thead>\n", indent=6)
        self.add_content("<tr>\n", indent=8)
        for i in headings:
            self.add_content("<th colspan='{}'>{}</th>\n".format(heading_span, i), indent=10)
        self.add_content("</tr>\n", indent=8)
        self.add_content("<tbody>\n", indent=6)

        for num, i in enumerate(contents):
            self.add_content("<tr>\n", indent=8)
            if heading_span == 2:
                for j, col in zip(i, ["#ffffff"]+colors*(len(i)-1)):
                    self.add_content("<td style='background-color: {}'>{}</td>\n".format(col, j), indent=10)
                self.add_content("</tr>", indent=8)
            else:
                for j in i:
                    self.add_content("<td>{}</td>\n".format(j), indent=10)
                self.add_content("</tr>\n", indent=8)

        self.add_content("</tbody>\n", indent=6)
        self.add_content("</table>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

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

    def make_table_of_images(self, contents=None):
        """Generate a table of images in bootstrap format.

        Parameters
        ----------
        headings: list, optional
            list of headings
        contents: list, optional
            nd list giving the contents of the table.
        """
        self.add_content("<script type='text/javascript' src='../js/modal.js'></script>\n")
        self.add_content("<link rel='stylesheet' href='../css/image_styles.css'>\n")
        self.add_content("<div class='container' style='margin-top:5em; margin-bottom:5em;"
                         "background-color:#FFFFFF; box-shadow: 0 0 5px grey;'>\n")
        ind = 0
        for i in contents:
            self.add_content("<div class='row justify-content-center'>\n", indent=2)
            for num, j in enumerate(i):
                self.add_content("<div class='column'>\n", indent=4)
                self.add_content("<a href='#demo' data-slide-to='%s'>" %(ind), indent=6)
                self.add_content("<img src='{}'".format(self.base_url+"/plots/"+j.split("/")[-1]) +
                                 "alt='No image available' style='width:{}px;' "
                                 "id='{}' onclick='modal(\"{}\")'>\n".format(1050./len(i), j.split("/")[-1][:-4],
                                 j.split("/")[-1][:-4], indent=8))
                self.add_content("</a>", indent=6)
                self.add_content("</div>\n", indent=4)
                ind += 1
            self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def insert_image(self, path):
        """Generate an image in bootstrap format.

        Parameters
        ----------
        path: str, optional
            path to the image that you would like inserted
        """
        self.add_content("<div class='container' style='margin-top:5em; margin-bottom:5em;"
                         "background-color:#FFFFFF; box-shadow: 0 0 5px grey;'>\n") 
        self.add_content("<img src='{}' alt='No image available' "
                         "style='align-items:center; width:700px;'".format(path) +
                         "class='mx-auto d-block'>\n", indent=2)
        self.add_content("</p>\n", indent=2)
        self.add_content("<div style='clear: both;'></div>\n", indent=2)
        self.add_content("</div>\n")

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
        self.add_content("<div class='row justify-content-center'>\n")
        self.add_content("<div class='accordian' id='accordian' style='width:70%'>\n", indent=2)
        for num, i in enumerate(headings):
            self.add_content("<div class='card' style='border: 0px solid black'>\n", indent=4)
            self.add_content("<div class='card-header' id='{}'>\n".format(i), indent=6)
            self.add_content("<h5 class='mb-0'>\n", indent=8)
            self.add_content("<button class='btn btn-link collapsed' type='button' data-toggle='collapse' "
                             "data-target='#collapse{}' aria-expanded='false' ".format(i) +
                             "aria-controls='collapse{}'>\n".format(i), indent=10)
            self.add_content("{}\n".format(i), indent=12)
            self.add_content("</button>\n", indent=10)
            self.add_content("</h5>\n", indent=8)
            self.add_content("</div>\n", indent=6)
            self.add_content("<div id='collapse{}' class='collapse' ".format(i) +
                             "aria-labelledby='{}' data-parent='#accordian'>\n".format(i), indent=6)
            self.add_content("<div class='card-body'>\n", indent=8)
            self.add_content("<img src='{}' ".format(content[num]) +
                             "alt='No image available' style='width:700px;' " +
                             "class='mx-auto d-block'>\n", indent=10)
            self.add_content("</div>\n", indent=8)
            self.add_content("</div>\n", indent=6)
            self.add_content("</div>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def make_search_bar(self, popular_options=None, code="combine"):
        """Generate a search bar to combine the corner plots
        javascript.

        Parameters
        ----------
        popular_options: list, optional
            a list of popular options for your search bar
        """
        ids = "canvas" if code == "combine" else code
        self.add_content("<link rel='stylesheet' href='../css/side_bar.css'>\n")
        self.add_content("<script type='text/javascript' src='../js/combine_corner.js'></script>\n")
        self.add_content("<script type='text/javascript' src='../js/side_bar.js'></script>\n")
        self.add_content("<script type='text/javascript' src='../js/multiple_posteriors.js'></script>\n")
        self.add_content("<div class='w3-sidebar w3-bar-block w3-border-right sidenav' "
                         "style='display:none' id='mySidebar'>\n")
        self.add_content("<button onclick='side_bar_close()' class='close'>&times;</button>\n", indent=2)
        corner_parameters = ["luminosity_distance", "dec", "a_2",
                             "a_1", "geocent_time", "phi_jl", "psi", "ra", "phase",
                             "mass_2", "mass_1", "phi_12", "tilt_2", "iota",
                             "tilt_1", "chi_p", "chirp_mass", "mass_ratio",
                             "symmetric_mass_ratio", "total_mass", "chi_eff"]
        for i in corner_parameters:
            self.add_content("<input type='checkbox' name='type' "
                             "value='{}' id='{}' style='text-align: center; margin: 0 5px 0;'"
                             ">{}<br>\n".format(i, i,i,i), indent=2)
        self.add_content("</div>")
        self.add_content("<div class='row justify-content-center'>")
        self.add_content("<p style='margin-top:2.5em'> Input the parameter names that you would like to compare</p>", indent=2)
        self.add_content("</div>")
        self.add_content("<div class='row justify-content-center'>\n")
        self.add_content("<input type='text' placeholder='search' id='corner_search'>\n", indent=2)
        self.add_content("<button type='submit' onclick='{}()'>Submit</button>\n".format(code), indent=2)
        self.add_content("<button class='w3-button w3-teal w3-xlarge' "
                         "onclick='side_bar_open()'>&times </button>\n", indent=2) 
        self.add_content("</div>\n")
        self.add_content("<div class='row justify-content-center'>\n")
        if popular_options:
            for i in popular_options:
                self.add_content("<button type='button' class='btn btn-info' "
                                 "onclick='{}(\"{}\")' "
                                 "style='margin-left:0.25em; margin-right:0.25em; "
                                 "margin-top: 1.0em'>{}</button>\n".format(code, i, i), indent=2)
        self.add_content("</div>")
        self.add_content("<div class='container' style='margin-top:5em; margin-bottom:5em;"
                         "background-color:#FFFFFF; box-shadow: 0 0 5px grey;'>\n")  
        self.add_content("<div class='row justify-content-center' id='corner_plot'>\n", indent=2)
        self.add_content("<canvas id='{}' width='600' height='600'></canvas>\n".format(ids), indent=4)
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
        self.add_content("<div id='demo' class='carousel slide' data-ride='carousel'>\n", indent=6)
        self.add_content("<ul class='carousel-indicators'>\n", indent=8)
        for num, i in enumerate(images):
            if num == 0:
                self.add_content("<li data-target='#demo' data-slide-to-'%s' "
                                 "class='active'></li>\n" %(num), indent=10)
            self.add_content("<li data-target='#demo' data-slide-to-'%s'>"
                             "</li>\n" %(num), indent=10)
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
