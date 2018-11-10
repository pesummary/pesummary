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

import utils
import shutil

def make_html(web_dir, title="Summary Pages", pages=["mass1", "corner"]):
    """Make the initial html page. 

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    title: str, optional
        header title of html page
    """
    f = open(web_dir+"/home.html", "w")
    utils.make_dir(web_dir+"/html")
    if "home" not in pages:
        pages.append("home")
    for i in pages:
        if i != "home":
            i = "html/" + i
        f = open("{}/{}.html".format(web_dir, i), "w")
        doc_type = "<!DOCTYPE html>\n"
        bootstrap = "<html lang='en'>\n" + \
                    "  <title>{}</title>\n".format(title) + \
                    "  <meta charset='utf-8'>\n" + \
                    "  <meta name='viewport' content='width=device-width, initial-scale=1'>\n" + \
                    "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'>\n" + \
                    "  <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>\n" + \
                    "  <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>\n" + \
                    "  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>\n" + \
                    "</head>\n"
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

    def _header(self, title, colour):
        """
        """
        self.add_content("<div class='jumbotron text-center' style='background-color: {}; margin-bottom:0'>\n".format(colour))
        self.add_content("  <h1>{}</h1>\n".format(title))
        self.add_content("</div>\n")

    def _footer(self, user, rundir):
        """
        """
        self.add_content("<div class='jumbotron text-center' style='margin-bottom:0'>\n")
        self.add_content("<p>Simulation run by {}. Run directories found at {}</p>\n".format(user, rundir), indent=2)
        self.add_content("<p>Command line: fjndsjvnbfdjvndf</p>\n", indent=2)
        self.add_content("</div>\n")

    def _setup_navbar(self):
        self.add_content("<nav class='navbar navbar-expand-sm navbar-dark bg-dark'>\n")
        self.add_content("<a class='navbar-brand' href='#'>Navbar</a>\n", indent=2)
        self.add_content("<button class='navbar-toggler' type='button' "
                         "data-toggle='collapse' data-target='#collapsibleNavbar'>\n", indent=2)
        self.add_content("<span class='navbar-toggler-icon'></span>\n", indent=4)
        self.add_content("</button>\n", indent=2)

    def make_header(self, title="Parameter Estimation Summary Pages", background_colour="#eee"):
        """Make header for document in bootstrap format.

        Parameters
        ----------
        title: str, optional
            header title of html page
        """
        self._header(title, background_colour)

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
                self.add_content("<div class='dropdown-menu' aria-labelledby='navbarDropdown'>\n", indent=12)
                for j in i[1]:
                        self.add_content("<a class='dropdown-item' "
                                         "href='{}/html/{}.html'>{}</a>\n".format(self.base_url, j, j), indent=14)
                self.add_content("</div>\n", indent=12)
                self.add_content("</li>\n", indent=10)  
            else:
                self.add_content("<li class='nav-item'>\n", indent=8)
                if i == "home":
                    self.add_content("<a class='nav-link' "
                                     "href='{}/{}.html'>{}</a>\n".format(self.base_url, i, i), indent=10)
                else:
                    self.add_content("<a class='nav-link' "
                                     "href='{}/html/{}.html'>{}</a>\n".format(self.base_url, i, i), indent=10)
                self.add_content("</li>\n", indent=8)
        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)

        if search:
            self.add_content("<input type='text' placeholder='search' id='search'>\n", indent=4)
            self.add_content("<script type='text/javascript' src='js/search.js'></script>\n", indent=4)
            self.add_content("<button type='submit' onclick='myFunction()'>Search</button>", indent=4)
        self.add_content("</nav>")

    def make_table_of_images(self, headings=None, contents=None):
        """Generate a table in bootstrap format.

        Parameters
        ----------
        headings: list, optional
            list of headings
        contents: list, optional
            nd list giving the contents of the table.
        """
        for i in contents:
            if len(headings) != len(i):
                raise Exception("Ensure that your contents match the headings")
        self.add_content("<div class='container' style='margin-top:30p'>\n")
        self.add_content("<div class='table-responsive'>\n", indent=2)
        self.add_content("<table class='table'>\n", indent=4)
        self.add_content("<thead>", indent=6)
        self.add_content("<tr>", indent=8)
        for i in headings:
            self.add_content("<th>{}</th>\n".format(i), indent=10)
        self.add_content("</tr>\n", indent=8)
        self.add_content("<tbody>\n", indent=6)

        for i in contents:
            for j in i:
                self.add_content("<th><img src='{}' ".format(self.base_url+"/plots/"+j.split("/")[-1]) +
                                 "alt='No image available' style='width:350px;'></td>\n")

        self.add_content("</tbody>\n", indent=6)
        self.add_content("</table>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def insert_image(self, path, indent=0):
        """Generate an image in bootstrap format.

        Parameters
        ----------
        path: str, optional
            path to the image that you would like inserted
        """
        shutil.copyfile(path, self.base_url + "/plots/" + path.split("/")[-1])
        self.add_content("<img src='{}' ".format(self.base_url+"/plots/"+path.split("/")[-1]) +
                         "alt='No image available' style='width:350px;'>\n", indent=indent)
