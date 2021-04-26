# Licensed under an MIT style license -- see LICENSE.md

import pesummary
from pesummary.core.webpage import tables
from pesummary.core.webpage.base import Base

import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import time

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Edward Fauchon-Jones <edward.fauchon-jones@ligo.org>"
]

BOOTSTRAP = """<!DOCTYPE html>

<!--
                                    Made by
            ____  ___________
           / __ \/ ____/ ___/__  ______ ___  ____ ___  ____ ________  __
          / /_/ / __/  \__ \/ / / / __ `__ \/ __ `__ \/ __ `/ ___/ / / /
         / ____/ /___ ___/ / /_/ / / / / / / / / / / / /_/ / /  / /_/ /
        /_/   /_____//____/\__,_/_/ /_/ /_/_/ /_/ /_/\__,_/_/   \__, /
                                                               /____/

                                   MIT License

       PESummary was developed by Hoy et al. and source code can be seen
       here: git.ligo.org/lscsoft/pesummary. If you wish to use PESummary
   for your own work, please cite PESummary. The following page gives details
   https://lscsoft.docs.ligo.org/pesummary/stable_docs/citing_pesummary.html.
                                     Thanks!
  -->
<html lang='en'>
    <title>title</title>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'>
    stylesheet elements
</head>
<body style='background-color:#F8F8F8; margin-top:5em; min-height: 100%'>
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
    <script src='./js/html_to_csv.js'></script>
    <script src='./js/html_to_json.js'></script>
    <script src='./js/html_to_shell.js'></script>
    <script src='./js/expert.js'></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="./css/navbar.css">
    <link rel="stylesheet" href="./css/font.css">
    <link rel="stylesheet" href="./css/table.css">
    <link rel="stylesheet" href="./css/image_styles.css">
    <link rel="stylesheet" href="./css/watermark.css">
    <link rel="stylesheet" href="./css/toggle.css">
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
    <script src='../js/html_to_csv.js'></script>
    <script src='../js/html_to_json.js'></script>
    <script src='../js/html_to_shell.js'></script>
    <script src='../js/expert.js'></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="../css/navbar.css">
    <link rel="stylesheet" href="../css/font.css">
    <link rel="stylesheet" href="../css/table.css">
    <link rel="stylesheet" href="../css/image_styles.css">
    <link rel="stylesheet" href="../css/watermark.css">
    <link rel="stylesheet" href="../css/toggle.css">
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

    def _footer(self, user, rundir, fix_bottom=False):
        """
        """
        self.add_content("<script>")
        self.add_content("$(document).ready(function(){", indent=2)
        self.add_content("$('[data-toggle=\"tooltip\"]').tooltip();", indent=4)
        self.add_content("});", indent=2)
        self.add_content("</script>")
        if fix_bottom:
            self.make_div(
                _class='container', _style='bottom: 0px; '
                + 'top: 0px; min-height: 100%; left: 0; right: 0;'
                + 'min-width: 100%; padding-left: 0px; padding-right: 0px'
            )
            self.make_div(
                _class='jumbotron', _style='margin-bottom:0; line-height: 0.5;'
                + 'background-color:#989898; bottom:0; position:absolute;'
                + 'width:100%')
        else:
            self.make_div(
                _class='jumbotron', _style='margin-bottom:0; line-height: 0.5;'
                + 'background-color:#989898; bottom:0; position:bottom;'
                + 'width:100%')
        self.add_content("<div class='container'>")
        self.add_content("<div class='row'>", indent=2)
        self.add_content("<div class='col-sm-4 icon-bar'>", indent=4)
        self.add_content("<div class='icon'>", indent=6)
        self.add_content(
            "<a href='https://git.ligo.org/lscsoft/pesummary/tree/v{}' "
            "data-toggle='tooltip' title='View PESummary-v{}'>"
            "<i class='fa fa-code' style='font-size: 30px; color: #E8E8E8; "
            "font-weight: 900; padding-right:10px'></i></a>".format(
                pesummary.__short_version__, pesummary.__short_version__
            )
        )
        self.add_content(
            "<a href='https://git.ligo.org/lscsoft/pesummary/issues' "
            "data-toggle='tooltip' title='Open an issue ticket'>"
            "<i class='fa fa-ticket' style='font-size: 30px; color: #E8E8E8; "
            "font-weight: 900; padding-right:10px'></i></a>"
        )
        self.add_content(
            "<a href='https://lscsoft.docs.ligo.org/pesummary/' "
            "data-toggle='tooltip' title='View the docs!'>"
            "<i class='fa fa-book' style='font-size: 30px; color: #E8E8E8; "
            "font-weight: 900; padding-right:10px'></i></a>"
        )
        link = (
            "https://lscsoft.docs.ligo.org/pesummary/stable_docs/tutorials/"
            "make_your_own_page_from_metafile.html"
        )
        self.add_content(
            "<a href='{}' data-toggle='tooltip' title='Make your own page'>"
            "<i class='fa fa-window-restore' style='font-size: 30px; "
            "color: #E8E8E8; font-weight: 900; padding-right:10px'></i>"
            "</a>".format(link)
        )
        self.add_content("</div>", indent=6)
        self.add_content("</div>", indent=4)
        self.add_content("<div class='col-sm-7'>", indent=4)
        self.add_content(
            "<p style='color: #E8E8E8; font-weight: bold; "
            "font-family: arial-body; margin-top:14px'>This page was produced "
            "by {} at {} on {}</p>".format(
                user, time.strftime("%H:%M"), time.strftime("%B %d %Y")
            )
        )
        self.add_content("</div>", indent=4)
        self.add_content("</div>", indent=2)
        self.add_content("</div>")
        if fix_bottom:
            self.add_content("</div>")
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

    def make_footer(self, user=None, rundir=None, fix_bottom=False):
        """Make footer for document in bootstrap format.
        """
        self._footer(user, rundir, fix_bottom=fix_bottom)

    def make_banner(
        self, approximant=None, key="Summary", _style=None, link=None,
        custom=""
    ):
        """Make a banner for the document.
        """
        self.make_div(indent=2, _class='banner', _style=_style)
        self.add_content("%s" % (approximant))
        self.end_div()
        self.make_div(indent=2, _class='paragraph')
        if key == "Summary":
            self.add_content(
                "The figures below show the summary plots for the run")
        elif key == "config":
            self.add_content(
                "Below is the config file for %s" % (approximant))
        elif key == "prior":
            self.add_content(
                "Below is the prior file for %s" % (approximant))
        elif key == "corner":
            self.add_content(
                "Below is the custom corner plotter for %s" % (approximant))
        elif key == "additional":
            self.add_content(
                "Below we show plots which have been generated previously "
                "and passed to pesummary via the `--add_existing_plot` "
                "command line argument"
            )
        elif key == "interactive_corner":
            self.add_content(
                "Below are interative corner plots for %s. Simply use the "
                "Box Select to select the points you wish to look at" % (
                    approximant
                )
            )
        elif key == "Comparison":
            self.add_content(
                "Below are the summary comparison plots")
        elif key == "Version":
            self.add_content(
                "Below is the version information for all files passed and the "
                "PESummary version used to generate these pages")
        elif key == "Publication":
            self.add_content(
                "Below are publication quality plots for the passed result "
                "files")
        elif key == "Logging":
            self.add_content("Below is the output from the PESummary code")
        elif key == "Notes":
            self.add_content("Below are your custom notes")
        elif key == "classification":
            self.add_content(
                "Below we look at the source probabilities and plots for the "
                "passed result file")
        elif key == "sampler_kwargs":
            self.add_content(
                "Sampler information extracted from each result file")
        elif key == "meta_data":
            self.add_content(
                "Meta data extracted from each result file")
        elif key == "summary_table":
            self.add_content(
                "Table summarising the key data for each posterior "
                "distribution."
            )
        elif key == "ks_test":
            self.add_content(
                "Table summarising the Kolmogorov-Smirnov p-values for each "
                "posterior distribution. 0 means the samples from analysis A "
                "are not drawn from analysis B, and 1 means the samples from "
                "analysis A are drawn from analysis B"
            )
        elif key == "js_test":
            self.add_content(
                "Table summarising the Jensen-Shannon divergence for each "
                "posterior distributions. 0 means the distributions are "
                "identical and 1 means maximal divergence"
            )
        elif key == "command_line":
            if link is not None:
                self.add_content(
                    "This page was generated with the following command-line "
                    "call from the directory %s" % (link)
                )
            else:
                self.add_content(
                    "This page was generated with the following command-line "
                    "call:"
                )
        elif key == "detchar":
            base_string = "Below are summary plots for the detector %s.{}" % (
                approximant
            )
            if link is not None:
                base_string = base_string.format(
                    " For more details see the LIGO summary pages "
                    "<a href=%s>here</a>" % (link)
                )
            else:
                base_string = base_string.format("")
            self.add_content(base_string)
        elif key == "Downloads":
            self.add_content(
                "Below are links to download all relevant information"
            )
        elif key == "About":
            self.add_content(
                "Below is information about how these pages were generated"
            )
        elif key == "custom":
            self.add_content(custom)
        else:
            self.add_content(
                "The figures below show the plots for %s" % (approximant))
        self.end_div()

    def make_navbar(self, links=None, samples_path="./samples", search=True,
                    histogram_download=None,
                    background_color="navbar-dark",
                    hdf5=False, about=True, toggle=False):
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
                                        if "external:" in k[key]:
                                            self.add_content(
                                                "<li class='dropdown-item' "
                                                "href='#' onclick='window.location"
                                                "=\"{}\"'><a>{}</a></li>\n".format(
                                                    k[key].split("external:")[1],
                                                    key
                                                ), indent=18
                                            )
                                        else:
                                            self.add_content(
                                                "<li class='dropdown-item' "
                                                "href='#' onclick='grab_html"
                                                "(\"{}\", label=\"{}\")'>"
                                                "<a>{}</a></li>\n".format(
                                                    key, k[key], key
                                                ), indent=18
                                            )
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
                                        if "external:" in k[key]:
                                            self.add_content(
                                                "<li class='dropdown-item' "
                                                "href='#' onclick='window.location"
                                                "=\"{}\"'><a>{}</a></li>\n".format(
                                                    k[key].split("external:")[1],
                                                    key
                                                ), indent=18
                                            )
                                        else:
                                            self.add_content(
                                                "<li class='dropdown-item' "
                                                "href='#' onclick='grab_html"
                                                "(\"{}\", label=\"{}\")'>"
                                                "<a>{}</a></li>\n".format(
                                                    key, k[key], key
                                                ), indent=14
                                            )

                                    else:
                                        self.add_content(
                                            "<li class='dropdown-item' href='#' "
                                            "onclick='grab_html(\"{}\")'>"
                                            "<a>{}</a></li>\n".format(k, k), indent=14)

                        else:
                            if type(j[0]) == dict:
                                key = list(j[0].keys())[0]
                                if 'external:' in j[0][key]:
                                    self.add_content(
                                        "<li class='dropdown-item' href='#' "
                                        "onclick='window.location=\"{}\"'>"
                                        "<a>{}</a></li>\n".format(
                                            j[0][key].split("external:")[1], key
                                        ), indent=14
                                    )
                                else:
                                    self.add_content(
                                        "<li class='dropdown-item' href='#' "
                                        "onclick='grab_html(\"{}\", label=\"{}\")'>"
                                        "<a>{}</a></li>\n".format(key, j[0][key], key),
                                        indent=14
                                    )

                            else:
                                if "external:" in j[0]:
                                    self.add_content(
                                        "<li class='dropdown-item' href='#' "
                                        "onclick='window.location=\"{}\"'>"
                                        "<a>{}</a></li>\n".format(
                                            j[0].split("external:")[1], j[0]
                                        ), indent=14
                                    )
                                else:
                                    self.add_content(
                                        "<li class='dropdown-item' href='#' "
                                        "onclick='grab_html(\"{}\")'>"
                                        "<a>{}</a></li>\n".format(j[0], j[0]),
                                        indent=14
                                    )

                self.add_content("</ul>\n", indent=12)
                self.add_content("</li>\n", indent=10)
            else:
                self.add_content("<li class='nav-item'>\n", indent=8)
                if i == "home":
                    if "external:" in i:
                        self.add_content(
                            "<a class='nav-link' href='#' onclick='window.location"
                            "=\"{}\"'>{}</a>\n".format(i.split("external:")[1], i),
                            indent=10)
                    else:
                        self.add_content("<a class='nav-link' "
                                         "href='#' onclick='grab_html(\"{}\")'"
                                         ">{}</a>\n".format(i, i), indent=10)
                else:
                    if type(i) == dict:
                        key = list(i.keys())[0]
                        if "external:" in i[key]:
                            self.add_content(
                                "<a class='nav-link' "
                                "href='#' onclick='window.location=\"{}\"'"
                                ">{}</a>\n".format(
                                    i[key].split("external:")[1], key
                                ), indent=10)
                        else:
                            self.add_content(
                                "<a class='nav-link' "
                                "href='#' onclick='grab_html(\"{}\", label=\"{}\")'"
                                ">{}</a>\n".format(key, i[key], key), indent=10)

                    else:
                        if "external:" in i:
                            self.add_content(
                                "<a class='nav-link' "
                                "href='#' onclick='window.location=\"{}\"'"
                                ">{}</a>\n".format(
                                    i.split("external:")[1], i
                                ), indent=10)
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

        self.add_content("<div class='collapse navbar-collapse' id='collapsibleNavbar'>\n", indent=4)
        self.add_content(
            "<ul class='navbar-nav flex-row ml-md-auto d-none d-md-flex'"
            "style='margin-right:1em;'>\n", indent=6)
        if toggle:
            self.add_content(
                "<div style='margin-top:0.5em; margin-right: 1em;' "
                "data-toggle='tooltip' title='Activate expert mode'>", indent=6
            )
            self.add_content("<label class='switch'>", indent=8)
            self.add_content(
                "<input type='checkbox' onchange='show_expert_div()'>", indent=10
            )
            self.add_content("<span class='slider round'></span>", indent=10)
            self.add_content("</label>", indent=8)
            self.add_content("</div>", indent=6)
        self.add_content(
            "<a class='nav-link' href='#', onclick='grab_html(\"{}\")'"
            ">{}</a>\n".format("Downloads", "Downloads"), indent=2
        )
        if about:
            self.add_content(
                "<a class='nav-link' href='#', onclick='grab_html(\"{}\")'"
                ">{}</a>\n".format("About", "About"), indent=2
            )
        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)
        if search:
            self.add_content("<input type='text' placeholder='search' id='search'>\n", indent=4)
            self.add_content("<button type='submit' onclick='myFunction()'>Search</button>\n", indent=4)
        self.add_content("</nav>\n")

    def make_table(self, headings=None, contents=None, heading_span=1,
                   colors=None, accordian_header="Summary Table",
                   accordian=True, format="table-striped table-sm",
                   sticky_header=False, scroll_table=False, **kwargs):
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
        if sticky_header:
            self.add_content(
                "<style>\n"
                ".header-fixed > tbody > tr > td,\n"
                ".header-fixed > thead > tr > th {\n"
            )
            self.add_content("    width: {width}%;\n".format(
                width=100. / len(headings)
            ))
            self.add_content(
                "    float: left;\n}\n</style>"
            )

        if scroll_table:
            self.add_content(
                "<style>\n"
                ".scroll-table tbody, tr, th {\n"
                "    width: 100%;\n"
                "    float: left;\n}\n"
                ".scroll-table td {\n"
            )
            self.add_content("    width: {}%;\n".format(100 / len(headings)))
            self.add_content(
                "    float: left;\n}\n</style>"
            )

        if accordian:
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
        self.make_div(_class='container', _style='max-width:1400px', indent=18)
        if type(contents) == list:
            self.make_div(indent=20, _class='table-responsive')
            if heading_span > 1:
                self.add_content(
                    "<table class='table table-sm' style='max-width:1400px'>\n",
                    indent=22
                )
            else:
                self.add_content(
                    "<table class='table %s' style='max-width:1400px'>\n" % (format),
                    indent=24
                )
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
                self.add_content(
                    "<table class='table %s' style='max-width:1400px'>\n" % (format),
                    indent=24
                )
            self.add_content("<tbody>\n", indent=26)
            for j in contents.keys():
                self.add_content("<tr bgcolor='#F0F0F0'>\n", indent=28)
                self.add_content(
                    "<th colspan='{}' style='width: 100%' class='text-center'>"
                    "{}</th>\n".format(len(contents[j][0]), j), indent=30)
                self.add_content("</tr>\n", indent=28)
                self.add_content("<tr>\n", indent=28)
                for i in headings:
                    self.add_content(
                        "<td colspan='{}' class='text-center'>"
                        "{}</td>\n".format(heading_span, i), indent=30)
                self.add_content("</tr>\n", indent=28)

                for num, i in enumerate(contents[j]):
                    self.add_content("<tr>\n", indent=28)
                    for j in i:
                        self.add_content(
                            "<td>{}</td>\n".format(j), indent=30
                        )
                    self.add_content("</tr>\n", indent=28)
            self.add_content("</tbody>\n", indent=26)
            self.add_content("</table>\n", indent=24)
            self.end_div(indent=20)

        self.end_div(indent=18)
        if accordian:
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
        styles += ".highlight {margin: 5px; padding: 10px; background: #FFFFFF}"
        return styles

    def make_table_of_images(self, contents=None, rows=None, columns=None,
                             code="modal", cli=None, autoscale=False,
                             unique_id=None, captions=None, extra_div=False,
                             mcmc_samples=False, margin_left=None, display=None,
                             container_id=None, **kwargs):
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
                                       code=code, cli=cli, autoscale=autoscale,
                                       unique_id=unique_id, captions=captions,
                                       extra_div=extra_div, display=display,
                                       mcmc_samples=mcmc_samples,
                                       margin_left=margin_left,
                                       container_id=container_id, **kwargs)
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
                                 ">{}<br>\n".format(i, i, i), indent=2)
        self.add_content("</div>")
        self.add_content("<div class='row justify-content-center'>")
        self.add_content(
            "<p style='margin-top:2.5em; font-family: Arial-body; font-size:14px'>"
            " Input the parameter names that you would like to compare</p>", indent=2)
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
        self.make_container()
        self.add_content("<div class='row justify-content-center' id='corner_plot'>\n", indent=2)
        self.add_content("<canvas id='{}' width='1000' height='1000'></canvas>\n".format(ids), indent=4)
        if code == "combine":
            self.add_content("<img src='' id='mirror'/>", indent=4)
        self.add_content("</div>\n", indent=2)
        self.end_container()

    def make_modal_carousel(self, images=None, unique_id=None):
        """Make a pop up window that appears on top of the home page showing
        images in a carousel.

        Parameters
        ----------
        images: list
            list of image locations that you would like included in the
            carousel
        """
        if unique_id is not None:
            modal_id = "Modal_{}".format(unique_id)
            demo_id = "demo_{}".format(unique_id)
        else:
            modal_id = "MyModal"
            demo_id = "demo"
        self.add_content("<div class='modal fade bs-example-modal-lg' tabindex='-1' "
                         "role='dialog' aria-labelledby='myLargeModalLabel' "
                         "aria-hidden='true' id='{}' style='margin-top: 200px;'>"
                         "\n".format(modal_id))
        self.add_content("<div class='modal-dialog modal-lg' style='width:90%'>\n", indent=2)
        self.add_content("<div class='modal-content'>\n", indent=4)
        self.add_content("<div id='{}' class='carousel slide' data-ride='carousel'"
                         " data-interval='false'>\n".format(demo_id), indent=6)
        self.add_content("<ul class='carousel-indicators'>\n", indent=8)
        for num, i in enumerate(images):
            if num == 0:
                self.add_content("<li data-target='#%s' data-slide-to-'%s' "
                                 "class='active'></li>\n" % (demo_id, num), indent=10)
            self.add_content("<li data-target='#%s' data-slide-to-'%s'>"
                             "</li>\n" % (demo_id, num), indent=10)
        self.add_content("<li data-target='#{}' data-slide-to='0' "
                         "class='active'></li>\n".format(demo_id), indent=10)
        self.add_content(
            "<li data-target='#{}' data-slide-to='1'></li>\n".format(demo_id),
            indent=10
        )
        self.add_content(
            "<li data-target='#{}' data-slide-to='2'></li>\n".format(demo_id),
            indent=10
        )
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
        self.add_content("<a class='carousel-control-prev' href='#{}' "
                         "data-slide='prev'>\n".format(demo_id), indent=8)
        self.add_content("<span class='carousel-control-prev-icon'></span>\n", indent=10)
        self.add_content("</a>\n", indent=8)
        self.add_content("<a class='carousel-control-next' href='#{}' "
                         "data-slide='next'>\n".format(demo_id), indent=8)
        self.add_content("<span class='carousel-control-next-icon'></span>\n", indent=10)
        self.add_content("</a>\n", indent=8)
        self.add_content("</div>\n", indent=6)
        self.add_content("</div>\n", indent=4)
        self.add_content("</div>\n", indent=2)
        self.add_content("</div>\n")

    def make_cli_button(self, cli):
        """Make a button showing the command line used

        Parameters
        ----------
        cli: str
            the command line that you wish to display in the modal
        """
        self.add_content("<style>")
        self.add_content(".popover {", indent=2)
        self.add_content("max-width: 550px;", indent=4)
        self.add_content("width: 550px;", indent=4)
        self.add_content("}", indent=2)
        self.add_content("</style>")
        self.make_div(2, _class="imgButton", _style=None)
        self.add_content("<button value='test' class='btn btn-info btn-xs' "
                         "style='cursor: pointer' "
                         "data-toggle='popover' data-placement='top' "
                         "data-content='%s'>Command Line</button>" % (cli),
                         indent=12)
        self.end_div(0)

    def make_watermark(self, text="Preliminary"):
        """Add a watermark to the html page

        Parameters
        ----------
        text: str
            work you wish to use as a watermark
        """
        self.add_content("<div id='background'>")
        for _ in range(3):
            self.add_content(
                "<p id='bg-text'>{} {}</p>".format(text, text)
            )
        self.end_div()

    def export(
        self, filename, csv=True, json=False, shell=False, histogram_dat=None,
        requirements=False, conda=False, margin_top="-4em", margin_bottom="5em",
    ):
        """Make a button which to export a html table to csv

        Parameters
        ----------
        filename: str
            the name of the file you wish to save the data too
        """
        if "." in filename:
            basename = ".".join(filename.split(".")[:-1]) + ".{}"
        else:
            basename = filename + ".{}"
        self.add_content(
            "<div class='container' style='margin-top:{}; margin-bottom:{}; "
            "max-width: 1400px'>".format(margin_top, margin_bottom)
        )
        json_margin = "0em"
        bash_margin = "0.5em"
        if csv and json:
            json_margin = "0.5em"
            bash_margin = "0em"
            self.add_content("<div class='row' style='margin-left: 0.2em'>")
        if csv:
            self.add_content(
                "<button type='button' onclick='export_table_to_csv(\"{}\")' "
                "class='btn btn-outline-secondary btn-table'>Export to CSV"
                "</button>".format(basename.format("csv"))
            )
        if requirements:
            self.add_content(
                "<button type='button' onclick='export_table_to_pip(\"{}\")' "
                "class='btn btn-outline-secondary btn-table'>Export to pip"
                "</button>".format(basename.format("txt"))
            )
        if conda:
            self.add_content(
                "<button type='button' onclick='export_table_to_conda(\"{}\")' "
                "class='btn btn-outline-secondary btn-table'>Export to conda"
                "</button>".format(basename.format("txt"))
            )
        if json:
            self.add_content(
                "<button type='button' onclick='export_table_to_json(\"{}\")' "
                "style='margin-left: {}' class='btn btn-outline-secondary "
                "btn-table'>Export to JSON</button>".format(
                    basename.format("json"), json_margin
                )
            )
        if shell:
            self.add_content(
                "<button type='button' onclick='export_table_to_shell(\"{}\")' "
                "style='margin-left: {}; margin-bottom: {}' "
                "class='btn btn-outline-secondary btn-table'>Export to bash"
                "</button>".format(
                    basename.format("sh"), json_margin, bash_margin
                )
            )
        if histogram_dat is not None:
            self.add_content("<a href='%s' download>" % (histogram_dat))
            self.add_content(
                "<button type='button' style='margin-left: {}; margin-bottom: "
                "{}' class='btn btn-outline-secondary btn-table'>Export to dat"
                "</button>".format(json_margin, margin_bottom)
            )
            self.add_content("</a>")
        if csv and json:
            self.end_div()
        self.end_div(0)
