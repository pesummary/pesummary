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

import os
import socket
import shutil
from glob import glob

from pesummary.webpage import webpage

import pytest


class TestWebpage(object):

    def setup(self):
        directory = './.outdir'
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

    def test_make_html(self):
        webdir = "./.outdir"
        assert os.path.isfile("./.outdir/home.html") == False
        webpage.make_html(webdir, pages=["home"])
        assert os.path.isfile("./.outdir/home.html") == True

    def test_open_html(self):
        webdir = "./.outdir" 
        baseurl = "https://example"
        open("./.outdir/home.html", "a").close()
        f = webpage.open_html(webdir, baseurl, "home")
        assert isinstance(f, webpage.page) == True


class TestPage(object):

    def setup(self):
        webdir = "./.outdir" 
        baseurl = "https://example"
        webpage.make_html(webdir, pages=["home"])
        self.html = webpage.open_html(webdir, baseurl, "home")

    def open_and_read(self, path):
        f = open(path)
        f = f.readlines()
        return f

    def test_add_content(self):
        content = "testing\n" 
        self.html.add_content(content, indent=2)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        assert any(elem == "  testing\n" for elem in f) == True

    def test_header(self):
        self.html.make_header(title="My title", approximant="approx")
        self.html.close()
        f = self.open_and_read("./.outdir/home.html") 
        assert any("<div class='jumbotron text-center'" in elem for elem in f) == True
        assert any(elem == "  <h1 id=approx>My title</h1>\n" for elem in f) == True

    def test_footer(self):
        self.html.make_footer()
        self.html.close()
        f = self.open_and_read("./.outdir/home.html") 
        assert any("<div class='jumbotron text-center'" in elem for elem in f) == True
        assert any("<p>Simulation run by" in elem for elem in f) == True

    @pytest.mark.parametrize('links', [("other", "example")])
    def test_navbar(self, links):
        self.html.make_navbar(links)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="<a class='nav-link' href='#' onclick='grab_html(\"other\")'>other</a>"
        string2="<a class='nav-link' href='#' onclick='grab_html(\"example\")'>example</a>"
        assert any(string1 in elem for elem in f) == True
        assert any(string2 in elem for elem in f) == True

    @pytest.mark.parametrize('headings, contents', [(["column1", "column2"],
        [["entry1", "entry2"], ["entry3", "entry4"]]),])
    def test_table(self, headings, contents):
        self.html.make_table(headings=headings, contents=contents)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="          <th colspan='1'>column1</th>"
        string2="          <th colspan='1'>column2</th>"
        string3="          <td>entry2</td>"
        string4="          <td>entry4</td>"
        for i in [string1, string2, string3, string4]:
            assert any(i in elem for elem in f) == True

    @pytest.mark.parametrize('language', [('ini'),]) 
    def test_code_block(self, language):
        with open("tests/files/example.ini", 'r') as f:
            contents = f.read()
        styles = self.html.make_code_block(language='ini', contents=contents)
        with open('./.outdir/example_config.css', 'w') as f:
            f.write(styles)
            f.close()
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="<div class=\"highlight\"><pre><span></span><span class=\"k\">[engine]</span>"
        string2="<span class=\"na\">example</span><span class=\"o\">=</span>"
        assert any(string1 in elem for elem in f) == True
        assert any(string2 in elem for elem in f) == True
        assert any("example_config.css" in elem for elem in glob("./.outdir/*")) == True

    def test_table_of_images(self):
        contents = [["image1.png"], ["image2.png"]]
        self.html.make_table_of_images(contents=contents)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="<img src=\'https://example/plots/image1.png\'alt=\'No image available\'"
        string2="<img src=\'https://example/plots/image2.png\'alt=\'No image available\'"
        string3="<div class='column'>\n"
        for i in [string1, string2, string3]:
            assert any(i in elem for elem in f) == True

    def test_insert_image(self):
        path = "./path/to/image.png"
        self.html.insert_image(path)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html") 
        string1="<img src='./path/to/image.png' alt='No image available'"
        assert any(string1 in elem for elem in f) == True

    def test_accordian(self):
        headings = ["example"]
        content = ["./path/to/image.png"]
        self.html.make_accordian(headings=headings, content=content)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="<img src='./path/to/image.png' alt='No image available'"
        string2="<button class='btn btn-link collapsed' type='button' data-toggle='collapse'"
        string3="<div id='collapseexample' class='collapse' aria-labelledby='example'"
        string4="<div class='accordian' id='accordian'"
        for i in [string1, string2, string3, string4]:
            assert any(i in elem for elem in f) == True

    def test_search_bar(self):
        self.html.make_search_bar()
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        # check that the scripts are there
        scripts = ["combine_corner.js", "side_bar.js", "multiple_posteriors.js"]
        for i in scripts:
            string = "<script type='text/javascript' src='../js/{}'></script>\n".format(i)
            assert any(string in elem for elem in f) == True
        string1="<button type='submit' onclick='combine()'>Submit</button>\n"
        assert any(string1 in elem for elem in f)

    def test_modal_carousel(self):
        images = ["./path/to/image.png"]
        self.html.make_modal_carousel(images=images)
        self.html.close()
        f = self.open_and_read("./.outdir/home.html")
        string1="<div class='modal-dialog modal-lg' style='width:90%'>"
        string2="<div id='demo' class='carousel slide' data-ride='carousel'>"
        string3="<div class='carousel-item active'>"
        string4="<img src=./path/to/image.png style='align-items:center;' "
        print(f)
        for i in [string1, string2, string3, string4]:
            assert any(i in elem for elem in f) == True
