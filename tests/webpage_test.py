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

from bs4 import BeautifulSoup

import pytest


class TestWebpage(object):

    def setup(self):
        directory = './.outdir'
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

    def teardown(self):
        directory = './.outdir'
        try:
            shutil.rmtree(directory)
        except:
            pass

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
        try:
            os.mkdir(webdir)
        except:
            shutil.rmtree(webdir)
            os.mkdir(directory)
        os.mkdir("./.outdir/css")
        f = open("./.outdir/css/command_line.css", "w")
        f.close()
        baseurl = "https://example"
        webpage.make_html(webdir, pages=["home"])
        self.html = webpage.open_html(webdir, baseurl, "home")

    def teardown(self):
        directory = './.outdir'
        try:
            shutil.rmtree(directory)
        except:
            pass

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
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        assert str(soup.h1.string) == 'My title'
        assert soup.div["class"] == ['jumbotron', 'text-center']

    def test_footer(self):
        self.html.make_footer()
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        assert "This page was produced by" in str(soup.p)
        assert soup.div["class"] == ['jumbotron']

    @pytest.mark.parametrize('links', [("other", "example")])
    def test_navbar(self, links):
        self.html.make_navbar(links)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        all_links = soup.find_all("a", class_="nav-link")
        assert len(all_links) == 2
        assert all_links[0].text == "other"
        assert all_links[1].text == "example"

    @pytest.mark.parametrize('headings, contents', [(["column1", "column2"],
        [["entry1", "entry2"], ["entry3", "entry4"]]),])
    def test_table(self, headings, contents):
        self.html.make_table(headings=headings, contents=contents)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        columns = soup.find_all("th")
        assert len(columns) == 2
        assert columns[0].text == "column1"
        assert columns[1].text == "column2"
        entries = soup.find_all("td")
        assert len(entries) == 4
        for num, i in enumerate(entries):
            assert i.text == "entry{}".format(num+1)

    @pytest.mark.parametrize('language', [('ini'),]) 
    def test_code_block(self, language):
        with open("tests/files/example.ini", 'r') as f:
            contents = f.read()
        styles = self.html.make_code_block(language='ini', contents=contents)
        with open('./.outdir/example_config.css', 'w') as f:
            f.write(styles)
            f.close()
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        assert soup.div["class"] == ["highlight"]
        all_entries = soup.find_all("span")
        assert all_entries[1].text == "[engine]"
        assert all_entries[2].text == "example"

    def test_table_of_images(self):
        contents = [["image1.png"], ["image2.png"]]
        self.html.make_table_of_images(contents=contents)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        all_images = soup.find_all("img")
        assert len(all_images) == 2
        assert all_images[0]["src"] == "image1.png"
        assert all_images[1]["src"] == "image2.png"

    def test_insert_image(self):
        path = "./path/to/image.png"
        self.html.insert_image(path)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        all_images = soup.find_all("img")
        assert len(all_images) == 1
        assert all_images[0]["src"] == "./path/to/image.png"

    def test_accordian(self):
        headings = ["example"]
        content = ["./path/to/image.png"]
        self.html.make_accordian(headings=headings, content=content)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        assert soup.img["src"] == "./path/to/image.png"
        assert soup.button["class"] == ["btn", "btn-link", "collapsed"]
        assert len(soup.find_all("div", class_="accordian")) == 1
        assert "example" in soup.button.text

    def test_search_bar(self):
        self.html.make_search_bar()
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        scripts = ["combine_corner.js", "side_bar.js", "multiple_posteriors.js"]
        html_scripts = soup.find_all("script")
        for i in scripts:
            assert any("../js/%s" %(i) in elem["src"] for elem in html_scripts)
        assert soup.find_all("button", class_="")[0].text == "Submit"

    def test_modal_carousel(self):
        images = ["./path/to/image.png"]
        self.html.make_modal_carousel(images=images)
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        images = soup.find_all("img")
        assert len(images) == 1
        assert images[0]["src"] == "./path/to/image.png"
        assert len(soup.find_all("div", class_="carousel-item")) == 1
        assert len(soup.find_all("div", class_="modal-lg")) == 1
        assert len(soup.find_all("div", class_="carousel")) == 1
