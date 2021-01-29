# Licensed under an MIT style license -- see LICENSE.md

import os
import socket
import shutil
from glob import glob
import numpy as np

from pesummary.core.webpage import webpage
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from .base import data_dir

from bs4 import BeautifulSoup

import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
            os.mkdir(webdir)
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
        self.html.make_header(approximant="approx")
        self.html.close()
        with open("./.outdir/home.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
        assert str(soup.h7.string) == 'None'

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
        assert len(all_links) == 4
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
        with open(data_dir + "/example.ini", 'r') as f:
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


class TestWebpage(object):
    """
    """
    def setup(self):
        """
        """
        from pesummary.gw.webpage.main import _WebpageGeneration

        self.labels = ["one", "two"]
        self.samples = MultiAnalysisSamplesDict({
            label: {
                param: np.random.uniform(0.2, 1.0, 1000) for param in
                ["chirp_mass", "mass_ratio"]
            } for label in self.labels
        })
        self.webpage = _WebpageGeneration(
            webdir=".outdir", labels=self.labels, samples=self.samples,
            pepredicates_probs={label: None for label in self.labels},
            same_parameters=["chirp_mass", "mass_ratio"]
        )
        self.webpage.generate_webpages()

    def test_comparison_stats(self):
        """
        """
        from scipy.spatial.distance import jensenshannon
        from pesummary.utils.utils import jensen_shannon_divergence
        from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde

        comparison_stats = self.webpage.comparison_stats
        for param in ['chirp_mass', 'mass_ratio']:
            if param == "chirp_mass":
                xlow = 0.
                xhigh = None
            else:
                xlow = 0.
                xhigh = 1.
            js = comparison_stats[param][1][0][1]
            samples = [
                self.samples[label][param] for label in self.labels
            ]
            x = np.linspace(np.min(samples), np.max(samples), 100)
            _js = jensenshannon(
                Bounded_1d_kde(samples[0], xlow=xlow, xhigh=xhigh)(x),
                Bounded_1d_kde(samples[1], xlow=xlow, xhigh=xhigh)(x)
            )**2
            np.testing.assert_almost_equal(js, _js, 5)

    def test_displayed_label_summary_table(self):
        """Test that the summary table displayed on the webpages show the
        correct information
        """
        for label in self.labels:
            with open("./.outdir/html/{}_{}.html".format(label, label)) as fp:
                soup = BeautifulSoup(fp, features="html.parser")
                table = soup.find(lambda tag: tag.name=='table')
                rows = table.findAll(lambda tag: tag.name=='tr')
            _tags = ["th"] + ["td"] * (len(rows) - 1)
            data = [
                [
                    hh.string for hh in _row.findAll(
                        lambda tag: tag.name==_tags[num]
                    )
                ] for num, _row in enumerate(rows)
            ]
            _samples = self.samples[label]
            _key_data = _samples.key_data
            for row in data[1:]:
                for num, header in enumerate(data[0][1:]):
                    try:
                        np.testing.assert_almost_equal(
                            _key_data[row[0]][header], float(row[num + 1]), 3
                        )
                    except ValueError:
                        assert _key_data[row[0]][header] is None
                        assert row[num + 1] == 'None'

    def test_displayed_comparison_parameter_summary_table(self):
        """
        """
        with open("./.outdir/html/Comparison_chirp_mass.html") as fp:
            soup = BeautifulSoup(fp, features="html.parser")
            table = soup.find(lambda tag: tag.name=='table')
            rows = table.findAll(lambda tag: tag.name=='tr')
        _tags = ["th"] + ["td"] * (len(rows) - 1)
        data = [
            [
                hh.string for hh in _row.findAll(
                    lambda tag: tag.name==_tags[num]
                )
            ] for num, _row in enumerate(rows)
        ]
        _key_data = {
            label: self.samples[label].key_data["chirp_mass"] for label in
            self.labels
        }
        for row in data[1:]:
            for num, header in enumerate(data[0][1:]):
                try:
                    np.testing.assert_almost_equal(
                        _key_data[row[0]][header], float(row[num + 1]), 3
                    )
                except ValueError:
                    assert _key_data[row[0]][header] is None
                    assert row[num + 1] == 'None'
