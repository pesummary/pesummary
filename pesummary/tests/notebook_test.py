# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.fetch import fetch_open_samples
from pesummary.gw.notebook import make_public_notebook
import os
import shutil

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class TestPublicNoteBook(object):
    """Test the `make_public_notebook` function
    """
    def setup(self):
        """Setup the TestCoreDat class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_public_notebook(self):
        file_name = fetch_open_samples(
            "GW190424_180648", read_file=False, outdir=".", unpack=True,
            path="GW190424_180648.h5", catalog="GWTC-2"
        )
        make_public_notebook(
            "./GW190424_180648.h5", "Title", default_analysis="PublicationSamples",
            default_parameter="mass_ratio", outdir=".outdir",
            filename="posterior_samples.ipynb"
        )
        assert os.path.isfile(os.path.join(".outdir", "posterior_samples.ipynb"))
