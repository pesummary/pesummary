# Licensed under an MIT style license -- see LICENSE.md

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

    def test_GW190412_public_notebook(self):
        import requests
        data = requests.get(
            "https://dcc.ligo.org/public/0163/P190412/012/GW190412_posterior_samples_v3.h5"
        )
        with open("GW190412_posterior_samples.h5", "wb") as f:
            f.write(data.content)
        make_public_notebook(
            "GW190412_posterior_samples.h5", "Title", default_analysis="combined",
            default_parameter="mass_ratio", outdir=".outdir",
            filename="posterior_samples.ipynb"
        )
        assert os.path.isfile(os.path.join(".outdir", "posterior_samples.ipynb"))
