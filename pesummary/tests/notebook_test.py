# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
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

from pesummary.gw.notebook import make_public_notebook
import os
import shutil


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
            "https://dcc.ligo.org/public/0163/P190412/008/posterior_samples.h5"
        )
        with open("GW190412_posterior_samples.h5", "wb") as f:
            f.write(data.content)
        make_public_notebook(
            "GW190412_posterior_samples.h5", "Title", default_analysis="combined",
            default_parameter="mass_ratio", outdir=".outdir",
            filename="posterior_samples.ipynb"
        )
        assert os.path.isfile(os.path.join(".outdir", "posterior_samples.ipynb"))
