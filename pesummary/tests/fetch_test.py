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

from pesummary.io import read
from pesummary.core.fetch import download_and_read_file
from pesummary.gw.fetch import fetch_open_data
import numpy as np
import requests


def test_download_and_read_file():
    """Test that the `pesummary.core.fetch.download_and_read_file` function
    works as expected
    """
    data = download_and_read_file(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"
    )
    _data = requests.get(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"
    )
    with open("GW150914_posterior_samples.h5", "wb") as f:
        f.write(_data.content)
    data2 = read("GW150914_posterior_samples.h5")
    np.testing.assert_almost_equal(
        np.array(data.samples), np.array(data2.samples)
    )


def test_fetch_open_data():
    """Test that the `pesummary.gw.fetch.fetch_open_data` function works as
    expected
    """
    data = fetch_open_data("GW150914")
    _data = requests.get(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"
    )
    with open("GW150914_posterior_samples.h5", "wb") as f:
        f.write(_data.content)
    data2 = read("GW150914_posterior_samples.h5")
    np.testing.assert_almost_equal(
        np.array(data.samples), np.array(data2.samples)
    )
