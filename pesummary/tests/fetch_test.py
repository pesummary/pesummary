# Licensed under an MIT style license -- see LICENSE.md

from pesummary.io import read
from pesummary.core.fetch import download_and_read_file
from pesummary.gw.fetch import fetch_open_data
import numpy as np
import requests
import os

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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


def test_download_and_keep_file():
    """Test that when the `read=False` kwarg is passed to the
    download_and_read_file function the filename is returned
    """
    file_name = download_and_read_file(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5",
        outdir=".", read_file=False
    )
    assert os.path.isfile(file_name)


def test_fetch_tarball_and_keep():
    """Test that the `pesummary.gw.fetch.fetch_open_data` function is able to
    fetch, unpack and keep a tarball
    """
    directory_name = fetch_open_data(
        "GW190814", read_file=False, outdir=".", unpack=True
    )
    assert os.path.isdir("./GW190814")
    assert os.path.isdir(directory_name)


def test_fetch_tarball_and_keep_single_file():
    """Test that the `pesummary.gw.fetch.fetch_open_data` function is able to
    fetch, unpack and keep a single file stored in a tarball
    """
    file_name = fetch_open_data(
        "GW190814", read_file=False, outdir=".", unpack=True, path="GW190814.h5"
    )
    assert os.path.isfile("./GW190814.h5")
    assert os.path.isfile(file_name)


def test_fetch_and_open_tarball():
    """Test that a `pesummary.gw.fetch.fetch_open_data` function is able to
    fetch, unpack and read a single file stored in a tarball
    """
    import pesummary.gw.file.formats.pesummary

    f = fetch_open_data(
        "GW190814", read_file=True, outdir=".", unpack=True, path="GW190814.h5"
    )
    assert isinstance(f, pesummary.gw.file.formats.pesummary.PESummary)


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
