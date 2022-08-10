# Licensed under an MIT style license -- see LICENSE.md

from pesummary.io import read
from pesummary.core.fetch import download_and_read_file
from pesummary.gw.fetch import fetch_open_samples
import numpy as np
import requests
import os

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def test_download_and_read_file():
    """Test that the `pesummary.core.fetch.download_and_read_file` function
    works as expected
    """
    data = download_and_read_file(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW170608_GWTC-1.hdf5"
    )
    _data = requests.get(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW170608_GWTC-1.hdf5"
    )
    with open("GW170608_posterior_samples.h5", "wb") as f:
        f.write(_data.content)
    data2 = read("GW170608_posterior_samples.h5")
    np.testing.assert_almost_equal(
        np.array(data.samples), np.array(data2.samples)
    )


def test_download_and_keep_file():
    """Test that when the `read=False` kwarg is passed to the
    download_and_read_file function the filename is returned
    """
    file_name = download_and_read_file(
        "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5",
        outdir=".", read_file=False
    )
    assert os.path.isfile(file_name)


def test_fetch_tarball_and_keep():
    """Test that the `pesummary.gw.fetch.fetch_open_samples` function is able to
    fetch, unpack and keep a tarball
    """
    directory_name = fetch_open_samples(
        "GW190424_180648", read_file=False, outdir=".", unpack=True,
        catalog="GWTC-2", download_kwargs={"timeout": 60}
    )
    assert os.path.isdir("./GW190424_180648")
    assert os.path.isdir(directory_name)


def test_fetch_tarball_and_keep_single_file():
    """Test that the `pesummary.gw.fetch.fetch_open_samples` function is able to
    fetch, unpack and keep a single file stored in a tarball
    """
    file_name = fetch_open_samples(
        "GW190424_180648", read_file=False, outdir=".", unpack=True,
        path="GW190424_180648.h5", catalog="GWTC-2",
        download_kwargs={"timeout": 60}
    )
    assert os.path.isfile("./GW190424_180648.h5")
    assert os.path.isfile(file_name)


def test_fetch_and_open_tarball():
    """Test that a `pesummary.gw.fetch.fetch_open_samples` function is able to
    fetch, unpack and read a single file stored in a tarball
    """
    import pesummary.gw.file.formats.pesummary

    f = fetch_open_samples(
        "GW190424_180648", read_file=True, outdir=".", unpack=True,
        path="GW190424_180648.h5", catalog="GWTC-2",
        download_kwargs={"timeout": 60}
    )
    assert isinstance(f, pesummary.gw.file.formats.pesummary.PESummary)


def test_fetch_open_samples():
    """Test that the `pesummary.gw.fetch.fetch_open_samples` function works as
    expected
    """
    data = fetch_open_samples("GW150914", download_kwargs={"timeout": 60})
    _data = requests.get(
        "https://zenodo.org/api/files/ecf41927-9275-47da-8b37-e299693fe5cb/" +
        "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
    )
    with open("GW150914_posterior_samples.h5", "wb") as f:
        f.write(_data.content)
    data2 = read("GW150914_posterior_samples.h5")
    for num in range(len(data.labels)):
        np.testing.assert_almost_equal(
            np.array(data.samples[num]), np.array(data2.samples[num])
        )


def test_fetch_open_strain():
    """Test that the `pesummary.gw.fetch.fetch_open_strain` function works as
    expected
    """
    from pesummary.gw.fetch import fetch_open_strain
    from gwpy.timeseries import TimeSeries
    data = fetch_open_strain(
        "GW190412", IFO="H1", channel="H1:GWOSC-4KHZ_R1_STRAIN",
        sampling_rate=4096
    )
    _data = requests.get(
        "https://www.gw-openscience.org/eventapi/html/GWTC-2/GW190412/v3/"
        "H-H1_GWOSC_4KHZ_R1-1239082247-32.gwf"
    )
    with open("H-H1_GWOSC_4KHZ_R1-1239082247-32.gwf", "wb") as f:
        f.write(_data.content)
    data2 = TimeSeries.read(
        "H-H1_GWOSC_4KHZ_R1-1239082247-32.gwf",
        channel="H1:GWOSC-4KHZ_R1_STRAIN"
    )
    np.testing.assert_almost_equal(data.value, data2.value)
    np.testing.assert_almost_equal(data.times.value, data2.times.value)
