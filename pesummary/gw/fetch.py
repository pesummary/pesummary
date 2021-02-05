# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.fetch import (
    download_and_read_file, _download_authenticated_file
)
from gwosc.api import fetch_event_json

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def fetch(url, download_kwargs={}, **kwargs):
    """Download and read files from LIGO authenticated URLs

    Parameters
    ----------
    url: str
        url you wish to download
    download_kwargs: dict, optional
        optional kwargs passed to _download_autheticated_file
    **kwargs: dict, optional
        additional kwargs passed to pesummary.io.read function
    """
    if "idp" not in download_kwargs.keys():
        download_kwargs["idp"] = "LIGO"
    return download_and_read_file(
        url, download_kwargs=download_kwargs,
        _function=_download_authenticated_file, **kwargs
    )


def _DCC_url(event):
    """Return the url for posterior samples stored on the DCC for a given event

    Parameters
    ----------
    event: str
        name of the event you wish to return posterior samples for
    """
    data, = fetch_event_json(event)["events"].values()
    url = None
    for key, item in data["parameters"].items():
        if "_pe_" in key:
            url = item["data_url"]
            break
    if url is None:
        raise RuntimeError("Failed to find PE data URL for {}".format(event))
    return url


def fetch_open_data(event, **kwargs):
    """Download and read publically available gravitational wave posterior
    samples

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    """
    try:
        url = _DCC_url(event)
    except RuntimeError:
        raise ValueError(
            "Unknown URL for {}. If the URL is known, please run "
            "download_and_read_file(URL)"
        )
    return download_and_read_file(url, **kwargs)
