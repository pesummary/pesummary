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

from pesummary.core.fetch import (
    download_and_read_file, _download_authenticated_file
)
from gwosc.api import fetch_event_json


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
