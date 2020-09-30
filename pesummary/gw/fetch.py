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

DCC = "https://dcc.ligo.org/public/"
GWTC1_base = DCC + "0157/P1800370/005/"
GWTC1_events = [
    "GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
    "GW170809", "GW170814", "GW170817", "GW170818", "GW170823"
]
DCC_MAP = {
    key: GWTC1_base + "{}_GWTC-1.hdf5".format(key) for key in GWTC1_events
}
DCC_MAP.update({
    "GW190412": DCC + "0163/P190412/012/GW190412_posterior_samples_v3.h5",
    "GW190814": DCC + "0168/P2000183/008/GW190814_posterior_samples.h5"
})


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


def fetch_open_data(event, **kwargs):
    """Download and read publically available gravitational wave posterior
    samples

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    """
    if event not in DCC_MAP.keys():
        raise ValueError(
            "Unknown URL for {}. If the URL is known, please run "
            "download_and_read_file(URL)"
        )
    return download_and_read_file(DCC_MAP[event], **kwargs)
