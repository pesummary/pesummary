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

import os
from pathlib import Path
from astropy.utils.data import download_file
from pesummary.io import read


def _download_file(url, **kwargs):
    """Downloads a URL and optionally caches the result

    Parameters
    ----------
    url: str
        url you wish to download
    **kwargs: dict, optional
        additional kwargs passed to astropy.utils.data.download_file
    """
    return download_file(url, **kwargs)


def download_and_read_file(url, download_kwargs={}, **kwargs):
    """Downloads a URL and reads the file with pesummary.io.read function

    Parameters
    ----------
    url: str
        url you wish to download
    download_kwargs: dict, optional
        optional kwargs passed to _download_file
    **kwargs: dict, optional
        additional kwargs passed to pesummary.io.read function
    """
    local = _download_file(url, **download_kwargs)
    filename = Path(url).name
    new_name = Path(local).parent / filename
    os.rename(local, new_name)
    data = read(new_name, **kwargs)
    os.rename(new_name, local)
    return data
