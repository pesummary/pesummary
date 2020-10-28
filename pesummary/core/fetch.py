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
import sys
import shutil
from pathlib import Path
from astropy.utils.console import ProgressBarOrSpinner
from astropy.utils.data import download_file, conf, _tempfilestodel
from pesummary.io import read
from tempfile import NamedTemporaryFile

try:
    import ciecplib
    CIECPLIB = True
except ImportError:
    CIECPLIB = False


def _download_authenticated_file(url, block_size=2**16, **kwargs):
    """Downloads a URL from an authenticated site

    Parameters
    ----------
    url: str
        url you wish to download
    **kwargs: dict, optional
        additional kwargs passed to ciecplib.Session
    """
    if not CIECPLIB:
        raise ImportError(
            "Please install 'ciecplib' in order to download authenticated urls"
        )

    with ciecplib.Session(**kwargs) as sess:
        pid = os.getpid()
        prefix = "pesummary-download-%s-" % (pid)
        response = sess.get(url, stream=True)
        size = int(response.headers.get('content-length', 0))
        dlmsg = "Downloading {}".format(url)
        bytes_read = 0
        with ProgressBarOrSpinner(size, dlmsg, file=sys.stdout) as p:
            with NamedTemporaryFile(prefix=prefix, delete=False) as f:
                for data in response.iter_content(block_size):
                    bytes_read += len(data)
                    p.update(bytes_read)
                    f.write(data)

    if conf.delete_temporary_downloads_at_exit:
        global _tempfilestodel
        _tempfilestodel.append(f.name)
    return f.name


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


def download_and_read_file(
    url, download_kwargs={}, read_file=True, delete_on_exit=True, outdir=None,
    _function=_download_file,
    **kwargs
):
    """Downloads a URL and reads the file with pesummary.io.read function

    Parameters
    ----------
    url: str
        url you wish to download
    download_kwargs: dict, optional
        optional kwargs passed to _download_file
    read_file: Bool, optional
        if True, read the downloaded file and return the opened object.
        if False, return the path to the downloaded file. Default True
    delete_on_exit: Bool, optional
        if True, delete the file on exit. Default True
    outdir: str, optional
        save the file to outdir. Default the default directory from
        tmpfile.NamedTemporaryFile
    **kwargs: dict, optional
        additional kwargs passed to pesummary.io.read function
    """
    conf.delete_temporary_downloads_at_exit = delete_on_exit
    local = _function(url, **download_kwargs)
    filename = Path(url).name
    if outdir is None:
        outdir = Path(local).parent
    new_name = Path(outdir) / filename
    shutil.move(local, new_name)
    if not read_file:
        if conf.delete_temporary_downloads_at_exit:
            global _tempfilestodel
            _tempfilestodel.append(new_name)
        return new_name
    data = read(new_name, **kwargs)
    shutil.move(new_name, local)
    return data
