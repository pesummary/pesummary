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
import tarfile

try:
    import ciecplib
    CIECPLIB = True
except ImportError:
    CIECPLIB = False


def _unpack_and_extract(path_to_file, filename, path=None):
    """
    """
    path_to_file = Path(path_to_file)
    if not tarfile.is_tarfile(path_to_file):
        raise ValueError("unable to unpack file")
    outdir = path_to_file.parent
    tar = tarfile.open(path_to_file, 'r')
    _files = tar.getnames()
    if path is None:
        print("Extracting all files from {}".format(path_to_file))
        tar.extractall(path=outdir)
        return outdir / Path(filename).stem
    if not any(path in _file for _file in _files):
        raise ValueError(
            "Unable to find a file called '{}' in tarball. The list of "
            "available files are: {}".format(path, ", ".join(_files))
        )
    _path = [_file for _file in _files if path in _file][0]
    tar.extract(_path, path=outdir)
    unpacked_file = path_to_file.parent / _path
    if conf.delete_temporary_downloads_at_exit:
        global _tempfilestodel
        _tempfilestodel.append(unpacked_file)
    return unpacked_file


def _download_authenticated_file(
    url, unpack=False, path=None, block_size=2**16, **kwargs
):
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


def _download_file(url, unpack=False, path=None, **kwargs):
    """Downloads a URL and optionally caches the result

    Parameters
    ----------
    url: str
        url you wish to download
    unpack: Bool, optional
        if True, unpack tarball. Default False
    **kwargs: dict, optional
        additional kwargs passed to astropy.utils.data.download_file
    """
    return download_file(url, **kwargs)


def download_and_read_file(
    url, download_kwargs={}, read_file=True, delete_on_exit=True, outdir=None,
    unpack=False, path=None, _function=_download_file,
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
    if unpack:
        local = _unpack_and_extract(local, path=path, filename=filename)
        filename = Path(local).name
        if os.path.isdir(local):
            filename = Path(filename).stem
    if outdir is None:
        outdir = Path(local).parent
    if os.path.isdir(filename):
        new_name = Path(outdir)
    else:
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
