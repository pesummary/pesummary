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

from ._version_helper import (
    get_version_information, GitInformation, PackageInformation, GitDummy
)
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

version, last_stable_release, git_hash, git_author = [""] * 4
git_status, git_builder, git_build_date = [""] * 3

try:
    path = Path(__file__).parent
    with open(path / ".version", "r") as f:
        data = f.read()
    exec(data)
    __version__ = version
    __last_release__ = last_stable_release
    __short_version__ = __last_release__
    __git_hash__ = git_hash
    __git_author__ = git_author
    __git_status__ = git_status
    __git_builder__ = git_builder
    __git_build_date__ = git_build_date
except Exception:
    try:
        git_info = GitInformation()
    except TypeError:
        try:
            from pesummary import _version
            import os
            cfg = _version.get_config()
            ff = _version.__file__
            root = os.path.realpath(ff)
            for i in cfg.versionfile_source.split('/'):
                root = os.path.dirname(root)
            git_info = GitInformation(directory=root)
        except Exception:
            git_info = GitDummy()

    packages = PackageInformation()
    __version__ = get_version_information()
    __short_version__ = get_version_information(short=True)
    __last_release__ = git_info.last_version
    __git_hash__ = git_info.hash
    __git_author__ = git_info.author
    __git_status__ = git_info.status
    __git_builder__ = git_info.builder
    __git_build_date__ = git_info.build_date

__version_string__ = (
    "# pesummary version information\n\n"
    "version = '%s'\nlast_stable_release = '%s'\n\ngit_hash = '%s'\n"
    "git_author = '%s'\ngit_status = '%s'\ngit_builder = '%s'\n"
    "git_build_date = '%s'\n\n" % (
        __version__, __last_release__, __git_hash__, __git_author__,
        __git_status__, __git_builder__, __git_build_date__
    )
)

__bilby_compatibility__ = "0.3.6"
