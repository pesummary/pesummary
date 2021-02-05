# Licensed under an MIT style license -- see LICENSE.md

from ._version_helper import (
    get_version_information, GitInformation, PackageInformation, GitDummy
)
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

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
