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

import os
import json
import subprocess
import sys
from pathlib import Path


class GitInformation(object):
    """Helper class to handle the git information
    """
    def __init__(self, directory=None):
        if directory is None and not os.path.isdir(".git"):
            raise TypeError(
                "Not a git repository. Unable to get git information"
            )
        elif directory is None:
            directory = "."
        cwd = os.getcwd()
        os.chdir(directory)
        self.last_commit_info = self.get_last_commit_info()
        self.last_version = self.get_last_version()
        self.hash = self.last_commit_info[0]
        self.author = self.last_commit_info[1]
        self.status = self.get_status()
        self.builder = self.get_build_name()
        self.build_date = self.get_build_date()
        os.chdir(cwd)

    def call(self, arguments):
        """Launch a subprocess to run the bash command

        Parameters
        ----------
        arguments: list
            list of bash commands
        """
        return subprocess.check_output(arguments)

    def get_build_name(self):
        """Return the username and email of the current builder
        """
        try:
            name = self.call(["git", "config", "user.name"])
            email = self.call(["git", "config", "user.email"])
            name = name.strip()
            email = email.strip()
            return "%s <%s>" % (name.decode("utf-8"), email.decode("utf-8"))
        except Exception:
            return ""

    def get_build_date(self):
        """Return the current datetime
        """
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S +0000', time.gmtime())

    def get_last_commit_info(self):
        """Return the details of the last git commit
        """
        try:
            string = self.call(
                ["git", "log", "-1", "--pretty=format:%h,%an,%ae"])
            string = string.decode("utf-8").split(",")
            hash, username, email = string
            author = "%s <%s>" % (username, email)
            return hash, author
        except Exception:
            return ""

    def get_status(self):
        """Return the state of the git repository
        """
        git_diff = self.call(["git", "diff", "."]).decode("utf-8")
        if git_diff:
            return "UNCLEAN: Modified working tree"
        return "CLEAN: All modifications committed"

    def get_last_version(self):
        """Return the last stable version
        """
        try:
            tag_list = self.call(["git", "tag"]).decode("utf-8").split("\n")
            tag_list = [i for i in tag_list if i.startswith('v')]
            return tag_list[-1].split("v")[-1]
        except Exception:
            return "Not found"


class PackageInformation(GitInformation):
    """Helper class to handle package versions
    """
    def __init__(self):
        self.package_info = self.get_package_info()
        self.package_dir = self.get_package_dir()

    def get_package_info(self):
        """Return the package information
        """
        if (Path(sys.prefix) / "conda-meta").is_dir():
            self.package_manager = "conda"
            raw = self.call([
                "conda",
                "list",
                "--json",
                "--prefix", sys.prefix,
            ])
        else:
            self.package_manager = "pypi"
            raw = self.call([
                sys.executable,
                "-m", "pip",
                "list", "installed",
                "--format", "json",
            ])
        return json.loads(raw.decode('utf-8'))

    def get_package_dir(self):
        """Return the package directory
        """
        return sys.prefix


def make_version_file(
    version_file=None, return_string=False, version=None, add_install_path=False
):
    """Write a version file

    Parameters
    ----------
    version_file: str
        the path to the version file you wish to write
    return_sting: Bool, optional
        if True, return the version file as a string. Default False
    """
    git_info = GitInformation()
    packages = PackageInformation()

    if version is None:
        from ._version import get_versions

        version = get_versions()['version']

    string = (
        "# pesummary version information\n\n"
        "version = %s\nlast_release = %s\n\ngit_hash = %s\n"
        "git_author = %s\ngit_status = %s\ngit_builder = %s\n"
        "git_build_date = %s\n\n" % (
            version, git_info.last_version, git_info.hash,
            git_info.author, git_info.status, git_info.builder,
            git_info.build_date
        )
    )
    if add_install_path:
        string += install_path(return_string=True)
    if not return_string and version_file is None:
        raise ValueError("Please provide a version file")
    elif not return_string:
        with open(version_file, "w") as f:
            f.write(string)
        return version_file
    return string


def install_path(return_string=False):
    """Return the install path of a package
    """
    packages = PackageInformation()
    install_path = packages.package_dir
    string = "# Install information\n\ninstall_path = %s\n" % (
        install_path
    )
    if return_string:
        return string
    return packages.package_dir


def get_version_information(short=False):
    """Grab the version from the .version file

    Parameters
    ----------
    short: Bool
        If True, only return the version. If False, return git hash
    """
    from ._version import get_versions

    version = get_versions()['version']
    if short:
        version_file = Path(__file__).parent / ".version"
        try:
            with open(version_file, "r") as f:
                f = f.readlines()
                f = [i.strip() for i in f]

            version = [i.split("= ")[1] for i in f if "last_release" in i][0]
        except IndexError:
            print("No version information found")
        except FileNotFoundError as exc:
            # if we're inside setup.py, then the file not existing is ok
            try:
                if _PESUMMARY_SETUP:
                    return
            except NameError:
                pass
            raise
    return version
