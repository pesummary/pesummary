# Licensed under an MIT style license -- see LICENSE.md

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

CONDA_EXE = os.getenv("CONDA_EXE", shutil.which("conda")) or "conda"


class GitDummy(object):
    def __getattr__(self, attr):
        return ""


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
            tag = self.call(["git", "describe", "--tags", "--abbrev=0"]).decode(
                "utf-8"
            ).strip("\n")
            if tag[0] != "v":
                return tag
            return tag[1:]
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
                CONDA_EXE,
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
    from pesummary import __version_string__

    string = __version_string__
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
        return version.split("+")[0]
    return version
