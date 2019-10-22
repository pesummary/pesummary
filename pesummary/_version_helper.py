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

import subprocess
from pathlib import Path


class GitInformation(object):
    """Helper class to handle the git information
    """
    def __init__(self):
        self.last_commit_info = self.get_last_commit_info()
        self.last_version = self.get_last_version()
        self.hash = self.last_commit_info[0]
        self.author = self.last_commit_info[1]
        self.status = self.get_status()
        self.builder = self.get_build_name()
        self.build_date = self.get_build_date()

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

    def get_package_info(self):
        """Return the package information
        """
        packages = self.call(["pip", "freeze"]).decode("utf-8")
        return packages


def get_version_information():
    """Grab the version from the .version file
    """
    version_file = Path(__file__).parent / ".version"

    string = ""
    try:
        with open(version_file, "r") as f:
            f = f.readlines()
            f = [i.strip() for i in f]

        version = [i.split("= ")[1] for i in f if "last_release" in i][0]
        hash = [i.split("= ")[1] for i in f if "git_hash" in i][0]
        status = [i.split("= ")[1] for i in f if "git_status" in i][0]
        string += "%s: %s %s" % (version, status, hash)
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
    return string
