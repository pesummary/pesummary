# Copyrigh (C) 2020 Charlie Hoy <charlie.hoy@ligo.org>
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
import glob
from datetime import date

python_major_version, python_minor_version = (3, 6)
pesummary_version = "0.9.1"
major, minor, build = pesummary_version.split(".")

option_1 = glob.glob("*")
option_2 = glob.glob(os.path.join("containers", "*"))

if "Dockerfile-template" in option_1:
    path = ".."
elif os.path.join("containers", "Dockerfile-template") in option_2:
    path = "."
else:
    raise FileNotFoundError("Unable to find a template docker file")

with open(os.path.join(path, "containers", "Dockerfile-template"), "r") as f:
    template = f.read()

with open(os.path.join(path, "requirements.txt"), "r") as f:
    _requirements = f.read().strip().split("\n")
    _requirements = [
        "'{}'".format(r) if '<=' in r else r for r in _requirements
    ]
    requirements = ' \\\n'.join(_requirements)

with open(os.path.join(path, "optional_requirements.txt"), "r") as f:
    optional = ' \\\n'.join(f.read().strip().split("\n"))

Docker_filename = "Dockerfile-pesummary-python{}{}".format(
    python_major_version, python_minor_version
)
with open(os.path.join(path, "containers", Docker_filename), "w") as f:
    content = "# This file was made automatically based on a template\n\n"
    content += template.format(
        date=date.today().strftime("%Y%m%d"),
        python_major_version=python_major_version,
        python_minor_version=python_minor_version,
        requirements=requirements, optional_requirements=optional,
        pesummary_major_version=major, pesummary_minor_version=minor,
        pesummary_build_version=build
    )
    f.write(content)
