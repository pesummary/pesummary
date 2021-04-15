# Licensed under an MIT style license -- see LICENSE.md

import os
import glob
from datetime import date

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def format_requirements(path):
    """Format a file of requirements for a Docker file

    Parameters
    ----------
    path: str
        path to the file of requirements
    """
    with open(path, "r") as f:
        _requirements = f.read().strip().split("\n")
        _requirements = [
            "'{}'".format(r) if '<=' in r else r for r in _requirements
        ]
        requirements = ' \\\n'.join(_requirements)
    return requirements


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

requirements = format_requirements(os.path.join(path, "requirements.txt"))
optional = format_requirements(os.path.join(path, "optional_requirements.txt"))

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
