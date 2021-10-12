# Licensed under an MIT style license -- see LICENSE.md

import argparse
from datetime import date
from pathlib import Path

import pkg_resources

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

HERE = Path(__file__).parent


def read_requirements(file):
    """Yield requirements out of a (nested) pip-style requirements file

    Parameters
    ----------
    file : file object
        an open file to read

    Yields
    ------
    req : pkg_resources.Requirement
        a formatted `Requirement` object
    """
    for line in file:
        if line.startswith("-r "):
            name = line[3:].rstrip()
            with open(name, "r") as file2:
                yield from read_requirements(file2)
        else:
            yield from pkg_resources.parse_requirements(line)


def format_requirements(path):
    """Format a file of requirements for a Docker file

    Parameters
    ----------
    path: str
        path to the file of requirements
    """
    requirements = []
    with open(path, "r") as f:
        for req in read_requirements(f):
            requirements.append("{}{}".format(req.name, req.specifier))
    return " \\\n".join(map(repr, requirements))


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--python",
    default="3.8",
    help="X.Y version of Python to use",
)
parser.add_argument(
    "--pesummary",
    default="0.9.1",
    help="X.Y.Z version of pesummary to use",
)
args = parser.parse_args()

python_major_version, python_minor_version = args.python.split(".", 1)
major, minor, build = args.pesummary.split(".")

template = (HERE / "Dockerfile-template").read_text()

requirements = format_requirements(HERE.parent / "requirements.txt")
optional = format_requirements(HERE.parent / "optional_requirements.txt")

Docker_filename = "Dockerfile-pesummary-python{}{}".format(
    python_major_version, python_minor_version
)

content = "# This file was made automatically based on a template\n\n"
content += template.format(
    date=date.today().strftime("%Y%m%d"),
    python_major_version=python_major_version,
    python_minor_version=python_minor_version,
    requirements=requirements, optional_requirements=optional,
    pesummary_major_version=major, pesummary_minor_version=minor,
    pesummary_build_version=build
)
(HERE / Docker_filename).write_text(content)
