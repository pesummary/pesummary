# Licensed under an MIT style license -- see LICENSE.md

import argparse
from datetime import date
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

HERE = Path(__file__).parent
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

Docker_filename = "Dockerfile-pesummary-python{}{}".format(
    python_major_version, python_minor_version
)

content = "# This file was made automatically based on a template\n\n"
content += template.format(
    date=date.today().strftime("%Y%m%d"),
    python_major_version=python_major_version,
    python_minor_version=python_minor_version,
    pesummary_major_version=major, pesummary_minor_version=minor,
    pesummary_build_version=build
)
(HERE / Docker_filename).write_text(content)
