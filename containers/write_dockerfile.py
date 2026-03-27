# Licensed under an MIT style license -- see LICENSE.md

import argparse
import sys
from datetime import date
from pathlib import Path

__author__ = "Charlie Hoy <charlie.hoy@ligo.org>"

HERE = Path(__file__).parent
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        default=PYTHON_VERSION,
        help="X.Y version of Python to use",
    )
    parser.add_argument(
        "--pesummary",
        default="0.9.1",
        help="X.Y.Z version of pesummary to use",
    )
    return parser


def main(args=None):
    parser = create_parser()
    opts = parser.parse_args(args=args)

    python_major_version, python_minor_version = opts.python.split(".", 1)
    major, minor, build = opts.pesummary.split(".")

    template = (HERE / "Dockerfile-template").read_text()

    Docker_filename = "Dockerfile-pesummary-python{}.{}".format(
        python_major_version,
        python_minor_version
    )
    outfile = HERE / Docker_filename

    content = "# This file was made automatically based on a template\n\n"
    content += template.format(
        date=date.today().strftime("%Y%m%d"),
        python_major_version=python_major_version,
        python_minor_version=python_minor_version,
        pesummary_major_version=major,
        pesummary_minor_version=minor,
        pesummary_build_version=build
    )
    outfile.write_text(content)
    print("Wrote {}".format(outfile))


if __name__ == "__main__":
    main()
