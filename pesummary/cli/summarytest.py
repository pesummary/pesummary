# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org>
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
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA

import subprocess
import os
import sys
import pesummary
from pesummary.utils.utils import logger
from pesummary.utils.decorators import tmp_directory
import argparse
import glob
from pathlib import Path

ALLOWED = [
    "executables", "imports", "tests", "workflow", "skymap", "bilby",
    "lalinference", "GW190412", "GW190425", "GWTC1", "examples"
]

PESUMMARY_DIR = Path(pesummary.__file__).parent.parent


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--type", dest="type", required=True, type=str,
        help=(
            "The test you wish to run. Available tests are: {}".format(
                ", ".join(ALLOWED)
            )
        )
    )
    parser.add_argument(
        "-c", "--coverage", default=False, dest="coverage", action="store_true",
        help="Generare a coverage report for the testing suite"
    )
    parser.add_argument(
        "-i", "--ignore", nargs="+", dest="ignore", default=[],
        help="Testing scripts you wish to ignore"
    )
    parser.add_argument(
        "-k", "--expression", dest="expression", default=None, type=str,
        help=(
            "Run tests which contain names that match the given string "
            "expression"
        )
    )
    parser.add_argument(
        "--pytest_config", dest="pytest_config", default=None,
        help="Path to configuration file to use with pytest"
    )
    parser.add_argument(
        "-o", "--output", dest="output", default=".",
        help="Directory to store the output from the testing scripts"
    )
    parser.add_argument(
        "-r", "--repository", dest="repository",
        default=os.path.join(".", "pesummary"),
        help="Location of the pesummary repository"
    )
    return parser


def launch(command):
    """Launch a subprocess and run a command line

    Parameters
    ----------
    command: str
        command you wish to run
    """
    logger.info("Launching subprocess to run: '{}'".format(command))
    return subprocess.check_call(command, shell=True)


def executables(*args, **kwargs):
    """Test all pesummary executables
    """
    command_line = (
        "bash {}".format(
            os.path.join(PESUMMARY_DIR, "pesummary", "tests", "executables.sh")
        )
    )
    return launch(command_line)


def imports(*args, **kwargs):
    """Test all pesummary imports
    """
    command_line = (
        "bash {}".format(
            os.path.join(PESUMMARY_DIR, "pesummary", "tests", "imports.sh")
        )
    )
    return launch(command_line)


@tmp_directory
def tests(*args, output="./", **kwargs):
    """Run the pesummary testing suite
    """
    command_line = "pytest --pyargs pesummary.tests "
    if kwargs.get("pytest_config", None) is not None:
        command_line += "-c {} ".format(kwargs.get("pytest_config"))
    if kwargs.get("coverage", False):
        command_line += (
            "--cov=pesummary --cov-report html:{}/htmlcov --cov-report "
            "term:skip-covered ".format(output)
        )
    for ignore in kwargs.get("ignore", []):
        command_line += "--ignore {} ".format(ignore)
    if kwargs.get("expression", None) is not None:
        command_line += "-k {} ".format(kwargs.get("expression"))
    launch(command_line)
    if kwargs.get("coverage", False):
        command_line = "coverage-badge -o {} -f".format(
            os.path.join(output, "coverage_badge.svg")
        )
        launch(command_line)


@tmp_directory
def workflow(*args, **kwargs):
    """Run the pesummary.tests.workflow_test test
    """
    command_line = "pytest --pyargs pesummary.tests.workflow_test "
    if kwargs.get("pytest_config", None) is not None:
        command_line += "-c {} ".format(kwargs.get("pytest_config"))
    if kwargs.get("expression", None) is not None:
        command_line += "-k '{}' ".format(kwargs.get("expression"))
    return launch(command_line)


@tmp_directory
def skymap(*args, **kwargs):
    """Run the pesummary.tests.ligo_skymap_test
    """
    command_line = "pytest --pyargs pesummary.tests.ligo_skymap_test"
    return launch(command_line)


@tmp_directory
def lalinference(*args, **kwargs):
    """Test a lalinference run
    """
    command_line = "bash {}".format(
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "lalinference.sh")
    )
    return launch(command_line)


@tmp_directory
def bilby(*args, **kwargs):
    """Test a bilby run
    """
    command_line = "bash {}".format(
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "bilby.sh")
    )
    return launch(command_line)


def _old_pesummary_result_file(GW190412=False, GW190425=False):
    """Test that pesummary can load in a previously released pesummary result
    file
    """
    if GW190412:
        URL = "https://dcc.ligo.org/public/0163/P190412/012/GW190412_posterior_samples_v3.h5"
        NAME = "GW190412"
    elif GW190425:
        URL = "https://dcc.ligo.org/public/0165/P2000026/001/GW190425_posterior_samples.h5"
        NAME = "GW190425"
    else:
        raise ValueError("Unsupported existing file")
    command_line = "curl {} -o {}_posterior_samples.h5".format(URL, NAME)
    launch(command_line)
    command_line = "{} {} -f {}_posterior_samples.h5".format(
        sys.executable,
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "existing_file.py"),
        NAME
    )
    return launch(command_line)


@tmp_directory
def GW190412(*args, **kwargs):
    """Test that pesummary can load the GW190412 public data release file
    file
    """
    return _old_pesummary_result_file(GW190412=True)


@tmp_directory
def GW190425(*args, **kwargs):
    """Test that pesummary can load the GW190425 public data release file
    """
    return _old_pesummary_result_file(GW190425=True)


@tmp_directory
def GWTC1(*args, **kwargs):
    """Test that pesummary works on the GWTC1 data files
    """
    command_line = (
        "curl -O https://dcc.ligo.org/public/0157/P1800370/004/GWTC-1_sample_release.tar.gz"
    )
    launch(command_line)
    command_line = "tar -xf GWTC-1_sample_release.tar.gz"
    launch(command_line)
    command_line = "{} {} -f {} -t {}".format(
        sys.executable,
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "existing_file.py"),
        "GWTC-1_sample_release/GW150914_GWTC-1.hdf5",
        "pesummary.gw.file.formats.GWTC1.GWTC1"
    )
    launch(command_line)
    command_line = (
        "summarypages --webdir ./GWTC1 --no_ligo_skymap --samples "
        "GWTC-1_sample_release/GW150914_GWTC-1.hdf5 "
        "GWTC-1_sample_release/GW170817_GWTC-1.hdf5 --path_to_samples "
        "None IMRPhenomPv2NRT_highSpin_posterior --labels GW150914 GW170818 "
        "--gw"
    )
    return launch(command_line)


@tmp_directory
def examples(*args, repository=os.path.join(".", "pesummary"), **kwargs):
    """Test that the examples in the `pesummary` repository work
    """
    examples_dir = os.path.join(repository, "examples")
    gw_examples = os.path.join(examples_dir, "gw")
    core_examples = os.path.join(examples_dir, "core")
    shell_scripts = glob.glob(os.path.join(gw_examples, "*.sh"))
    for script in shell_scripts:
        command_line = f"bash {script}"
        launch(command_line)
    python_scripts = glob.glob(os.path.join(gw_examples, "*.py"))
    python_scripts += [os.path.join(core_examples, "bounded_kdeplot.py")]
    for script in python_scripts:
        command_line = f"python {script}"
        launch(command_line)
    return


def main(args=None):
    """Top level interface for `summarytest`
    """
    from pesummary.gw.parser import parser

    _parser = parser(existing_parser=command_line())
    opts, unknown = _parser.parse_known_args(args=args)
    if opts.type not in ALLOWED:
        raise NotImplementedError(
            "Invalid test type {}. Please choose one from the following: "
            "{}".format(opts.type, ", ".join(ALLOWED))
        )
    type_mapping = {_type: eval(_type) for _type in ALLOWED}
    try:
        type_mapping[opts.type](
            coverage=opts.coverage, expression=opts.expression,
            ignore=opts.ignore, pytest_config=opts.pytest_config,
            output=opts.output, repository=os.path.abspath(opts.repository)
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            "The {} test failed with error {}".format(opts.type, e)
        )


if __name__ == "__main__":
    main()
