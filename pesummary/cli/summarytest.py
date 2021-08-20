# Licensed under an MIT style license -- see LICENSE.md

import subprocess
import os
import sys
import pesummary
from pesummary.utils.utils import logger
from pesummary.utils.decorators import tmp_directory
import numpy as np
import argparse
import glob
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
ALLOWED = [
    "executables", "imports", "tests", "workflow", "skymap", "bilby", "pycbc",
    "lalinference", "GWTC1", "GWTC2", "examples"
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


def launch(
    command, check_call=True, err=subprocess.DEVNULL, out=subprocess.DEVNULL
):
    """Launch a subprocess and run a command line

    Parameters
    ----------
    command: str
        command you wish to run
    """
    logger.info("Launching subprocess to run: '{}'".format(command))
    if check_call:
        return subprocess.check_call(command, shell=True)
    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err)
    return p


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


@tmp_directory
def pycbc(*args, **kwargs):
    """Test a pycbc run
    """
    command_line = "bash {}".format(
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "pycbc.sh")
    )
    return launch(command_line)


def _public_pesummary_result_file(event):
    """Test that pesummary can load in a previously released pesummary result
    file
    """
    from pesummary.gw.fetch import fetch_open_samples

    download = fetch_open_samples(
        event, read_file=False, delete_on_exit=False, outdir="./", unpack=True
    )
    command_line = "{} {} -f {}.h5".format(
        sys.executable,
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "existing_file.py"),
        os.path.join(download, download)
    )
    return launch(command_line)


@tmp_directory
def GWTC2(*args, size=5, include_exceptional=True, **kwargs):
    """Test that pesummary can load a random selection of samples from the
    GWTC-2 data release

    Parameters
    ----------
    size: int, optional
        number of events to randomly draw. Default 5
    include_exceptional: Bool, optional
        if True, add the exceptional event candidates to the random selection
        of events. This means that the total number of events could be as
        large as size + 4.
    """
    from bs4 import BeautifulSoup
    import requests
    page = requests.get("https://www.gw-openscience.org/eventapi/html/GWTC-2/")
    soup = BeautifulSoup(page.content, 'html.parser')
    entries = soup.find_all("td")
    events = [
        e.text.strip().replace(" ", "") for e in entries if "GW" in e.text
        and "GWTC" not in e.text
    ]
    specified = np.random.choice(events, replace=False, size=size).tolist()
    if include_exceptional:
        for event in ["GW190412", "GW190425", "GW190521", "GW190814"]:
            if event not in specified:
                specified.append(event)
    for event in specified:
        _ = _public_pesummary_result_file(event)
    return


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
