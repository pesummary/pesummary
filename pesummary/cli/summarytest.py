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
import multiprocessing

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
ALLOWED = [
    "executables", "imports", "tests", "workflow", "skymap", "bilby",
    "bilby_pipe", "pycbc", "lalinference", "GWTC1", "GWTC2", "GWTC3", "examples"
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
        "-m", "--mark", dest="mark", default="", type=str,
        help="only run tests matching given mark expression"
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
    parser.add_argument(
        "--multi_process", dest="multi_process", default=1, help=(
            "Number of CPUs to use for the 'tests' and 'workflow' tests. "
            "Default 1"
        )
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


def tests(*args, output="./", multi_process=1, **kwargs):
    """Run the pesummary testing suite
    """
    import requests
    # download files for tests
    logger.info("Downloading files for tests")
    data = requests.get(
        "https://dcc.ligo.org/public/0168/P2000183/008/GW190814_posterior_samples.h5"
    )
    with open("{}/GW190814_posterior_samples.h5".format(output), "wb") as f:
        f.write(data.content)
    data = requests.get(
        "https://dcc.ligo.org/public/0163/P190412/012/GW190412_posterior_samples_v3.h5"
    )
    with open("{}/GW190412_posterior_samples.h5".format(output), "wb") as f:
        f.write(data.content)
    # launch pytest job
    command_line = (
        "{} -m pytest -n {} --max-worker-restart=2 --dist=loadfile --reruns 2 "
        "--pyargs pesummary.tests ".format(sys.executable, multi_process)
    )
    if kwargs.get("pytest_config", None) is not None:
        command_line += "-c {} ".format(kwargs.get("pytest_config"))
    if kwargs.get("coverage", False):
        command_line += (
            "--cov=pesummary --cov-report html:{}/htmlcov --cov-report "
            "term:skip-covered --cov-append ".format(output)
        )
    for ignore in kwargs.get("ignore", []):
        command_line += "--ignore {} ".format(ignore)
    if len(kwargs.get("mark", "")):
        command_line += "-m '{}' ".format(kwargs.get("mark"))
    if kwargs.get("expression", None) is not None:
        command_line += "-k {} ".format(kwargs.get("expression"))
    launch(command_line)
    if kwargs.get("coverage", False):
        command_line = "coverage-badge -o {} -f".format(
            os.path.join(output, "coverage_badge.svg")
        )
        launch(command_line)


@tmp_directory
def workflow(*args, multi_process=1, **kwargs):
    """Run the pesummary.tests.workflow_test test
    """
    command_line = (
        "{} -m pytest -n {} --max-worker-restart=2 --reruns 2 --pyargs "
        "pesummary.tests.workflow_test ".format(sys.executable, multi_process)
    )
    if kwargs.get("pytest_config", None) is not None:
        command_line += "-c {} ".format(kwargs.get("pytest_config"))
    if kwargs.get("expression", None) is not None:
        command_line += "-k '{}' ".format(kwargs.get("expression"))
    return launch(command_line)


def skymap(*args, output="./", **kwargs):
    """Run the pesummary.tests.ligo_skymap_test
    """
    command_line = "{} -m pytest --pyargs pesummary.tests.ligo_skymap_test ".format(
        sys.executable
    )
    if kwargs.get("coverage", False):
        command_line += (
            "--cov=pesummary --cov-report html:{}/htmlcov --cov-report "
            "term:skip-covered --cov-append ".format(output)
        )
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
def bilby_pipe(*args, **kwargs):
    """Test a bilby_pipe run
    """
    command_line = "bash {}".format(
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "bilby_pipe.sh")
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


def _public_pesummary_result_file(event, catalog=None, unpack=True, **kwargs):
    """Test that pesummary can load in a previously released pesummary result
    file
    """
    from pesummary.gw.fetch import fetch_open_samples

    download = fetch_open_samples(
        event, catalog=catalog, read_file=False, delete_on_exit=False,
        outdir="./", unpack=unpack
    )
    command_line = "{} {} -f {}.h5".format(
        sys.executable,
        os.path.join(PESUMMARY_DIR, "pesummary", "tests", "existing_file.py"),
        os.path.join(download, download) if unpack else str(download).split(".h5")[0]
    )
    return launch(command_line)


def _grab_event_names_from_gwosc(webpage):
    """Grab a list of event names from a GWOSC 'Event Portal' web page

    Parameters
    ----------
    webpage: str
        web page url that you wish to grab data from
    """
    from bs4 import BeautifulSoup
    import requests
    page = requests.get(webpage)
    soup = BeautifulSoup(page.content, 'html.parser')
    entries = soup.find_all("td")
    events = [
        e.text.strip().replace(" ", "") for e in entries if "GW" in e.text
        and "GWTC" not in e.text
    ]
    return events


@tmp_directory
def GWTCN(
    *args, catalog=None, size=5, include_exceptional=[], **kwargs
):
    """Test that pesummary can load a random selection of samples from the
    GWTC-2 or GWTC-3 data releases

    Parameters
    ----------
    catalog: str
        name of the gravitational wave catalog you wish to consider
    size: int, optional
        number of events to randomly draw. Default 5
    include_exceptional: list, optional
        List of exceptional event candidates to include in the random selection
        of events. This means that the total number of events could be as
        large as size + N where N is the length of include_exceptional. Default
        []
    """
    if catalog is None:
        raise ValueError("Please provide a valid catalog")
    events = _grab_event_names_from_gwosc(
        "https://www.gw-openscience.org/eventapi/html/{}/".format(catalog)
    )
    specified = np.random.choice(events, replace=False, size=size).tolist()
    if len(include_exceptional):
        for event in include_exceptional:
            if event not in specified:
                specified.append(event)
    for event in specified:
        _ = _public_pesummary_result_file(event, catalog=catalog, **kwargs)
    return


@tmp_directory
def GWTC2(*args, **kwargs):
    """Test that pesummary can load a random selection of samples from the
    GWTC-2 data release

    Parameters
    ----------
    size: int, optional
        number of events to randomly draw. Default 5
    include_exceptional: list, optional
        List of exceptional event candidates to include in the random selection
        of events. This means that the total number of events could be as
        large as size + N where N is the length of include_exceptional. Default
        []
    """
    return GWTCN(
        *args, catalog="GWTC-2", unpack=True,
        include_exceptional=["GW190412", "GW190425", "GW190521", "GW190814"],
        **kwargs
    )


@tmp_directory
def GWTC3(*args, **kwargs):
    """Test that pesummary can load a random selection of samples from the
    GWTC-3 data release

    Parameters
    ----------
    size: int, optional
        number of events to randomly draw. Default 5
    include_exceptional: list, optional
        List of exceptional event candidates to include in the random selection
        of events. This means that the total number of events could be as
        large as size + N where N is the length of include_exceptional. Default
        []
    """
    return GWTCN(*args, catalog="GWTC-3-confident", unpack=False, **kwargs)


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
    process = {}
    for script in shell_scripts:
        command_line = f"bash {script}"
        p = launch(command_line, check_call=False)
        process[command_line] = p
    python_scripts = glob.glob(os.path.join(gw_examples, "*.py"))
    python_scripts += [os.path.join(core_examples, "bounded_kdeplot.py")]
    for script in python_scripts:
        command_line = f"python {script}"
        p = launch(command_line, check_call=False)
        process[command_line] = p
    failed = []
    while len(process):
        _remove = []
        for key, item in process.items():
            if item.poll() is not None and item.returncode != 0:
                failed.append(key)
            elif item.poll() is not None:
                logger.info("The following test passed: {}".format(key))
                _remove.append(key)
        for key in _remove:
            process.pop(key)
    if len(failed):
        raise ValueError(
            "The following tests failed: {}".format(", ".join(failed))
        )
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
            coverage=opts.coverage, mark=opts.mark, expression=opts.expression,
            ignore=opts.ignore, pytest_config=opts.pytest_config,
            output=opts.output, repository=os.path.abspath(opts.repository),
            multi_process=opts.multi_process
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            "The {} test failed with error {}".format(opts.type, e)
        )


if __name__ == "__main__":
    main()
