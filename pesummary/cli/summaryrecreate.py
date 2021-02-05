#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.command_line import DelimiterSplitAction
from pesummary.gw.file.read import read
from pesummary.utils.utils import logger, make_dir
from pesummary.utils.dict import edit_dictionary, paths_to_key
from pesummary.utils.exceptions import InputError
from pesummary.io import write

import subprocess
import os
import argparse
import copy

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to recreate the analysis which was used
to generate the posterior samples stored in the metafile"""


def launch_subprocess(command_line):
    """Run a command line

    Parameters
    ----------
    command_line: str
        the command line you wish to run
    """
    logger.info("Running '{}'".format(command_line))
    process = subprocess.Popen(command_line, shell=True)
    process.wait()


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--samples", dest="samples", required=True,
        help=("Path to the PESummary file which stores the analysis you wish "
              "to recreate.")
    )
    parser.add_argument(
        "--labels", dest="labels", nargs='+', default=None,
        help="The label/labels of the analysis you wish to recreate"
    )
    parser.add_argument(
        "-r", "--rundir", dest="rundir", default="./",
        help="The run directory of the analysis"
    )
    parser.add_argument(
        "-c", "--code", dest="code", default="lalinference",
        help="The sampling software you wish to run"
    )
    parser.add_argument(
        "--delimiter", dest="delimiter", default=":",
        help="Delimiter used to seperate the existing and new quantity"
    )
    parser.add_argument(
        "--config_override", action=DelimiterSplitAction, nargs='+',
        dest="config_override", help=(
            "Changes you wish to make to the configuration file. Must be in the "
            "form `key:item` where key is the entry in the config file you wish "
            "to modify, ':' is the default delimiter, and item is the string you "
            "wish to replace with."
        ), default={}
    )
    return parser


class LALInference(object):
    """Class to create a LALInference analysis
    """
    @staticmethod
    def pipe(rundir, config, **kwargs):
        """Launch the `lalinference_pipe` executable

        Parameters
        ----------
        rundir: str
            path to the run directory of the analysis
        config: str
            path to the configuration file
        kwargs: dict
            dictionary of command line arguments that are passed directly to
            `lalinference_pipe`
        """
        command_line = "lalinference_pipe {} -r {}".format(
            config, os.path.join(rundir, "outdir")
        )
        for key, item in kwargs.items():
            command_line += " --{} {}".format(key, item)
        launch_subprocess(command_line)

    @staticmethod
    def map(parameter, delimiter=":"):
        """Map a parameter name to LALInference format

        Parameters
        ----------
        parameter: str
            The parameter you wish to map
        delimiter: str, optional
            The delimiter to split the parameter
        """
        not_equal_to_length_two_error = (
            "The {} must be in the form '{}:{}' where '%s' is the default "
            "delimiter" % (delimiter)
        )
        p = parameter.split(delimiter)
        if "psd" in p:
            if len(p) != 2:
                raise ValueError(
                    not_equal_to_length_two_error.format("psd", "ifo", "psd")
                )
            return "{}-{}".format(p[0].lower(), p[1])
        elif "calibration" in p:
            if len(p) != 2:
                raise ValueError(
                    not_equal_to_length_two_error.format(
                        "calibration", "ifo", "calibration"
                    )
                )
            return "{}-spcal-envelope".format(p[0].lower())
        else:
            return delimiter.join(p)


class Code(object):
    lalinference = LALInference


class _Input(object):
    """Super class to handle the command line arguments
    """
    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if not os.path.isfile(samples):
            raise FileNotFoundError(
                "File '{}' does not exist".format(samples)
            )
        self._samples = read(samples)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = self.samples.labels
        if labels is not None:
            self._labels = labels
            for label in labels:
                if label not in self.samples.labels:
                    raise InputError(
                        "The label '{}' is not stored in the result file. "
                        "The list of available labels is {}".format(
                            label, ", ".join(self.samples.labels)
                        )
                    )
            logger.info(
                "Recreating the following analyses: '{}'".format(
                    ", ".join(labels)
                )
            )
        else:
            logger.info(
                "No labels provided. Recreating all analyses stored in the "
                "result file: {}".format(", ".join(self.samples.labels))
            )

    @property
    def rundir(self):
        return self._rundir

    @rundir.setter
    def rundir(self, rundir):
        self._rundir = os.path.abspath(rundir)
        make_dir(rundir)
        logger.info(
            "Setting '{}' to be the run directory of the analysis".format(
                rundir
            )
        )
        for label in self.labels:
            make_dir(os.path.join(rundir, label))

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, code):
        allowed_codes = ["lalinference"]
        if code.lower() not in allowed_codes:
            raise InputError(
                "The 'summaryrecreate' executable is currently only configured "
                "to run with the following codes: {}".format(
                    ", ".join(allowed_codes)
                )
            )
        self._code = getattr(Code, code.lower())

    def _write_to_file(self, attribute):
        """Write an attribute stored in the result file to file

        Parameters
        ----------
        attribute: str
            name of the attribute that you wish to write to file
        """
        f = self.samples
        data = getattr(f, attribute)
        if len(data) and all(data[label] != {} for label in self.labels):
            for label in self.labels:
                for ifo in data[label].keys():
                    filename = os.path.join(
                        self.rundir, label, "{}_{}.dat".format(ifo, attribute)
                    )
                    data[label][ifo].save_to_file(filename)
                    name = self.code.map(
                        ":".join([ifo, attribute]), delimiter=":"
                    )
                    self.settings_to_change[label][name] = filename
        else:
            logger.warning(
                "No {} data found in the file. Using the {}s stored in the "
                "configuration file".format(attribute, attribute)
            )

    def write_psd_to_file(self):
        """Write the psd to file
        """
        self._write_to_file("psd")

    def write_calibration_to_file(self):
        """Write the calibration data to file
        """
        self._write_to_file("calibration")

    def write_config_file(self):
        """Save the config file to the run directory and return the data
        stored within
        """
        if not len(self.samples.config):
            raise ValueError(
                "No configuration file stored in the result file. Unable to "
                "recreate the analysis"
            )
        config_files = {}
        for label in self.labels:
            config_data = copy.deepcopy(self.samples.config[label])
            if self.settings_to_change is not None:
                for key, item in self.settings_to_change[label].items():
                    try:
                        path, = paths_to_key(key, config_data)
                        config_data = edit_dictionary(config_data, path, item)
                    except ValueError:
                        logger.warning(
                            "Unable to change '{}' to '{}' in the config "
                            "file".format(key, item)
                        )
            outdir = os.path.join(self.rundir, label)
            write(
                config_data, outdir=outdir, filename="config.ini",
                file_format="ini"
            )
            logger.info("Writing the configuration file to: {}".format(outdir))
            config_files[label] = os.path.join(outdir, "config.ini")
        return config_files


class Input(_Input):
    """Class to handle the command line arguments

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace object containing the command line options
    """
    def __init__(self, opts):
        logger.info(opts)
        self.opts = opts
        self.samples = self.opts.samples
        self.labels = self.opts.labels
        self.settings_to_change = {label: {} for label in self.labels}
        self.rundir = self.opts.rundir
        self.code = self.opts.code
        self.write_psd_to_file()
        self.write_calibration_to_file()
        for key, item in self.opts.config_override.items():
            for label in self.labels:
                name = self.code.map(key)
                self.settings_to_change[label][name] = item
        self.config = self.write_config_file()


def main(args=None):
    """The main function for the `summaryrecreate` executable
    """
    parser = command_line()
    opts = parser.parse_args(args=args)
    inputs = Input(opts)
    for label in inputs.labels:
        inputs.code.pipe(os.path.join(inputs.rundir, label), inputs.config[label])


if __name__ == "__main__":
    main()
