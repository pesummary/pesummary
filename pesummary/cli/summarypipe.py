#! /usr/bin/env python

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

from pesummary.utils.utils import logger
from pesummary.utils.decorators import open_config
import argparse
import shutil
import os
import re
import ast
import glob


__doc__ = """This executable is used to generate a summarypages executable
given a rundir"""


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--rundir", dest="rundir",
                        help="run directory of the parameter estimation job",
                        default=None, required=True)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR",
                        default=None)
    parser.add_argument("--labels", dest="labels",
                        help="labels used to distinguish runs", nargs='+',
                        default=None)
    return parser


def is_lalinference_rundir(rundir):
    """Function to check if the rundir was made by LALInference

    Parameters
    ----------
    rundir: str
        the run directory of the parameter estimation job
    """
    std_dirs = ["ROQdata", "engine", "posterior_samples", "caches", "log"]
    directories = [i[0] for i in os.walk(os.path.join(rundir, ""))]
    if all(any(std in dd for dd in directories) for std in std_dirs):
        return True
    return False


class Base(object):
    """
    """
    __arguments__ = {
        "samples": "samples", "config": "config", "webdir": "webdir",
        "approximant": "approximant", "labels": "label", "{}_psd": "psd",
        "{}_calibration": "calibration", "gwdata": "gwdata"
    }

    def __init__(self, rundir, webdir=None, label=None):
        self.rundir = rundir
        self.samples = None
        self.config = None
        self.webdir = webdir
        self.label = label
        self.gracedb = None
        self.approximant = None
        self.psd = None
        self.calibration = None
        self.gwdata = None
        self.command_line_arguments = {
            arg.format(self.label): getattr(self, attribute, None)
            for arg, attribute in self.__arguments__.items()
        }
        self.command = self.make_command(self.command_line_arguments)
        self.print(self.command)

    @property
    def rundir(self):
        return self._rundir

    @rundir.setter
    def rundir(self, rundir):
        self._rundir = rundir
        if not os.path.isdir(rundir):
            raise ValueError(
                "'{}' is not a valid directory.".format(rundir)
            )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        files = self.glob(self.rundir, self.pattern["config_file"])
        self.check_files(
            files, "configuration file", self.pattern["config_file"],
            self.rundir
        )
        self._config = files[0]

    @property
    def webdir(self):
        return self._webdir

    @webdir.setter
    def webdir(self, webdir):
        if webdir is None:
            self._webdir = self.webdir_fallback()
        else:
            logger.info("Using the supplied webdir: {}".format(webdir))
            self._webdir = webdir

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        files = self.glob(self.rundir, self.pattern["posterior_file"])
        self.check_files(
            files, "result file", self.pattern["posterior_file"], self.rundir
        )
        self._samples = files[0]

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if label is None:
            logger.info("No label provided. Using default")
            self._label = self.default_label
        else:
            logger.info("Using the provided label: '{}'".format(label))
            self._label = label

    @property
    def gracedb(self):
        return self._gracedb

    @gracedb.setter
    def gracedb(self, gracedb):
        self._gracedb = self.gracedb_fallback()

    @property
    def approximant(self):
        return self._approximant

    @approximant.setter
    def approximant(self, approximant):
        self._approximant = self.approximant_fallback()

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, psd):
        self._psd = self.psd_fallback()

    @property
    def calibration(self):
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        self._calibration = self.calibration_fallback()

    @property
    def gwdata(self):
        return self._gwdata

    @gwdata.setter
    def gwdata(self, gwdata):
        self._gwdata = self.gwdata_fallback()

    @staticmethod
    def glob(directory, name):
        return glob.glob(os.path.join(directory, "**", name), recursive=True)

    @staticmethod
    def check_files(files, description, filename, rundir):
        if len(files) == 0:
            raise FileNotFoundError(
                "No file called '{}' found in {}.".format(filename, rundir)
            )
        elif len(files) == 1:
            logger.info(
                "Found the following {}: {}".format(description, files[0])
            )
        else:
            logger.info(
                "Multiple {}s found. Using {}".format(description, files[0])
            )

    @staticmethod
    def find_executable(executable):
        """Return the path to a given executable

        Parameters
        ----------
        executable: str
            name of executable you wish to get the path for
        """
        return shutil.which(executable)

    @staticmethod
    def make_command(dictionary):
        """Make the command line

        Parameters
        ----------
        dictionary: dict
            dictionary of commands and arguments
        """
        executable = Base.find_executable("summarypages")
        command = "summarypages "
        for key, val in dictionary.items():
            if val is True:
                command += "--{} ".format(key)
            elif val is None:
                continue
            else:
                command += "--{} {} ".format(key, val)
        return command

    @staticmethod
    def print(command):
        """Print the command line to std.out

        Parameters
        ----------
        command: str
            command you wish to print
        """
        logger.info("To run PESummary, run the following command line:")
        print("\n\t$ {}\n".format(command))

    @staticmethod
    @open_config(index=0)
    def load_config(path_to_config):
        """Load a configuration file with configparser

        Parameters
        ----------
        path_to_config: str
            path to the configuration file you wish to open
        """
        return path_to_config


class LALInference(Base):
    """Generate a `summarypages` executable for a lalinference run directory
    """
    def __init__(self, rundir, webdir=None, label=None):
        self.default_label = "LALInference"
        self.pattern = {
            "posterior_file": "posterior*.hdf5",
            "config_file": "config.ini"
        }
        super(LALInference, self).__init__(rundir, webdir=webdir, label=label)

    def webdir_fallback(self):
        """Grab the web directory from a LALInference configuration file
        """
        config = self.load_config(self.config)
        path = config["paths"]["webdir"]
        logger.info(
            "Using the same webdir as used in '{}': {}".format(
                self.config, path
            )
        )
        return config["paths"]["webdir"]

    def gracedb_fallback(self):
        """Grab the gracedb entry from a LALInference configuration file
        """
        config = self.load_config(self.config)
        try:
            gid = config["input"]["gid"]
            logger.info(
                "Found the following gracedb ID in '{}': '{}'".format(
                    self.config, gid
                )
            )
            return gid
        except KeyError:
            logger.info(
                "Unable to find a gracedb entry in '{}'".format(self.config)
            )
            return None

    def approximant_fallback(self):
        """Grab the approximant used from a LALInference configuration file
        """
        config = self.load_config(self.config)
        approx = config["engine"]["approx"]
        approx = approx.split("pseudo")[0].split("_ROQ")[0]
        logger.info(
            "Found the following approximant in '{}': '{}'".format(
                self.config, approx
            )
        )
        return approx

    def psd_fallback(self):
        """Try and grab data from a LALInference configuration file else
        look in the rundir for any PSD files
        """
        config = self.load_config(self.config)
        try:
            return self.grab_psd_calibration_data_from_config(config, "-psd")
        except KeyError:
            logger.info(
                "Unable to find any PSD information in '{}'. Looking in "
                "run directory".format(self.config)
            )
            files = self.glob(self.rundir, "*-PSD.dat")
            if len(files) == 0:
                logger.info(
                    "No PSD files found in '{}'".format(self.rundir)
                )
                return None
            ifos = {re.split("([A-Z][0-9]+)-PSD.dat", i)[-2] for i in files}
            return " ".join(
                [
                    [
                        "{}:{}".format(ifo, i) for i in files if
                        "{}-PSD.dat".format(ifo) in i
                    ][0] for ifo in ifos
                ]
            )

    def calibration_fallback(self):
        """Grab calibration data from a LALInference configuration file
        """
        config = self.load_config(self.config)
        try:
            return self.grab_psd_calibration_data_from_config(
                config, "-spcal-envelope"
            )
        except KeyError:
            logger.info(
                "Unable to find any calibration information in '{}'".format(
                    self.config
                )
            )
            return None

    def gwdata_fallback(self):
        """Grab the GW cache files from a LALInference run directory
        """
        try:
            cache_files = self.glob(self.rundir, "*.lcf")
            logger.info(
                "Found the following cache files in {}: {}".format(
                    self.rundir, ", ".join(cache_files)
                )
            )
            ifos = {re.split("([A-Z]-[A-Z][0-9]+)", i)[-2] for i in cache_files}
            ifos = [ifo.split("-")[-1] for ifo in ifos]
            config = self.load_config(self.config)
            channels = ast.literal_eval(config["data"]["channels"])
            return " ".join(
                [
                    "{}:{}".format(channels[ifo], path) for ifo, path
                    in zip(ifos, cache_files)
                ]
            )
        except ValueError:
            return None

    def grab_psd_calibration_data_from_config(self, config, pattern):
        """Grab psd/calibration data from a LALInference configuration file

        Parameters
        ----------
        config: configparser.ConfigParser
            open configuration file
        pattern: str
            string to identify the data stored in the configuration file
        """
        ifos = [i for i in list(config["engine"].keys()) if pattern in i]
        data = {
            ifo.split(pattern)[0].upper(): config["engine"][ifo] for ifo in
            ifos if len(ifos)
        }
        if not len(ifos):
            raise KeyError
        return " ".join(
            ["{}:{}".format(key, val) for key, val in data.items()]
        )


def main(args=None):
    """Top level interface for `summarypipe`
    """
    parser = command_line()
    opts = parser.parse_args(args=args)
    label = opts.labels[0] if opts.labels is not None else None
    if is_lalinference_rundir(opts.rundir):
        LALInference(opts.rundir, webdir=opts.webdir, label=label)
    else:
        raise NotImplementedError(
            "'{}' not understood. Currently 'summarypipe' only works with a "
            "LALInference rundir.".format(opts.rundir)
        )
