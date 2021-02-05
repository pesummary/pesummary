#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger, list_match
from pesummary.utils.decorators import open_config
import argparse
import shutil
import os
import numpy as np
import sys
import re
import ast
import glob
from pathlib import Path

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
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
    parser.add_argument("--config", dest="config", default=None,
                        help=(
                            "config file to extract information from. This can either "
                            "be an absolute path to a config file, or a pattern to "
                            "search for in run directory. If not provided, search run "
                            "directory for a config file to use."
                        ))
    parser.add_argument("--samples", dest="samples", default=None,
                        help="path to posterior samples file you wish to use")
    parser.add_argument("--pattern", dest="pattern", default=None,
                        help="pattern to use when searching for files")
    parser.add_argument("--return_string", action="store_true", default=False,
                        help="Return command line as a string for testing")
    parser.add_argument("additional options", nargs="?", default=None, type=str,
                        help=(
                            "all additional command line options are added to the "
                            "printed summarypages executable"
                        ))
    return parser


def _check_rundir(rundir, std_dirs):
    """Check that a list of directories are stored in a given rundir

    Parameters
    ----------
    rundir: str
        base directory to start searching
    std_dirs: list
        list of dirs which you expect to find in rundir or a subdir of rundir
    """
    directories = [i[0] for i in os.walk(os.path.join(rundir, ""))]
    if all(any(std in dd for dd in directories) for std in std_dirs):
        return True
    return False


def is_lalinference_rundir(rundir):
    """Function to check if the rundir was made by LALInference

    Parameters
    ----------
    rundir: str
        the run directory of the parameter estimation job
    """
    std_dirs = ["ROQdata", "engine", "posterior_samples", "caches", "log"]
    return _check_rundir(rundir, std_dirs)


def is_bilby_rundir(rundir):
    """Function to check if the rundir was made by Bilby

    Parameters
    ----------
    rundir: str
        the run directory of the parameter estimation job
    """
    std_dirs = ["data", "result", "submit", "log_data_analysis"]
    return _check_rundir(rundir, std_dirs)


class Base(object):
    """
    """
    __arguments__ = {
        "samples": "samples", "config": "config", "webdir": "webdir",
        "approximant": "approximant", "labels": "label", "{}_psd": "psd",
        "{}_calibration": "calibration", "gwdata": "gwdata",
        "prior_file": "prior_file", "add_existing_plot": "add_existing_plot"
    }

    def __init__(
        self, rundir, webdir=None, label=None, samples=None, config=None, other="",
        pattern=None
    ):
        if pattern is not None:
            self.pattern = {
                "posterior_file": "{}*{}".format(pattern, self.posterior_extension),
                "config_file": "{}*.ini".format(pattern),
                "prior_file": "{}.prior".format(pattern)
            }
        if "ignore_posterior_file" not in self.pattern.keys():
            self.pattern["ignore_posterior_file"] = None
        self.rundir = os.path.abspath(rundir)
        self.parent_dir = Path(self.rundir).parent
        self.samples = samples
        self.config = config
        self.webdir = webdir
        self.label = label
        self.other = other
        self.gracedb = None
        self.approximant = None
        self.prior_file = None
        self.add_existing_plot = None
        self.psd = None
        self.calibration = None
        self.gwdata = None
        self.command_line_arguments = {
            arg.format(self.label): getattr(self, attribute, None)
            for arg, attribute in self.__arguments__.items()
        }
        self.command = self.make_command(
            self.command_line_arguments, other=self.other
        )
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
        if config is not None and os.path.isfile(config):
            logger.info("Using specified config file: '{}'".format(config))
            self._config = config
            return
        else:
            if config is not None:
                CUSTOM = True
                pattern = config
            else:
                CUSTOM = False
                pattern = self.pattern["config_file"]
            files, _dir = self.find_config_file(pattern)
            if CUSTOM and not len(files):
                raise FileNotFoundError(
                    "The supplied config file: '{}' does not exist".format(config)
                )
            self._config = self.check_files(
                files, "configuration file", pattern, _dir, allow_multiple=False,
                multiple_warn=(
                    "If you wish to use a specific config file, please add the "
                    "'--config' flag and the path to the config file you wish to "
                    "use."
                )
            )

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
        if samples is not None and os.path.isfile(samples):
            logger.info("Using specified posterior file: '{}'".format(samples))
            self._samples = samples
            return
        elif samples is not None:
            raise ValueError("Unable to find posterior file: {}".format(samples))
        files = self.glob(
            self.rundir, self.pattern["posterior_file"],
            ignore=self.pattern["ignore_posterior_file"]
        )
        files = self.preferred_samples_from_options(files)
        self._samples = self.check_files(
            files, "result file", self.pattern["posterior_file"], self.rundir,
            allow_multiple=False, multiple_warn=(
                "If you wish to use a specific samples file, please add the "
                "'--samples' flag and the path to the samples file you wish to "
                "use."
            )
        )

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if label is None:
            try:
                self._label = self.label_fallback()
            except AttributeError:
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
    def prior_file(self):
        return self._prior_file

    @prior_file.setter
    def prior_file(self, prior_file):
        self._prior_file = self.prior_file_fallback()

    @property
    def add_existing_plot(self):
        return self._add_existing_plot

    @add_existing_plot.setter
    def add_existing_plot(self, add_existing_plot):
        self._add_existing_plot = self.add_existing_plot_fallback()

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
    def glob(directory, name, ignore=None):
        """Recursively search a directory for a given file

        Parameters
        ----------
        directory: str
            directory to search in
        name: str
            file you wish to search for. This can include wildcards
        ignore: list, optional
            list of patterns to ignore
        """
        files = glob.glob(os.path.join(directory, "**", name), recursive=True)
        if ignore is not None:
            files = list_match(files, ignore, return_false=True, return_true=False)
        return files

    @staticmethod
    def check_files(
        files, description, filename, rundir, allow_failure=False,
        allow_multiple=True, multiple_warn=None,
    ):
        """Check a list of files found with `glob`

        Parameters
        ----------
        files: list
            list of files to check
        description: str
            description for the type of file you are searching for. This could
            be 'configuration file' for example
        filename: str
            pattern that was used to find the list of files with glob
        rundir: str
            run directory which was searched
        allow_failure: Bool, optional
            if True, bypass FileNotFoundError if the file does not exist
        allow_multiple: Bool, optional
            if True, do not raise a ValueError when multiple files are found
            which match a given pattern
        multiple_warn: str, optional
            warning message to print if multiple files are found
        """
        if len(files) == 0:
            if not allow_failure:
                raise FileNotFoundError(
                    "No file called '{}' found in {}.".format(filename, rundir)
                )
            return None
        elif len(files) == 1:
            logger.info(
                "Found the following {}: {}".format(description, files[0])
            )
            return files[0]
        else:
            if allow_multiple:
                logger.info(
                    "Multiple {}s found: {}. Using {}".format(
                        description, ", ".join(files), files[0]
                    )
                )
                if multiple_warn is not None:
                    logger.warn(multiple_warn)
                return files[0]
            raise ValueError(
                "Multiple {}s found in {}: {}. Please either specify one from the command "
                "line or add a pattern unique to that file name via the '--pattern' "
                "command line argument".format(
                    description, ", ".join(files), rundir
                )
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
    def make_command(dictionary, other=""):
        """Make the command line

        Parameters
        ----------
        dictionary: dict
            dictionary of commands and arguments
        """
        executable = Base.find_executable("summarypages")
        command = "summarypages "
        for key, val in dictionary.items():
            cla = "--{}".format(key)
            if cla in other:
                ind = other.index(cla)
                logger.warn(
                    "Ignoring {}={} extracted from config file and using input "
                    "from command line {}={}".format(
                        key, val, other[ind].replace("-", ""), other[ind + 1]
                    )
                )
            elif val is True:
                command += "{} ".format(cla)
            elif val is None:
                continue
            elif not len(val):
                continue
            else:
                command += "{} {} ".format(cla, val)
        if len(other):
            command += " ".join(other)
            command += " "
        if "--gw" not in command:
            command += "--gw "
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

    @property
    def preferred_posterior_string(self):
        return ""

    def find_config_file(self, pattern):
        """Find a configuration file given a pattern

        Parameters
        ----------
        pattern: str
            pattern to find a specific config file
        """
        return self.glob(self.rundir, pattern), self.rundir

    def preferred_samples_from_options(self, files):
        """Return the preferred posterior samples file from a list of
        options

        Parameters
        ----------
        files: list
            list of available options
        """
        _files = list_match(files, self.preferred_posterior_string)
        if len(_files):
            return _files
        return files

    def webdir_fallback(self):
        return "webpage"

    def gracedb_fallback(self):
        return

    def approximant_fallback(self):
        return

    def prior_file_fallback(self):
        return

    def add_existing_plot_fallback(self):
        return

    def psd_fallback(self):
        return

    def calibration_fallback(self):
        return

    def gwdata_fallback(self):
        return

    def _webdir_fallback(self, webdir):
        """Print the webdir extracted from the config file

        Parameters
        ----------
        webdir: str
            webdir extracted from the config file
        """
        logger.info(
            "Using the webdir '{}' extracted from the config file: {}".format(
                webdir, self.config
            )
        )
        return webdir

    def _label_fallback(self, label):
        """Print the label extracted from the config file

        Parameters
        ----------
        label: str
            label extracted from the config file
        """
        logger.info(
            "Found the following label in '{}': '{}'".format(self.config, label)
        )
        return label

    def _prior_file_fallback(self, prior_file):
        """Print the prior file found in the run directory

        Parameters
        ----------
        prior_file: str
            path to prior file found in the run directory
        """
        logger.info(
            "Found the following prior file': '{}'".format(prior_file)
        )
        return prior_file

    def _gracedb_fallback(self, gid):
        """Print the gracedb ID extracted from the config file

        Parameters
        ----------
        gid: str
            gracedb ID extracted from the config file
        """
        if gid is not None:
            logger.info(
                "Found the following gracedb ID in '{}': '{}'".format(
                    self.config, gid
                )
            )
            return gid
        else:
            logger.info(
                "Unable to find a gracedb entry in '{}'".format(self.config)
            )
            return None

    def _approximant_fallback(self, approx):
        """Print the approximant extracted from the config file

        Parameters
        ----------
        approx: str
            approximant extracted from the config file
        """
        approx = approx.split("pseudo")[0].split("_ROQ")[0]
        logger.info(
            "Found the following approximant in '{}': '{}'".format(
                self.config, approx
            )
        )
        return approx


class Bilby(Base):
    """Generate a `summarypages` executable for a bilby_pipe run directory
    """
    def __init__(self, *args, **kwargs):
        self.default_label = "Bilby"
        self.posterior_extension = "_result.json"
        self.pattern = {
            "posterior_file": "*_result.json",
            "ignore_posterior_file": ["*checkpoint*"],
            "config_file": "*.ini",
            "prior_file": "*.prior"
        }
        super(Bilby, self).__init__(*args, **kwargs)

    @property
    def preferred_posterior_string(self):
        return "*merged_result.json"

    def find_config_file(self, pattern):
        """Find a configuration file given a pattern

        Parameters
        ----------
        pattern: str
            pattern to find a specific config file
        """
        files = self.glob(self.rundir, pattern, ignore="*_complete.ini")
        if not len(files):
            return self.glob(
                self.parent_dir, pattern, ignore="*_complete.ini"
            ), self.parent_dir
        return files, self.rundir

    def try_underscore_and_hyphen(self, config, option):
        """Try to find a key in a dictionary. If KeyError is raised, replace
        underscore with a hyphen and try again

        Parameters
        ----------
        config: dict
            dictionary you wish to search
        option: str
            key you wish to search for
        """
        original = "_" if "_" in option else "-"
        alternative = "-" if original == "_" else "_"
        try:
            return config[option]
        except KeyError:
            return config[option.replace(original, alternative)]

    def webdir_fallback(self):
        """Grab the web directory from a Bilby configuration file
        """
        config = self.load_config(self.config)
        outdir = config["config"]["outdir"]
        if self.rundir in outdir:
            path = os.path.join(outdir, "webpage")
        else:
            path = os.path.join(self.rundir, outdir, "webpage")
        return self._webdir_fallback(path)

    def label_fallback(self):
        """Grab the label from a Bilby configuration file
        """
        config = self.load_config(self.config)
        label = config["config"]["label"]
        return self._label_fallback(label)

    def gracedb_fallback(self):
        """Grab the gracedb entry from a Bilby configuration file
        """
        config = self.load_config(self.config)
        try:
            gid = config["config"]["gracedb"]
            return self._gracedb_fallback(gid)
        except KeyError:
            return self._gracedb_fallback(None)

    def approximant_fallback(self):
        """Grab the approximant used from a Bilby configuration file
        """
        config = self.load_config(self.config)
        option = "waveform_approximant"
        approx = self.try_underscore_and_hyphen(config["config"], option)
        return self._approximant_fallback(approx)

    def prior_file_fallback(self):
        """Grab the prior file used from a Bilby run directory
        """
        prior_files = self.glob(self.rundir, self.pattern["prior_file"])
        if not len(prior_files):
            prior_files = self.glob(self.parent_dir, self.pattern["prior_file"])
        return self.check_files(
            prior_files, "prior_file", self.pattern["prior_file"], self.rundir,
            allow_failure=True
        )

    def add_existing_plot_fallback(self):
        """Grab the trace checkpoint plots generated from a Bilby run directory
        """
        existing_plots = self.glob(self.rundir, "*checkpoint*.png")
        return " ".join(
            ["{}:{}".format(self.label, _plot) for _plot in existing_plots]
        )

    def psd_fallback(self):
        """Try and grab PSD data from a Bilby configuration file else
        look in the rundir for any PSD files
        """
        config = self.load_config(self.config)
        try:
            return self.grab_psd_calibration_data_from_config(config, "psd_dict")
        except KeyError:
            from pesummary.gw.inputs import _GWInput
            logger.info(
                "Unable to find any PSD information in '{}'. Looking in "
                "run directory".format(self.config)
            )
            return self.grab_psd_calibration_data_from_directory("PSDs")

    def calibration_fallback(self):
        """Try and grab calibration data from a Bilby configuration file else
        look in the rundir for any calibration files
        """
        config = self.load_config(self.config)
        try:
            return self.grab_psd_calibration_data_from_config(
                config, "spline_calibration_envelope_dict"
            )
        except KeyError:
            from pesummary.gw.inputs import _GWInput
            logger.info(
                "Unable to find any calibration information in '{}'. Looking "
                "in run directory".format(self.config)
            )
            return self.grab_psd_calibration_data_from_directory(
                "cal_env", _type="calibration"
            )

    def grab_psd_calibration_data_from_config(self, config, pattern):
        """Grab psd/calibration data from a Bilby configuration file

        Parameters
        ----------
        config: configparser.ConfigParser
            open configuration file
        pattern: str
            string to identify the data stored in the configuration file
        """
        from pesummary.core.command_line import ConfigAction

        data_dict = self.try_underscore_and_hyphen(config["config"], pattern)
        if data_dict is None or data_dict == "None":
            return
        try:
            data_dict = ConfigAction.dict_from_str(data_dict, delimiter=":")
        except IndexError:
            data_dict = ConfigAction.dict_from_str(data_dict, delimiter="=")
        if not len(data_dict):
            raise KeyError
        for key, value in data_dict.items():
            if not os.path.isfile(value[0]):
                config_dir = Path(self.config).parent
                if os.path.isfile(os.path.join(config_dir, value[0])):
                    data_dict[key] = os.path.join(config_dir, value[0])
                else:
                    logger.warn(
                        "Found file: '{}' in the config file, but it  "
                        "does not exist. This is likely because the config "
                        "was run on a different cluster. Ignoring from final "
                        "command.".format(value[0])
                    )
                    raise KeyError
            else:
                data_dict[key] = value[0]
        return " ".join(
            ["{}:{}".format(key, val) for key, val in data_dict.items()]
        )

    def grab_psd_calibration_data_from_directory(self, pattern, _type="PSD"):
        """Grab psd/calibration data from a given run directory

        Parameters
        ----------
        pattern: str
            pattern to find a specific set of files
        """
        from pesummary.gw.inputs import _GWInput

        files = self.glob(os.path.join(self.rundir, pattern), "*")
        if not len(files):
            files = self.glob(os.path.join(self.parent_dir, pattern), "*")
            if not len(files):
                logger.info(
                    "No {} files found in '{}'".format(_type, self.rundir)
                )
                return None
        data_dict = {
            _GWInput.get_ifo_from_file_name(ff): ff for ff in files
        }
        return " ".join(
            ["{}:{}".format(key, val) for key, val in data_dict.items()]
        )


class LALInference(Base):
    """Generate a `summarypages` executable for a lalinference run directory
    """
    def __init__(self, *args, **kwargs):
        self.default_label = "LALInference"
        self.posterior_extension = "hdf5"
        self.pattern = {
            "posterior_file": "posterior*.hdf5",
            "config_file": "config.ini"
        }
        super(LALInference, self).__init__(*args, **kwargs)

    def webdir_fallback(self):
        """Grab the web directory from a LALInference configuration file
        """
        config = self.load_config(self.config)
        path = config["paths"]["webdir"]
        return self._webdir_fallback(path)

    def gracedb_fallback(self):
        """Grab the gracedb entry from a LALInference configuration file
        """
        config = self.load_config(self.config)
        try:
            gid = config["input"]["gid"]
            return self._gracedb_fallback(gid)
        except KeyError:
            return self._gracedb_fallback(None)

    def approximant_fallback(self):
        """Grab the approximant used from a LALInference configuration file
        """
        config = self.load_config(self.config)
        approx = config["engine"]["approx"]
        return self._approximant_fallback(approx)

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
    opts, unknown = parser.parse_known_args(args=args)
    if args is None:
        all_options = sys.argv[1:]
    else:
        all_options = args
    idxs = [all_options.index(_) for _ in unknown if "--" in _]
    unknown = np.array([
        [all_options[i], all_options[i + 1]] if i + 1 < len(all_options)
        and "--" not in all_options[i + 1] else [all_options[i], ""] for i in
        idxs
    ]).flatten()
    label = opts.labels[0] if opts.labels is not None else None
    if is_lalinference_rundir(opts.rundir):
        cl = LALInference(
            opts.rundir, webdir=opts.webdir, label=label, samples=opts.samples,
            config=opts.config, other=unknown, pattern=opts.pattern
        )
    elif is_bilby_rundir(opts.rundir):
        cl = Bilby(
            opts.rundir, webdir=opts.webdir, label=label, samples=opts.samples,
            config=opts.config, other=unknown, pattern=opts.pattern
        )
    else:
        raise NotImplementedError(
            "'{}' not understood. Currently 'summarypipe' only works with a "
            "LALInference or Bilby rundir.".format(opts.rundir)
        )
    if opts.return_string:
        return cl.command
