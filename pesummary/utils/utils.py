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

import os
import sys
import logging
import contextlib

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import h5py


def resample_posterior_distribution(posterior, nsamples):
    """Randomly draw nsamples from the posterior distribution

    Parameters
    ----------
    posterior: ndlist
        nd list of posterior samples. If you only want to resample one
        posterior distribution then posterior=[[1., 2., 3., 4.]]. For multiple
        posterior distributions then posterior=[[1., 2., 3., 4.], [1., 2., 3.]]
    nsamples: int
        number of samples that you wish to randomly draw from the distribution
    """
    if len(posterior) == 1:
        n, bins = np.histogram(posterior, bins=50)
        n = np.array([0] + [i for i in n])
        cdf = cumtrapz(n, bins, initial=0)
        cdf /= cdf[-1]
        icdf = interp1d(cdf, bins)
        samples = icdf(np.random.rand(nsamples))
    else:
        posterior = np.array([i for i in posterior])
        keep_idxs = np.random.choice(
            len(posterior[0]), nsamples, replace=False)
        samples = [i[keep_idxs] for i in posterior]
    return samples


def check_condition(condition, error_message):
    """Raise an exception if the condition is not satisfied
    """
    if condition:
        raise Exception(error_message)


def rename_group_or_dataset_in_hf5_file(base_file, group=None, dataset=None):
    """Rename a group or dataset in an hdf5 file

    Parameters
    ----------
    group: list, optional
        a list containing the path to the group that you would like to change
        as the first argument and the new name of the group as the second
        argument
    dataset: list, optional
        a list containing the name of the dataset that you would like to change
        as the first argument and the new name of the dataset as the second
        argument
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The file %s does not exist" % (base_file))
    f = h5py.File(base_file, "a")
    if group:
        f[group[1]] = f[group[0]]
        del f[group[0]]
    elif dataset:
        f[dataset[1]] = f[dataset[0]]
        del f[dataset[0]]
    f.close()


def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


def guess_url(web_dir, host, user):
    """Guess the base url from the host name

    Parameters
    ----------
    web_dir: str
        path to the web directory where you want the data to be saved
    host: str
        the host name of the machine where the python interpreter is currently
        executing
    user: str
        the user that is current executing the python interpreter
    """
    ligo_data_grid = False
    if 'public_html' in web_dir:
        ligo_data_grid = True
    if ligo_data_grid:
        path = web_dir.split("public_html")[1]
        if "raven" in host or "arcca" in host:
            url = "https://geo2.arcca.cf.ac.uk/~{}".format(user)
        elif 'ligo-wa' in host:
            url = "https://ldas-jobs.ligo-wa.caltech.edu/~{}".format(user)
        elif 'ligo-la' in host:
            url = "https://ldas-jobs.ligo-la.caltech.edu/~{}".format(user)
        elif "cit" in host or "caltech" in host:
            url = "https://ldas-jobs.ligo.caltech.edu/~{}".format(user)
        elif 'uwm' in host or 'nemo' in host:
            url = "https://ldas-jobs.phys.uwm.edu/~{}".format(user)
        elif 'phy.syr.edu' in host:
            url = "https://sugar-jobs.phy.syr.edu/~{}".format(user)
        elif 'vulcan' in host:
            url = "https://galahad.aei.mpg.de/~{}".format(user)
        elif 'atlas' in host:
            url = "https://atlas1.atlas.aei.uni-hannover.de/~{}".format(user)
        elif 'iucca' in host:
            url = "https://ldas-jobs.gw.iucaa.in/~{}".format(user)
        else:
            url = "https://{}/~{}".format(host, user)
        url += path
    else:
        url = "https://{}".format(web_dir)
    return url


def command_line_arguments():
    """Return the command line arguments
    """
    return sys.argv[1:]


def gw_results_file(opts):
    """Determine if a GW results file is passed
    """
    cond1 = hasattr(opts, "gw") and opts.gw
    cond2 = hasattr(opts, "calibration") and opts.calibration
    cond3 = hasattr(opts, "gracedb") and opts.gracedb
    cond4 = hasattr(opts, "approximant") and opts.approximant
    cond5 = hasattr(opts, "psd") and opts.psd
    if cond1 or cond2 or cond3 or cond4 or cond5:
        return True
    else:
        return False


def functions(opts):
    """Return a dictionary of functions that are either specific to GW results
    files or core.
    """
    from cli.summarypages import WebpageGeneration, GWWebpageGeneration
    from cli.summaryplots import PlotGeneration, GWPlotGeneration
    from pesummary.core.inputs import Input
    from pesummary.gw.inputs import GWInput
    from pesummary.core.file.meta_file import MetaFile
    from pesummary.gw.file.meta_file import GWMetaFile
    from pesummary.core.finish import FinishingTouches

    print(gw_results_file)
    dictionary = {}
    dictionary["input"] = GWInput if gw_results_file(opts) else Input
    dictionary["PlotGeneration"] = GWPlotGeneration if gw_results_file(opts) else PlotGeneration
    dictionary["WebpageGeneration"] = GWWebpageGeneration if gw_results_file(opts) else WebpageGeneration
    dictionary["MetaFile"] = GWMetaFile if gw_results_file(opts) else MetaFile
    dictionary["FinishingTouches"] = FinishingTouches
    return dictionary


def get_version_information():
    """Grab the version from the .version file
    """
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".version")

    string = ""
    try:
        with open(version_file, "r") as f:
            f = f.readlines()
            f = [i.strip() for i in f]

        version = [i.split("= ")[1] for i in f if "last_release" in i][0]
        hash = [i.split("= ")[1] for i in f if "git_hash" in i][0]
        status = [i.split("= ")[1] for i in f if "git_status" in i][0]
        string += "%s: %s %s" % (version, status, hash)
    except Exception:
        print("No version information found")
    return string


def setup_logger():
    """Set up the logger output.
    """
    import tempfile

    if not os.path.isdir(".tmp/pesummary"):
        os.makedirs(".tmp/pesummary")
    dirpath = tempfile.mkdtemp(dir=".tmp/pesummary")
    level = 'INFO'
    if "-v" in sys.argv or "--verbose" in sys.argv:
        level = 'DEBUG'

    FORMAT = '%(asctime)s %(name)s %(levelname)-8s: %(message)s'
    logger = logging.getLogger('PESummary')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('%s/pesummary.log' % (dirpath), mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel('INFO')
    formatter = logging.Formatter(FORMAT, datefmt='%Y-%m-%d  %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


def remove_tmp_directories():
    """Remove the temporary directories created by PESummary
    """
    import shutil
    from glob import glob
    import time

    directories = glob(".tmp/pesummary/*")

    for i in directories:
        if os.path.isdir(i):
            shutil.rmtree(i)
        elif os.path.isfile(i):
            os.remove(i)


def customwarn(message, category, filename, lineno, file=None, line=None):
    import sys
    import warnings

    sys.stdout.write(warnings.formatwarning("%s" % (message), category, filename, lineno))


class RedirectLogger(object):
    """Class to redirect the output from other codes to the `pesummary`
    logger

    Parameters
    ----------
    level: str, optional
        the level to display the messages
    """
    def __init__(self, code, level="Debug"):
        self.logger = logging.getLogger('PESummary')
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)
        self.code = code

    def isatty(self):
        pass

    def write(self, msg):
        """Write the message to stdout

        Parameters
        ----------
        msg: str
            the message you wish to be printed to stdout
        """
        if msg and not msg.isspace():
            self.logger.log(self.level, "[from %s] %s" % (self.code, msg))

    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redirector.__exit__(exc_type, exc_value, traceback)


setup_logger()
logger = logging.getLogger('PESummary')
