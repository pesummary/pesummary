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

import h5py


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


def gw_results_file():
    """
    """
    command_line = command_line_arguments()
    if "--calibration" in command_line or "--gw" in command_line or \
            "--approximant" in command_line or "--gracedb" in command_line or \
            "--psds" in command_line or "--detectors" in command_line:
        return True
    else:
        return False


def functions():
    """
    """
    from cli.summarypages import WebpageGeneration, GWWebpageGeneration
    from cli.summaryplots import PlotGeneration, GWPlotGeneration
    from pesummary.core.inputs import Input
    from pesummary.gw.inputs import GWInput
    from pesummary.core.file.meta_file import MetaFile
    from pesummary.gw.file.meta_file import GWMetaFile
    from pesummary.core.finish import FinishingTouches

    dictionary = {}
    dictionary["input"] = GWInput if gw_results_file() else Input
    dictionary["PlotGeneration"] = GWPlotGeneration if gw_results_file() else PlotGeneration
    dictionary["WebpageGeneration"] = GWWebpageGeneration if gw_results_file() else WebpageGeneration
    dictionary["MetaFile"] = GWMetaFile if gw_results_file() else MetaFile
    dictionary["FinishingTouches"] = FinishingTouches
    return dictionary


def get_version_information():
    """Grab the version from the .version file
    """
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


def setup_logger():
    """Set up the logger output.
    """
    level = 'INFO'
    if "-v" in sys.argv or "--verbose" in sys.argv:
        level = 'DEBUG'

    logger = logging.getLogger('PESummary')
    logger.setLevel(level)
    FORMAT = '%(asctime)s %(name)s %(levelname)-8s: %(message)s'
    logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d  %H:%M:%S')


setup_logger()
logger = logging.getLogger('PESummary')
