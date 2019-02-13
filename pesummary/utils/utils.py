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
import numpy as np

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
    check_condition(condition, "The file %s does not exist" %(base_file))
    f = h5py.File(base_file, "a")
    if group:
        f[group[1]] = f[group[0]]
        del f[group[0]]
    elif dataset:
        f[dataset[1]] = f[dataset[0]]
        del f[dataset[0]]
    f.close()

def add_content_to_hdf_file(base_file, dataset_name, content, group=None):
    """Add new content to an hdf5 file

    Parameters
    ----------
    base_file: str
        path to the file that you want to add content to
    dataset_name: str
        name of the dataset
    content: array
        array of content that you want to add to your hdf5 file
    group: str, optional
        group that you want to add content to. Default if the base of the file
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The file %s does not exist" %(base_file))
    f = h5py.File(base_file, "a")
    if group:
        group = f[group]
        if dataset_name in list(group.keys()):
            del group[dataset_name]
        group.create_dataset(dataset_name, data=content)
    else:
        if dataset_name in list(f.keys()):
            del f[dataset_name]
        f.create_dataset(dataset_name, data=content)
    f.close()

def combine_hdf_files(base_file, new_file):
    """Combine two hdf5 files

    Parameters
    ----------
    base_file: str
        path to the file that you want to add content to
    new_file: str
        path to the file that you want to combine with the base file
    """
    condition = not os.path.isfile(base_file)
    check_condition(condition, "The base file %s does not exist" %(base_file))
    condition = not os.path.isfile(new_file)
    check_condition(condition, "The new file %s does not exist" %(new_file))
    g = h5py.File(new_file)
    label = list(g.keys())[0]
    approximant = list(g[label].keys())[0]
    path = "%s/%s" %(label, approximant)
    parameters = np.array([i for i in g["%s/parameter_names" %(path)]])
    samples = np.array([i for i in g["%s/samples" %(path)]])
    injection_parameters = np.array([i for i in g["%s/injection_parameters" %(path)]])
    injection_data = np.array([i for i in g["%s/injection_data" %(path)]])
    g.close()

    f = h5py.File(base_file, "a")
    current_labels = list(f.keys())
    if label not in current_labels:
        label_group = f.create_group(label)
        approx_group = label_group.create_group(approximant)
    else:
        approx_group = f[label].create_group(approximant)
    approx_group.create_dataset("parameter_names", data=parameters)
    approx_group.create_dataset("samples", data=samples)
    approx_group.create_dataset("injection_parameters", data=injection_parameters)
    approx_group.create_dataset("injection_data", data=injection_data)
    f.close()

def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

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
    ligo_data_grid=False
    if 'public_html' in web_dir:
        ligo_data_grid=True
    if ligo_data_grid:
        path = web_dir.split("public_html")[1]
        if "raven" in host or "arcca" in host:
            url = "https://geo2.arcca.cf.ac.uk/~{}".format(user)
        elif "cit" in host or "caltech" in host:
            url = "https://ldas-jobs.ligo.caltech.edu/~{}".format(user)
        elif 'ligo-wa' in host:
            url = "https://ldas-jobs.ligo-wa.caltech.edu/~{}".format(user)
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

def setup_logger():
    """Set up the logger output.
    """
    level = 'INFO'
    if "-v" or "--verbose" in sys.argv:
        level = 'DEBUG'

    logger = logging.getLogger('PESummary')
    logger.setLevel(level)
    FORMAT = '%(asctime)s %(name)s %(levelname)-8s: %(message)s'
    logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d  %H:%M:%S')

setup_logger()                                                                  
logger = logging.getLogger('PESummary')
