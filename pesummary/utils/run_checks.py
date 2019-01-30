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
import socket
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from pesummary.utils import utils
from pesummary.utils.utils import check_condition
from pesummary.utils.utils import rename_group_or_dataset_in_hf5_file

import h5py

from glob import glob

def run_checks(opts):
    """Check the command line inputs

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # check that there is a somewhere for us to save the data
    condition = not opts.webdir and not opts.existing
    check_condition(condition, ("Please provide a web directory to store the "
                                "webpages. If this is an existing directory "
                                "pass this path under the --existing_dir "
                                "argument. If this is a new directory then "
                                "pass this under the --webdir argument"))
    if opts.existing:
        # check to see if the existing directory actually exists
        condition = not os.path.isdir(opts.existing)
        check_condition(condition, ("The directory %s does not "
                                    "exist" %(opts.existing)))
        # check to see if the given existing directory is the base directory
        entries = glob(opts.existing+"/*")
        condition = "%s/home.html" %(opts.existing) not in entries
        check_condition(condition, ("Please give the base directory of an "
                                    "existing output"))
        opts.webdir = opts.existing
    # check to see if the web directory exists
    if not os.path.isdir(opts.webdir):
        logging.info("Given web directory does not exist. Creating it now")
        utils.make_dir(opts.webdir)
    # check that there is a results file
    for i in opts.samples:
        condition = not i
        check_condition(condition, "Please provide a results file")
    # check that there is a valid approximant
    if opts.approximant == None:
        logging.info("No approximant is given. Trying to extract from "
                     "results file")
        opts.approximant = []
        for i in opts.samples:
            f = h5py.File(i, "r")
            approx = list(f.keys())[0]
            condition = approx == "none"
            check_condition(condition, ("Failed to extract the approximant "
                                        "from the file: %s. Please pass the "
                                        "approximant with the flag "
                                        "--approximant" %(i.split("_temp")[0])))
            opts.approximant.append(approx)
    # check that there are the same number of approximants and result files
    condition = len(opts.samples) != len(opts.approximant)
    check_condition(condition, ("The number of results files does not match "
                                "the number of approximants"))
    # check that if add_to_existing is specified then existing html page
    # is also given
    condition = opts.add_to_existing and not opts.existing
    check_condition(condition, ("Please provide a current html page that you "
                                "wish to add content to"))
    if not opts.add_to_existing and opts.existing:
        opts.add_to_existing = True
        logging.info("Existing html page has been given without specifying "
                     "--add_to_existing flag. This is probably and error and so "
                     "manually adding --add_to_existing flag")
    # make relevant directories
    dirs = ["samples", "plots", "js", "html", "css", "plots/corner", "config"]
    for i in dirs:
        utils.make_dir(opts.webdir + "/{}".format(i))
    # check that numer of samples matches number of config files
    condition = opts.config and len(opts.samples) != len(opts.config)
    check_condition(condition, ("Ensure that the number of results files match "
                                "the number of configuration files"))
    # check that the number of injection fles match the number of results files
    condition = opts.inj_file and len(opts.inj_file) != len(opts.samples)
    check_condition(condition, ("Ensure that the number of samples match the "
                                "number of injection files"))
    for num, i in enumerate(opts.samples):
        condition = not os.path.isfile(i)
        check_condition(condition, "File %s does not exist" %(i))
        proposed_file = opts.webdir+"/samples/"+opts.approximant[num]+"_"+i.split("/")[-1]
        condition = os.path.isfile(proposed_file)
        check_condition(condition, ("File %s already exists under the name %s. "
                                    "Have you already generated a summary page "
                                    "with this file?" %(i, proposed_file)))
    if opts.add_to_existing and opts.existing:
        if opts.config:
            for i in glob(opts.existing+"/config/*"):
                opts.config.append(i)
        posterior_file = h5py.File(opts.existing+"/samples/posterior_samples.h5")
        approximants = list(posterior_file.keys())
        posterior_file.close()
        for i in approximants:
            if i not in opts.approximant:
                opts.approximant.append(i)
            else:
                logging.info("Data for the approximant %s already exists. This "
                             "approximant is being ignored" %(i))
    # check that the approximant in the data file is correct
    for num, i in enumerate(opts.samples):
        f = h5py.File(i, "r")
        condition = "none" in list(f.keys())
        f.close()
        if condition:
            rename_group_or_dataset_in_hf5_file(i,
                group=["none", opts.approximant[num]])
    # check to see if baseurl is provided. If not guess what it could be
    if opts.baseurl == None:
        try:
            user = os.environ["USER"]
            opts.user = user
        except Exception as e:
            logging.info("Failed to grab user information because %s. "
                         "Default will be used" %(e))
            user = opts.user
        host = socket.getfqdn()
        opts.baseurl = utils.guess_url(opts.webdir, host, user)
        logging.info("No url is provided. The url %s will be used" %(opts.baseurl))
    return opts
