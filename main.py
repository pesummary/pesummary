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

import argparse
import subprocess
import socket
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import webpage
import utils

import h5py

__doc__ == "Parameters to run post_processing.py from the command line"


def command_line():
    """Creates an ArgumentParser object which holds all of the information
    from the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR")
    parser.add_argument("-b", "--baseurl", dest="baseurl",
                        help="make the page at this url", metavar="DIR")
    parser.add_argument("-n", "--number_of_waveforms", dest="number",
                        help="the number of approximants you wish to compare",
                        metavar="int", default="one")
    parser.add_argument("-s1", "--samples1", dest="samples1",
                        help="Posterior samples hdf5 file", metavar="results.h5",
                        default=None)
    parser.add_argument("-s2", "--samples2", dest="samples2",
                        help="Posterior samples hdf5 file", metavar="results.h5",
                        default=None)
    parser.add_argument("--email", action="store",
                        help="Send an e-mail to the given address with a link to the finished page.",
                        default=None, metavar="user@ligo.org")
    return parser


def email_notify(address, path):
    """Send an email to notify the user that their output page is generated.

    Parameters
    ----------
    address: str
        email address that you want the output page to be emailed to
    path:str
        path to the directory where the html page will be generated
    """
    user = os.environ["USER"]
    host = socket.getfqdn()
    from_address = "{}@{}".format(user, host)
    subject = "BILBY output page available at {}".format(host)
    message = "Hi {},\n\nYour output page is ready on {}. You can view the result at {}\n".format(user, host, path)
    cmd = 'echo -e "%s" | mail -s "%s" "%s"' %(message, subject, address)
    ess = subprocess.Popen(cmd, shell=True)
    ess.wait()

def _make_plot(parameter, samples, opts):
    """Actually make the plot

    Parameters
    ----------
    parameter: str
        name of the parameter that you want to plot
    samples: list
        list of samples for parameter=parameter
    opts: argparse
        argument parser object to hold all information from command line 
    """
    latex_labels={"luminosity_distance": r"$d_{L} [Mpc]$",
                  "geocent_time": r"$t_{c} [s]$",
                  "dec": r"$\delta$",
                  "ra": r"$\alpha$",
                  "a_1": r"$a_{1}$",
                  "a_2": r"$a_{2}$",
                  "phi_jl": r"$\phi_{JL}$",
                  "phase": r"$\phi$",
                  "psi": r"$\Psi$",
                  "iota": r"$\iota$",
                  "tilt_1": r"$\theta_{1}$",
                  "tilt_2": r"$\theta_{2}$",
                  "phi_12": r"$\phi_{12}$",
                  "mass_2": r"$m_{2}$",
                  "mass_1": r"$m_{1}$"}
    fig = plt.figure()
    plt.hist(samples, histtype="step", bins=50, color='b')
    plt.xlabel(latex_labels[parameter], fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    plt.axvline(x=np.percentile(samples, 90), color='b', linestyle='--')
    plt.axvline(x=np.percentile(samples, 10), color='b', linestyle='--')
    median = np.round(np.median(samples), 2)
    upper = np.round(np.percentile(samples, 90), 2)
    lower = np.round(np.percentile(samples, 10), 2)
    plt.title(r"$%s^{+%s}_{-%s}$" %(median, upper, lower), fontsize=18)
    plt.grid()
    plt.savefig(opts.webdir + "/plots/1d_posterior_" + parameter + ".png")
    plt.close()

def make_plots(opts):
    """Generate the posterior sample plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    if os.path.isfile(opts.samples1) == False:
        raise Exception("File does not exist")
    # copy the hdf5 file to the webdir
    shutil.copyfile(opts.samples1, opts.webdir+"/samples/"+opts.samples1.split("/")[-1])
    f = h5py.File(opts.samples1)
    parameters = [i for i in f["posterior/block0_items"]]
    if "log_likelihood" in parameters:
        parameters.remove("log_likelihood")
    for num, i in enumerate(parameters):
        samples = [j[num] for j in f["posterior/block0_values"]]
        _make_plot(i, samples, opts)

def _single_html(opts):
    """Generate html pages for only one approximant

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # make the webpage
    webpage.make_html(web_dir=opts.webdir,
                      pages=["corner", "IMRPhenomPv2", "IMRPhenommass1", "home"])
    # edit the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    html_file.make_navbar(links=[["Approximant", ["IMRPhenomPv2"]]])
    # edit the home page for IMRPhenomPv2
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="IMRPhenomPv2")
    html_file.make_header(title="IMRPhenomPv2 Summary Page", background_colour="#8c6278")
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2"]], "corner",
                                 ["1d_histograms", ["IMRPhenommass1"]]])
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=[[opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png"]])
    html_file.make_footer(user="c1737564", rundir="./")
    # edit the mass 1 page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="IMRPhenommass1")
    html_file.make_header(title="Posterior PDF for mass1", background_colour="#8c6278")
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2"]], "corner",
                                 ["1d_histograms", ["IMRPhenommass1"]]])
    html_file.make_footer(user="c1737564", rundir="./")

def _double_html(opts):
    """Generate html pages for two approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    webpage.make_html(web_dir=opts.webdir,
                      pages=["corner", "IMRPhenomPv2", "SEOBNRv3", "IMRPhenommass1", "SEOBNRmass1", "home"])
    # edit the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    html_file.make_navbar(links=[["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]]])
    # edit the home page for IMRPhenomPv2
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="IMRPhenomPv2")
    html_file.make_header(title="IMRPhenomPv2 Summary Page", background_colour="#8c6278")
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                 "corner", ["1d_histograms", ["IMRPhenommass1"]]])
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=[[opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png"]])
    html_file.make_footer(user="c1737564", rundir="./")
    # edit the home page for SEOBNRv3
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="SEOBNRv3")
    html_file.make_header(title="SEOBNRv3 Summary Page", background_colour="#228B22") 
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                 "corner", ["1d_histograms", ["SEOBNRmass1"]]])
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=[[opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png",
                                              opts.webdir+"/plots/"+"1d_posterior_mass_1.png"]])
    html_file.make_footer(user="c1737564", rundir="./")
    # edit the mass1 page for both approximants
    for i,j in zip(["IMRPhenommass1", "SEOBNRmass1"], ["#8c6278", "#228B22"]):    
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page=i)
        html_file.make_header(title="Posterior PDF for mass1", background_colour=j)
        html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                     "corner", ["1d_histograms", [i]]])
        html_file.make_footer(user="c1737564", rundir="./")

def write_html(opts):
    """Generate an html page to show posterior plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line 
    """
    # make the webpages
    options = {"one": _single_html,
               "two": _double_html}
    options[opts.number](opts)

if __name__ == '__main__':
    # get arguments from command line
    parser = command_line()
    opts = parser.parse_args()
    # make relevant directories
    utils.make_dir(opts.webdir + "/plots")
    utils.make_dir(opts.webdir + "/samples")
    #make_plots(opts)
    write_html(opts)
    if opts.email:
        try:
            email_notify(opts.email, opts.baseurl+"/home.html")
        except Exception as e:
            print("Unable to send notification email because of error: {}".format(e))
