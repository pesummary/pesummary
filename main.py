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
from glob import glob

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
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Posterior samples hdf5 file", nargs='+',
                        default=None)
    parser.add_argument("-a", "--approximant", dest="approximant",
                        help="waveform approximant used to generate samples",
                        nargs='+', default=None)
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

def _grab_key_data(results):
    """Grab the key data for each parameter in the samples file.

    Parameters
    ----------
    results: str
        string to the results file
    """
    f = h5py.File(results)
    parameters = [i for i in f["posterior/block0_items"]]
    logL_index = parameters.index("log_likelihood")
    logL = [i[logL_index] for i in f["posterior/block0_values"]]
    if "log_likelihood" in parameters:
        parameters.remove("log_likelihood")
    data = {}
    for i in parameters:
        index = parameters.index
        samples = [j[index(i)] for j in f["posterior/block0_values"]]
        data[i] = {"mean": np.mean(samples),
                   "median": np.median(samples),
                   "maxL": samples[logL.index(np.max(logL))],
                   "std": np.std(samples)}
    return data

def _make_plot(parameter, app, samples, opts, latex_labels):
    """Actually make the plot

    Parameters
    ----------
    parameter: str
        name of the parameter that you want to plot
    app: str
        name of the approximant used to generate samples
    samples: list
        list of samples for parameter=parameter
    opts: argparse
        argument parser object to hold all information from command line
    latex_labels: dict
        dictionary of latex labels for each parameter
    """
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
    plt.savefig(opts.webdir + "/plots/1d_posterior_{}_{}.png".format(app, parameter))
    plt.close()

def _make_comparison(parameter, app1, app2, samples, samples2, opts, latex_labels):
    """Make the comparison pages

    Parameters
    ----------
    parameter: str
        name of the parameter that you want to plot
    app1: str
        name of waveform approximant used to generate samples
    app2: str
        name of waveform approximant used to generate samples2
    samples: list
        list of samples for the first waveform for parameter=parameter
    samples2: list
        list of samples for the second waveform for parameter=parameter
    opts: argparse
        argument parser object to hold all information from command line
    latex_labels: dict
        dictionary of latex labels for each parameter
    """
    fig = plt.figure()
    plt.hist(samples, histtype="step", bins=50, color="#8c6278", label=app1)
    plt.hist(samples2, histtype="step", bins=50, color="#228B22", label=app2)
    plt.xlabel(latex_labels[parameter], fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    plt.axvline(x=np.percentile(samples, 90), color="#8c6278", linestyle='--')
    plt.axvline(x=np.percentile(samples2, 90), color="#228B22", linestyle='--')
    plt.axvline(x=np.percentile(samples, 10), color="#8c6278", linestyle='--')
    plt.axvline(x=np.percentile(samples2, 10), color="#228B22", linestyle='--')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(opts.webdir + "/plots/combined_posterior_" + parameter + ".png")
    plt.close()

def make_plots(opts):
    """Generate the posterior sample plots

    Parameters
    ----------
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
    f = h5py.File(opts.samples[0])
    parameters = [i for i in f["posterior/block0_items"]]
    if "log_likelihood" in parameters:
        parameters.remove("log_likelihood")
    if opts.number != "one":
        g = h5py.File(opts.samples[1])
        for num, i in enumerate(parameters):
            samples = [j[num] for j in f["posterior/block0_values"]]
            samples2 = [j[num] for j in g["posterior/block0_values"]]
            _make_plot(i, opts.approximant[0], samples, opts, latex_labels)
            _make_plot(i, opts.approximant[1], samples2, opts, latex_labels)
            _make_comparison(i, opts.approximant[0], opts.approximant[1], samples,
                             samples2, opts, latex_labels)
    else:
        for num, i in enumerate(parameters):
            samples = [j[num] for j in f["posterior/block0_values"]]
            _make_plot(i, opts.approximant[0], samples, opts, latex_labels)

def _single_html(opts):
    """Generate html pages for only one approximant

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # get a list of all parameters
    f = h5py.File(opts.samples[0])                                                
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    # make the webpages
    pages = ["{}_{}".format(opts.approximant[0], j) for j in parameters]
    pages.append("corner")
    pages.append("home")
    pages.append("{}".format(opts.approximant[0]))
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    # edit the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    html_file.make_navbar(links=[["Approximant", ["{}".format(opts.approximant[0])]]])
    # make a summary table of information
    data = _grab_key_data(opts.samples[0])
    contents = [[i, data[i]["maxL"], data[i]["mean"], data[i]["median"], data[i]["std"]] for i in parameters]
    html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                         contents=contents)
    html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # edit the home page for first approximant
    html_file = webpage.open_html(web_dir=opts.webdir,base_url=opts.baseurl,
                                  html_page="{}".format(opts.approximant[0]))
    # make header for home page for first approximant
    html_file.make_header(title="{} Summary Page".format(opts.approximant[0]),
                          background_colour="#8c6278")
    # what links do you want in your nav bar
    links = ["home", ["Approximant", ["{}".format(opts.approximant[0])]], "corner"]
    links.append(["1d_histograms", ["{}_{}".format(opts.approximant[0], j) for j in parameters]])
    # make nav bar for home page for first approximant
    html_file.make_navbar(links=links)
    # create array of images that we want to be inserted in table
    contents = [[opts.webdir+"/plots/"+"1d_posterior_{}_mass_1.png".format(opts.approximant[0]),
                 opts.webdir+"/plots/"+"1d_posterior_{}_mass_1.png".format(opts.approximant[0]),
                 opts.webdir+"/plots/"+"1d_posterior_{}_mass_1.png".format(opts.approximant[0])]]
    # create a table of images for first approximant
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=contents)
    html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # generate pages for all parameters
    for i in parameters:
        # edit the i parameter page
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="{}_{}".format(opts.approximant[0], i))
        # make header
        html_file.make_header(title="Posterior PDF for {}".format(i), background_colour="#8c6278")
        # what links do you want in your nav bar
        links=["home", ["Approximant", ["{}".format(opts.approximant[0])]], "corner"]
        links.append(["1d_histograms", ["{}_{}".format(opts.approximant[0], j) for j in parameters]])
        # make nav bar
        html_file.make_navbar(links=links)
        html_file.insert_image("{}/plots/1d_posterior_{}_{}.png".format(opts.baseurl, opts.approximant[0], i))
        # make footer
        html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))

def _double_html(opts):
    """Generate html pages for two approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # get a list of all parameters
    f = h5py.File(opts.samples[0])                                                
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    # make the webpages
    pages = ["{}_{}".format(opts.approximant[0], j) for j in parameters]
    for i in parameters:
        pages.append("{}_{}".format(opts.approximant[1], i))
        pages.append("Comparison_{}".format(i))
    pages.append("corner")
    pages.append("home")
    pages.append("{}".format(opts.approximant[0]))
    pages.append("{}".format(opts.approximant[1]))
    pages.append("Comparison")
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    # edit the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    # what links do you want in your nav bar
    links = [["Approximant", ["{}".format(opts.approximant[0]),
                              "{}".format(opts.approximant[1]),
                              "Comparison"]]]
    html_file.make_navbar(links=links)
    # make summary table of information
    data = _grab_key_data(opts.samples[0])
    data2 = _grab_key_data(opts.samples[1])
    contents = [[i, data[i]["maxL"], data2[i]["maxL"], data[i]["mean"], data2[i]["mean"],
                    data[i]["median"], data2[i]["median"], data[i]["std"], data2[i]["std"]] for i in parameters]
    html_file.make_table(headings=[None, "maxL", "mean", "median", "std"],
                         contents=contents, multi_span=True)
    html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # edit the comparison page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="Comparison")
    html_file.make_header(title="Comparison Summary Page")
    # what links do you want in yur nav bar
    links = ["home"]
    links.append(["Approximant", ["{}".format(opts.approximant[0]),
                                  "{}".format(opts.approximant[1]),
                                  "Comparison"]])
    links.append(["1d_histograms", ["Comparison_{}".format(i) for i in parameters]])
    html_file.make_navbar(links=links)
    html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # edit all comparison pages
    for i in parameters:
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="Comparison_{}".format(i))
        # edit the header
        html_file.make_header(title="Comparison page for {}".format(i))
        # make the nav bar
        html_file.make_navbar(links=links)
        # insert the comparison plot
        html_file.insert_image("{}/plots/combined_posterior_{}.png".format(opts.baseurl, i))
        # add the footer
        html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # edit the home pages for both approximants
    for app, col in zip([opts.approximant[0], opts.approximant[1]], ["#8c6278", "#228B22"]):
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="{}".format(app))
        # make header
        html_file.make_header(title="{} Summary Page".format(app),
                              background_colour=col)
        # what links do you want in your nav bar
        links = ["home", ["Approximant", ["{}".format(opts.approximant[0]),
                                          "{}".format(opts.approximant[1]),
                                          "Comparison"]],
                 "corner", ["1d_histograms", ["{}_{}".format(app, i) for i in parameters]]]
        # make nav bar
        html_file.make_navbar(links=links)
        # make a table of images
        html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                       contents=[["{}/plots/1d_posterior_{}_mass_1.png".format(opts.baseurl, app),
                                                  "{}/plots/1d_posterior_{}_mass_1.png".format(opts.baseurl, app),
                                                  "{}/plots/1d_posterior_{}_mass_1.png".format(opts.baseurl, app)]])
        # make footer
        html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
    # edit all parameters pages for both approximants
    for i in parameters:
        for app, col in zip([opts.approximant[0], opts.approximant[1]], ["#8c6278", "#228B22"]):    
            html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                          html_page="{}_{}".format(app, i))
            html_file.make_header(title="Posterior PDF for {}".format(i), background_colour=col)
            html_file.make_navbar(links=links)
            html_file.insert_image("{}/plots/1d_posterior_{}_{}.png".format(opts.baseurl, app, i))
            html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))

def write_html(opts):
    """Generate an html page to show posterior plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line 
    """
    # make the webpages
    options = {1: _single_html,
               2: _double_html}
    options[len(opts.samples)](opts)

if __name__ == '__main__':
    # get arguments from command line
    parser = command_line()
    opts = parser.parse_args()
    # make relevant directories
    utils.make_dir(opts.webdir + "/samples")
    utils.make_dir(opts.webdir + "/plots")
    utils.make_dir(opts.webdir + "/js")
    # check that number of samples matches number of approximants
    if len(opts.samples) != len(opts.approximant):
        raise Exception("Ensure that the number of approximants match the "
                        "number of samples files")
    # copy over the samples
    if os.path.isfile(opts.samples[0]) == False:
        raise Exception("File does not exist")
    shutil.copyfile(opts.samples[0], opts.webdir+"/samples/"+opts.samples[0].split("/")[-1])
    if len(opts.samples) != 1:
        if os.path.isfile(opts.samples[1]) == False:
            raise Exception("File does not exist")
        else:
            shutil.copyfile(opts.samples[1], opts.webdir+"/samples/"+opts.samples[1].split("/")[-1])
    # location of this file
    path = os.path.dirname(os.path.abspath(__file__))
    # copy over the javascript scripts
    shutil.copyfile(path+"/js/search.js", opts.webdir+"/js/search.js")
    make_plots(opts)
    write_html(opts)
    if opts.email:
        try:
            email_notify(opts.email, opts.baseurl+"/home.html")
        except Exception as e:
            print("Unable to send notification email because of error: {}".format(e))
