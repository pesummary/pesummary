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
import corner

import numpy as np
from scipy.ndimage import gaussian_filter

import webpage
import utils

import h5py

import lal
import lalsimulation as lalsim

__doc__ == "Parameters to run post_processing.py from the command line"


def command_line():
    """Creates an ArgumentParser object which holds all of the information
    from the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--webdir", dest="webdir",
                        help="make page and plots in DIR", metavar="DIR")
    parser.add_argument("-b", "--baseurl", dest="baseurl",
                        help="make the page at this url", metavar="DIR",
                        default=None)
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Posterior samples hdf5 file", nargs='+',
                        default=None)
    parser.add_argument("-a", "--approximant", dest="approximant",
                        help="waveform approximant used to generate samples",
                        nargs='+', default=None)
    parser.add_argument("--email", action="store",
                        help="send an e-mail to the given address with a link to the finished page.",
                        default=None, metavar="user@ligo.org")
    parser.add_argument("--dump", action="store_true",
                        help="dump all information onto a single html page",
                        default=False)
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
    subject = "Output page available at {}".format(host)
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

def _make_comparison(opts, parameter, approximants, samples, colors, latex_labels):
    """Make the comparison pages

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    parameter: str
        name of the parameter that you want to plot
    approximants: list
        list of approximants that you would like to compare
    samples: 2d list
        list of samples for each waveform for parameter=parameter
    opts: argparse
        argument parser object to hold all information from command line
    latex_labels: dict
        dictionary of latex labels for each parameter
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    fig = plt.figure()
    for num, i in enumerate(samples):
        plt.hist(i, histtype="step", bins=50, color=colors[num], label=approximants[num], linewidth=2.0)
        plt.axvline(x=np.percentile(i, 90), color=colors[num], linestyle='--', linewidth=2.0)
        plt.axvline(x=np.percentile(i, 10), color=colors[num], linestyle='--', linewidth=2.0)
    plt.xlabel(latex_labels[parameter], fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(opts.webdir + "/plots/combined_posterior_" + parameter + ".png")
    plt.close()

def _make_corner_plots(opts, approximants, samples, latex_labels):
    """Make the corner plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximant: list
        list of approximants that you would like to analyse
    samples: 2d list
        list of samples for each approximant for parameter=parameter
    """
    # make the directory to store corner plots
    utils.make_dir(opts.webdir + "/plots/corner")
    # set the defaukt corner kwargs
    default_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)
    # get a list of parameters
    f = h5py.File(samples[0])
    parameters = [i for i in f["posterior/block0_items"]]
    if "log_likelihood" in parameters:
        parameters.remove("log_likelihood")
    # loop over each approximant and samples
    for app, sam in zip(approximants, samples):
        f = h5py.File(sam)
        # make the corner plot
        xs = np.zeros([len(parameters), len(f["posterior/block0_values"])])
        for num, i in enumerate(parameters):
            index = parameters.index
            xs[num] = [j[index(i)] for j in f["posterior/block0_values"]]
        # set the axis labels
        default_kwargs["labels"] = [latex_labels[i] for i in parameters]
        figure = corner.corner(xs.T, **default_kwargs)
        # grab the axes of the subplots
        axes = figure.get_axes()

        for i in xrange(1, len(parameters)):
            for j in xrange(1, i+1):
                ax = axes[i*len(parameters) + j - 1]
                extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
                # sort the parameters into alphabetical order
                ordered = sorted([parameters[i], parameters[j-1]])
                # only save each density plot
                plt.savefig('{}/plots/corner/{}_{}_{}_density_plot.png'.format(opts.webdir,
                                                                               app,
                                                                               ordered[0],
                                                                               ordered[1]),
                            bbox_inches=extent.expanded(1.05, 1.05))
        for i in xrange(len(parameters)-1):
            ax = axes[i*len(parameters)+i]
            extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
            plt.savefig('{}/plots/corner/{}_{}_histogram_plot.png'.format(opts.webdir,
                                                                           app,
                                                                           parameters[i]),
                        bbox_inches=extent.expanded(1.05, 1.05))
        # save the total corner plot
        plt.savefig("{}/plots/corner/{}_all_density_plot.png".format(opts.webdir,
                                                                     app))
        plt.close()

def _make_waveform_plots(opts, approximant, samples, colors, reference_frequency=10.0):
    """Plot the maximum likelihood frequency for each approximant

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximant: list
        list of approximants that you would like to analyse
    samples: list
        list of samples for each approximant
    reference_frequency: float, optional
        
    """
    delta_frequency = 1./256
    minimum_frequency = 10.
    maximum_frequency = 1000
    frequency_array = np.arange(minimum_frequency, maximum_frequency, delta_frequency)
    
    data = [_grab_key_data(i) for i in samples]
    waveform = {}
    for num, i in enumerate(approximant):
        approx = lalsim.GetApproximantFromString(i)
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                data[num]["iota"]["maxL"], data[num]["phi_jl"]["maxL"],
                data[num]["tilt_1"]["maxL"], data[num]["tilt_1"]["maxL"],
                data[num]["phi_12"]["maxL"], data[num]["a_1"]["maxL"],
                data[num]["a_2"]["maxL"], data[num]["mass_1"]["maxL"]*lal.MSUN_SI,
                data[num]["mass_2"]["maxL"]*lal.MSUN_SI, reference_frequency,
                data[num]["phase"]["maxL"])
        h_plus, h_cross = lalsim.SimInspiralChooseFDWaveform(data[num]["mass_1"]["maxL"]*lal.MSUN_SI,
                                                             data[num]["mass_2"]["maxL"]*lal.MSUN_SI,
                                                             spin_1x, spin_1y,
                                                             spin_1z, spin_2x,
                                                             spin_2y, spin_2z,
                                                             data[num]["luminosity_distance"]["maxL"],
                                                             iota, data[num]["phase"]["maxL"],
                                                             0.0, 0.0, 0.0, delta_frequency,
                                                             minimum_frequency, maximum_frequency,
                                                             reference_frequency, None, approx)
        h_plus = h_plus.data.data
        h_cross = h_cross.data.data
        h_plus = h_plus[:len(frequency_array)]
        h_cross = h_cross[:len(frequency_array)]
        waveform[i] = h_plus
        fig = plt.figure()
        plt.plot(frequency_array, h_plus, color='b', linewidth=2.0) 
        plt.xscale("log")
        plt.grid()
        plt.savefig("{}/plots/{}_waveform.png".format(opts.webdir, i))
        plt.close()
    if len(approximant) > 1:
        fig = plt.figure()
        for num, i in enumerate(approximant):
            plt.plot(frequency_array, waveform[i], color=colors[num], label=i, linewidth=2.0)
        plt.xscale("log")
        plt.grid()
        plt.legend(loc="best")
        plt.xlabel("Frequency $[Hz]$", fontsize=16)
        plt.ylabel(r"Strain $[1/\sqrt{Hz}]$", fontsize=16)
        plt.savefig("{}/plots/compare_waveforms.png".format(opts.webdir))
        plt.close()

def _make_sky_maps(opts, approximant, samples, colors):
    """
    """
    levels = [1.0 - np.exp(-0.5), 1 - np.exp(-2), 1-np.exp(-9./2.)]
    contours = {}
    for i, col, app in zip(samples, colors, approximant):
        data = h5py.File(i)
        index_ra = [i for i in data["posterior/block0_items"]].index("ra")
        index_dec = [i for i in data["posterior/block0_items"]].index("dec")
        ra = [i[index_ra] for i in data["posterior/block0_values"]]
        dec = [i[index_dec] for i in data["posterior/block0_values"]]
        # generate 2d histogram
        H, X, Y = np.histogram2d(ra, dec, bins=50)
        H = gaussian_filter(H, 0.9)
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        # normalise the pdf to 1
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()
        # Compute the bin centers.
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])
        contours[app] = [X2, Y2, H2.T, V]
    for i in approximant:
        fig = plt.figure()
        ax = plt.subplot(111, projection="hammer")
        ax.cla()
        ax.grid()
        CS = plt.contour(contours[i][0],contours[i][1], contours[i][2], contours[i][3], colors='b', linewidths=2)
        plt.savefig("{}/plots/{}_skymap.png".format(opts.webdir, i))
    if len(approximant) > 1:
        fig = plt.figure()
        ax = plt.subplot(111, projection="hammer")
        ax.cla()
        ax.grid()
        for i in approximant:
            CS = plt.contour(contours[i][0], contours[i][1], contours[i][2], contours[i][3], colors=col, linewidths=2)
            CS.collections[0].set_label(app)
        plt.legend(loc="best")
        plt.savefig("{}/plots/combined_skymap.png".format(opts.webdir))

def make_plots(opts, colors=None):
    """Generate the posterior sample plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    colors: list
        list of colors in hexadecimal format for the different approximants
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
    #_make_corner_plots(opts, opts.approximant, opts.samples, latex_labels)
    _make_waveform_plots(opts, opts.approximant, opts.samples, colors)
    _make_sky_maps(opts, opts.approximant, opts.samples, colors)
    if len(opts.approximant) > 1:
        g = h5py.File(opts.samples[1])
        for num, i in enumerate(parameters):
            samples = [[j[num] for j in f["posterior/block0_values"]],
                       [k[num] for k in g["posterior/block0_values"]]]
            for app, sam in zip(opts.approximant, samples):
                _make_plot(i, app, sam, opts, latex_labels)
            _make_comparison(opts, i, opts.approximant, samples,
                             colors, latex_labels)
    else:
        for num, i in enumerate(parameters):
            samples = [j[num] for j in f["posterior/block0_values"]]
            _make_plot(i, opts.approximant[0], samples, opts, latex_labels)

def make_home_pages(opts, approximants, samples, colors):
    """Make the home pages for all approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximants: list
        list of approximants you wish to include
    samples: list
        list of samples you wish to include
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    f = h5py.File(samples[0])
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    pages = [i for i in approximants]
    pages.append("home")
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    # design the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    if len(approximants) > 1:
        links = ["home", ["Approximants", [i for i in approximants+["Comparison"]]]]
    else:
        links = ["home", ["Approximants", [i for i in approximants]]]
    html_file.make_navbar(links=links)
    # make summary table of information
    data = [_grab_key_data(i) for i in samples]
    contents = []
    for i in parameters:
        row = []
        row.append(i)
        for j in xrange(len(samples)):
            row.append(np.round(data[j][i]["maxL"], 3))
        for j in xrange(len(samples)):
            row.append(np.round(data[j][i]["mean"], 3))
        for j in xrange(len(samples)):
            row.append(np.round(data[j][i]["median"], 3))
        for j in xrange(len(samples)):
            row.append(np.round(data[j][i]["std"], 3))
        contents.append(row)
    html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                         contents=contents, heading_span=len(samples),
                         colors=colors[:len(samples)])
    # design the home page for each approximant used
    for num, i in enumerate(approximants):
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page=i)
        html_file.make_header(title="{} Summary Page".format(i),
                              approximant="{}".format(i),
                              background_colour=colors[num])
        if len(approximants) > 1:
            links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                     "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
        else:
            links = ["home", ["Approximants", [k for k in approximants]],
                     "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
        html_file.make_navbar(links=links)
        # make an array of images that we want inserted in table
        contents = [["{}/plots/{}_skymap.png".format(opts.baseurl, i),
                     "{}/plots/{}_waveform.png".format(opts.baseurl, i),
                     "{}/plots/1d_posterior_{}_mass_1.png".format(opts.baseurl, i)]]
        html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                       contents=contents)
        html_file.make_footer(user='c1737564', rundir=opts.webdir)
        
def make_1d_histograms_pages(opts, approximants, samples, colors):
    """Make the 1d_histogram pages for all approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximants: list
        list of approximants you wish to include
    samples: list
        list of samples you wish to include
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    f = h5py.File(samples[0])
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    pages = ["{}_{}".format(i, j) for i in approximants for j in parameters]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for i in parameters:
        for app, col in zip(approximants, colors):
            if len(approximants) > 1:
                links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                         "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
            else:
                links = ["home", ["Approximants", [k for k in approximants]],
                         "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
            html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                          html_page="{}_{}".format(app, i))
            html_file.make_header(title="{} Posterior PDF for {}".format(app, i), background_colour=col,
                                  approximant=app)
            html_file.make_navbar(links=links)
            html_file.insert_image("{}/plots/1d_posterior_{}_{}.png".format(opts.baseurl, app, i))
            html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))

def make_comparison_pages(opts, approximants, samples, colors):
    """Make the comparison pages to compare all approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximants: list
        list of approximants you wish to include
    samples: list
        list of samples you wish to include
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    f = h5py.File(samples[0])
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    webpage.make_html(web_dir=opts.webdir, pages=["Comparison"])
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                   html_page="Comparison")
    html_file.make_header(title="Comparison Summary Page")
    links = ["home", ["Approximant", [i for i in approximants+["Comparison"]]],
                     ["1d_histograms", ["{}".format(i) for i in parameters]]]
    html_file.make_navbar(links=links)
    contents = [["{}/plots/combined_skymap.png".format(opts.baseurl),
                 "{}/plots/compare_waveforms.png".format(opts.baseurl)]]
    html_file.make_table_of_images(headings=["sky_map", "waveform"],
                                   contents=contents)
    html_file.make_footer(user="c1737564", rundir=opts.webdir)
    # edit all of the comparison pages
    pages = ["Comparison_{}".format(i) for i in parameters]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for i in parameters:
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="Comparison_{}".format(i))
        html_file.make_header(title="Comparison page for {}".format(i),
                              approximant="Comparison")
        html_file.make_navbar(links=links)
        html_file.insert_image("{}/plots/combined_posterior_{}.png".format(opts.baseurl, i))
        html_file.make_footer(user="c1737564", rundir=opts.webdir)

def make_corner_pages(opts, approximants, samples, colors):
    """Make the corner pages for all approximants

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    approximants: list
        list of approximants you wish to include
    samples: list
        list of samples you wish to include
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    f = h5py.File(samples[0])
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    pages = ["{}_corner".format(i) for i in approximants]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for app, col in zip(approximants, colors):
        if len(approximants) > 1:
            links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                     "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
        else:
            links = ["home", ["Approximants", [k for k in approximants]],
                     "corner", ["1d_histograms", ["{}".format(j) for j in parameters]]]
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="{}_corner".format(app))
        html_file.make_header(title="{} Corner plots".format(app), background_colour=col,
                              approximant=app)
        html_file.make_navbar(links=links)
        html_file.make_search_bar()
        html_file.make_footer(user="c1737564", rundir="{}".format(opts.webdir))
 
def write_html(opts, colors):
    """Generate an html page to show posterior plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    colors: list
        list of colors in hexadecimal format for the different approximants 
    """
    # make the webpages
    make_home_pages(opts, opts.approximant, opts.samples, colors)
    make_1d_histograms_pages(opts, opts.approximant, opts.samples, colors)
    make_corner_pages(opts, opts.approximant, opts.samples, colors)
    if len(opts.approximant) != 1:
        make_comparison_pages(opts, opts.approximant, opts.samples, colors)

def write_html_data_dump(opts, colors):
    """Generate a single html page to show posterior plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    # grab the parameters from the samples
    f = h5py.File(opts.samples[0])
    parameters = [i for i in f["posterior/block0_items"]]                       
    if "log_likelihood" in parameters:                                          
        parameters.remove("log_likelihood")
    # make the relevant pages
    pages = ["home"]
    # links for all pages
    if len(opts.approximant) > 1:
        for i in opts.approximant:
            pages.append(i)
        links = ["home", ["Approximant", [i for i in opts.approximant]]]
    else:
        links = ["home"]
    # make the relevant pages
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    if len(opts.approximant) > 1:
        # setup the home comparison page
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="home")
        html_file.make_header()
        html_file.make_navbar(links=links)
        # content for accordian
        content = ["{}/plots/combined_posterior_{}.png".format(opts.baseurl, i) for i in parameters]
        html_file.make_accordian(headings=[i for i in parameters], content=content)
        html_file.make_footer(user="c1737564", rundir=opts.webdir)
    for num, i in enumerate(opts.approximant):
        if len(opts.approximant) == 1:
            html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                          html_page="home")
            html_file.make_header()
        else:
            html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                          html_page=i)
            html_file.make_header("{} Summary Page".format(i), background_colour=colors[num])
        html_file.make_navbar(links=links)
        # content for accordian
        content = ["{}/plots/1d_posterior_{}_{}.png".format(opts.baseurl, i, j) for j in parameters]
        html_file.make_accordian(headings=[i for i in parameters], content=content)
        html_file.make_footer(user="c1737564", rundir=opts.webdir)

if __name__ == '__main__':
    # default colors
    colors = ["#a6b3d0", "#baa997", "#FF6347", "#FFA500", "#003366"]
    # get arguments from command line
    parser = command_line()
    opts = parser.parse_args()
    # check to see if base_url is given
    if opts.baseurl == None:
        opts.baseurl = utils.guess_url(opts.webdir)
    # make relevant directories
    utils.make_dir(opts.webdir + "/samples")
    utils.make_dir(opts.webdir + "/plots")
    utils.make_dir(opts.webdir + "/js")
    utils.make_dir(opts.webdir + "/html")
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
    shutil.copyfile(path+"/js/combine_corner.js", opts.webdir+"/js/combine_corner.js")
    shutil.copyfile(path+"/js/grab.js", opts.webdir+"/js/grab.js")
    shutil.copyfile(path+"/js/variables.js", opts.webdir+"/js/variables.js")
    make_plots(opts, colors=colors)
    if opts.dump:
        write_html_data_dump(opts, colors=colors)
    else:
        write_html(opts, colors=colors)
    if opts.email:
        try:
            email_notify(opts.email, opts.baseurl+"/home.html")
        except Exception as e:
            print("Unable to send notification email because of error: {}".format(e))
