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

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import argparse
import subprocess
import socket
import os
import shutil
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import webpage
import utils
import plot
from _version import __bilby_version__
from data_format import one_format

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
    parser.add_argument("-c", "--config", dest="config",
                        help="configuration file associcated with each samples file.", nargs='+',
                        default=None)
    parser.add_argument("--sensitivity", action="store_true",
                        help="generate sky sensitivities for HL, HLV",
                        default=False)
    parser.add_argument("--add_to_existing", action="store_true",
                        help="add new results to an existing html page",
                        default=False)
    parser.add_argument("-e", "--existing_webdir", dest="existing",
                        help="web directory of existing output",
                        default=None)
    return parser

def convert_to_standard_format(samples):
    """Convert the input files to the standard format

    Parameters
    ----------
    samples: str/list
        either a string or a list of strings giving the path to the input
        samples file
    """
    for num, i in enumerate(samples):
        opts.samples[num] = one_format(i)

def run_checks(opts):
    """Check the command line inputs

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # make the web directory
    utils.make_dir(opts.webdir)
    # check the command line arguments
    if opts.webdir:
        if opts.samples and opts.approximant and opts.config:
            pass
        else:
            raise Exception("Please run python main.py --samples [results.hdf] "
                            "--approximant [approx] --config [config.ini]")
    # check that if add_to_existing is specified then existing html page
    # is also given
    if opts.add_to_existing and opts.existing == None:
        raise Exception("Please provide a current html page that you wish "
                        "to add content to")
    if not opts.add_to_existing and opts.existing:
        opts.add_to_existing = True
        logging.info("Existing html page has been given without specifying "
                     "--add_to_existing flag. This is probably and error and so "
                     "manually adding --add_to_existing flag")
    # make relevant directories
    dirs = ["samples", "plots", "js", "html", "css", "plots/corner", "config"]
    if opts.webdir:
        for i in dirs:
            utils.make_dir(opts.webdir + "/{}".format(i))
    if opts.existing:
        # check to see if the existing directory actually exists
        if not os.path.isdir(opts.existing):
            raise Exception("The directory %s does not exist" %(opts.existing))
        # check to see if the given existing directory is the base directory
        entries = glob(opts.existing+"/*")
        if "%s/home.html" %(opts.existing) not in entries:
            raise Exception("Please give the base directory of an existing "
                            "output")
        opts.webdir = opts.existing
    # check to see if webdir exists
    if not os.path.isdir(opts.webdir):
        logging.info("Given web directory does not exist. Creating it now")
        utils.make_dir(opts.webdir)
    # check that number of samples matches number of approximants
    if len(opts.samples) != len(opts.approximant):
        raise Exception("Ensure that the number of approximants match the "
                        "number of samples files")
    # check that numer of samples matches number of config files
    if len(opts.samples) != len(opts.config):
        raise Exception("Ensure that the number of samples files match the "
                        "number of config files")
    for num, i in enumerate(opts.samples):
        if os.path.isfile(i) == False:
            raise Exception("File %s does not exist" %(i))
        proposed_file = opts.webdir+"/samples/"+opts.approximant[num]+"_"+i.split("/")[-1]
        if os.path.isfile(proposed_file):
            raise Exception("File %s already exists under the name %s. Have "
                            "you already generated a summary page with this "
                            "file?" %(i, proposed_file))
    if opts.add_to_existing and opts.existing:
        for i in glob(opts.existing+"/config/*"):
            opts.config.append(i)
        for i in glob(opts.existing+"/html/*_corner.html"):
            example_file = i.split("/")[-1]
            opts.approximant.append(example_file.split("_corner.html")[0])
    # check to see if baseurl is provided. If not guess what it could be
    if opts.baseurl == None:
        opts.baseurl = utils.guess_url(opts.webdir)
        logging.info("No url is provided. The url %s will be used" %(opts.baseurl))

def copy_files(opts):
    """Copy over files to the web directory

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    """
    # copy over the javascript scripts
    path = os.path.dirname(os.path.abspath(__file__))
    scripts = ["search.js", "combine_corner.js", "grab.js", "multi_dropbar.js",
               "multiple_posteriors.js", "side_bar.js"]
    for i in scripts:
        shutil.copyfile(path+"/js/%s" %(i), opts.webdir+"/js/%s" %(i))
    # copy over the css scripts
    scripts = ["image_styles.css", "side_bar.css"]
    for i in scripts:
        shutil.copyfile(path+"/css/%s" %(i), opts.webdir+"/css/%s" %(i))
    # copy over the config file
    for num, i in enumerate(opts.config):
        if opts.webdir not in i:
            shutil.copyfile(i, opts.webdir+"/config/"+opts.approximant[num]+"_"+i.split("/")[-1])
    for num, i in enumerate(opts.samples):
        if opts.webdir not in i:
            shutil.copyfile(i, opts.webdir+"/samples/"+opts.approximant[num]+"_"+i.split("/")[-1])

def email_notify(address, path):
    """Send an email to notify the user that their output page is generated.

    Parameters
    ----------
    address: str
        email address that you want the output page to be emailed to
    path:str
        path to the directory where the html page will be generated
    """
    logging.info("Sending email to %s" %(address))
    user = os.environ["USER"]
    host = socket.getfqdn()
    from_address = "{}@{}".format(user, host)
    subject = "Output page available at {}".format(host)
    message = "Hi {},\n\nYour output page is ready on {}. You can view the result at {}\n".format(user, host, path)
    cmd = 'echo -e "%s" | mail -s "%s" "%s"' %(message, subject, address)
    ess = subprocess.Popen(cmd, shell=True)
    ess.wait()

def _grab_parameters(results):
    """Grab the list of parameters that the sampler varies over

    Parameters
    ----------
    results: str
        string to the results file
    """
    # grab the parameters from the samples
    f = h5py.File(opts.samples[0])
    parameters = [i for i in f["parameter_names"]]
    f.close()                   
    return parameters

def _grab_key_data(samples, logL, parameters):
    """Grab the key data for each parameter in the samples file.

    Parameters
    ----------
    samples: list
        list of samples you wish to include
    logL: list
        list of likelihoods for each sample
    parameters: list
        list of parameters that the sampler varies over
    """
    data = {}
    for i in parameters:
        index = parameters.index
        subset = [j[index(i)] for j in samples]
        data[i] = {"mean": np.mean(subset),
                   "median": np.median(subset),
                   "maxL": subset[logL.index(np.max(logL))],
                   "std": np.std(subset)}
    return data

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
                  "mass_1": r"$m_{1}$",
                  "total_mass": r"$M$",
                  "chirp_mass": r"$\mathcal{M}$",
                  "log_likelihood": r"$\log{\mathcal{L}}$",
                  "H1_matched_filter_snr": r"$\rho^{H}_{mf}$",
                  "L1_matched_filter_snr": r"$\rho^{L}_{mf}$",
                  "H1_optimal_snr": r"$\rho^{H}_{opt}$",
                  "L1_optimal_snr": r"$\rho^{L}_{opt}$",
                  "spin_1x": r"$S_{1}x$",
                  "spin_1y": r"$S_{1}y$",
                  "spin_1z": r"$S_{1}z$",
                  "spin_2x": r"$S_{2}x$",
                  "spin_2y": r"$S_{2}y$",
                  "spin_2z": r"$S_{2}z$",
                  "chi_p": r"$\chi_{p}$",
                  "chi_eff": r"$\chi_{eff}$",
                  "mass_ratio": r"$q$",
                  "symmetric_mass_ratio": r"$\eta$",
                  "phi_1": r"$\phi_{1}$",
                  "phi_2": r"$\phi_{2}$",
                  "cos_tilt_1": r"$\cos{\theta_{1}}$",
                  "cos_tilt_2": r"$\cos{\theta_{2}}$",
                  "redshift": r"$z$",
                  "comoving_distance": r"$d_{com}$",
                  "mass_1_source": r"$m_{1}^{source}$",
                  "mass_2_source": r"$m_{2}^{source}$",
                  "chirp_mass_source": r"$\mathcal{M}^{source}$",
                  "total_mass_source": r"$M^{source}$"}
    # generate array of both samples
    combined_samples = []
    combined_maxL = []
    # get the parameter names
    parameters = _grab_parameters(opts.samples[0])
    ind_ra = parameters.index("ra")
    ind_dec = parameters.index("dec")
    # generate the individual plots
    for num, i in enumerate(opts.samples):
        approx = opts.approximant[num]
        logging.info("Starting to generate plots for %s" %(approx))
        with h5py.File(i) as f:
            params = [j for j in f["parameter_names"]]
            index = params.index("log_likelihood")
            samples = [j for j in f["samples"]]
            likelihood = [j[index] for j in samples]
            f.close()
        combined_samples.append(samples)
        data = _grab_key_data(samples, likelihood, parameters)
        ra = [j[ind_ra] for j in samples]
        dec = [j[ind_dec] for j in samples]
        maxL_params = {j: data[j]["maxL"] for j in parameters}
        maxL_params["approximant"] = approx
        combined_maxL.append(maxL_params)

        fig = plot._make_corner_plot(opts, samples, parameters, approx, latex_labels)
        plt.savefig("%s/plots/corner/%s_all_density_plots.png" %(opts.webdir, approx))
        plt.close()
        fig = plot._sky_map_plot(ra, dec)
        plt.savefig("%s/plots/%s_skymap.png" %(opts.webdir, approx))
        plt.close()
        fig = plot._waveform_plot(maxL_params)
        plt.savefig("%s/plots/%s_waveform.png" %(opts.webdir, approx))
        plt.close()
        for ind, j in enumerate(parameters):
            index = parameters.index(j)
            param_samples = [k[index] for k in samples]
            fig = plot._1d_histogram_plot(j, param_samples, latex_labels[j])
            plt.savefig("%s/plots/1d_posterior_%s_%s.png" %(opts.webdir, approx, j))
            plt.close()
        if opts.sensitivity:
            fig = plot._sky_sensitivity(["H1", "L1"], 0.2,
                                        combined_maxL[num])
            plt.savefig("%s/plots/%s_sky_sensitivity_HL" %(opts.webdir, approx))
            plt.close()
            fig = plot._sky_sensitivity(["H1", "L1", "V1"], 0.2,
                                        combined_maxL[num])
            plt.savefig("%s/plots/%s_sky_sensitivity_HLV" %(opts.webdir, approx))
            plt.close()
    # open the new results file and store the data
    if opts.add_to_existing and opts.existing:
        for i in glob(opts.existing+"/samples/*"):
            opts.samples.append(i)
        for num, i in enumerate(opts.samples):
            with h5py.File(i) as f:
                params = [j for j in f["parameter_names"]]
                index = params.index("log_likelihood")
                samples = [j for j in f["samples"]]
                likelihood = [j[index] for j in samples]
                f.close()
            combined_samples.append(samples)
            data = _grab_key_data(samples, likelihood, parameters)
            maxL_params = {j: data[j]["maxL"] for j in parameters}
            if opts.existing in i:
                approx = i.split("/")[-1].split("_")[0]
            else:
                approx = opts.approximant[num]
            maxL_params["approximant"] = approx
            combined_maxL.append(maxL_params)
        
    # if len(approximants) > 1, then we need to do comparison plots
    if len(opts.approximant) > 1:
        for ind, j in enumerate(parameters):
            index = parameters.index(j)
            param_samples = [[k[index] for k in l] for l in combined_samples]
            fig = plot._1d_comparison_histogram_plot(j, opts.approximant,
                                                     param_samples, colors,
                                                     latex_labels[j])
            plt.savefig("%s/plots/combined_posterior_%s" %(opts.webdir, j))
            plt.close()
        ra_list = [[k[ind_ra] for k in l] for l in combined_samples]
        dec_list = [[k[ind_dec] for k in l] for l in combined_samples]
        fig = plot._waveform_comparison_plot(combined_maxL, colors)
        plt.savefig("%s/plots/compare_waveforms.png" %(opts.webdir))
        plt.close()
        fig = plot._sky_map_comparison_plot(ra_list, dec_list, opts.approximant,
                                            colors)
        plt.savefig("%s/plots/combined_skymap.png" %(opts.webdir))
        plt.close()

def make_navbar_links(parameters):
    """Generate the links for the navbar

    Parameters
    ----------
    parameters: list
        list of parameters that were used in your study
    """
    links = ["1d_histograms", ["multiple"]]
    if any("mass" in s for s in parameters):
        condition = lambda i: True if "mass" in i and "source" not in i or "q" in i \
                              or "symmetric_mass_ratio" in i else False
        links.append(["masses", [i for i in parameters if condition(i)]])
    if any("spin" in s for s in parameters):
        condition = lambda i: True if "spin" in i or "chi_p" in i \
                              or "chi_eff" in i or "a_1" in i or "a_2" in i \
                              else False
        links.append(["spins", [i for i in parameters if condition(i)]])
    if any("source" in s for s in parameters):
        condition = lambda i: True if "source" in i else False
        links.append(["source_frame", [i for i in parameters if condition(i)]])
    if any("phi" in s for s in parameters):
        condition = lambda i: True if "phi" in i or "tilt" in i else False
        links.append(["spin_angles", [i for i in parameters if condition(i)]])
    if any("ra" in s for s in parameters):
        condition = lambda i: True if "ra" in i or "dec" in i or "psi" in i \
                              else False
        links.append(["sky_location", [i for i in parameters if condition(i)]])
    if any("snr" in s for s in parameters):
        links.append(["SNR", [i for i in parameters if "snr" in i]])
    if any("distance" in s for s in parameters):
        condition = lambda i: True if "distance" in i or "time" in i or \
                              "redshift" in i else False
        links.append(["Distance and Time", [i for i in parameters if condition(i)]])
    if any("phase" in s for s in parameters):
        condition = lambda i: True if "phase" in i or "likelihood" in i \
                              else False
        links.append(["Others", [i for i in parameters if condition(i)]])
    return links

def make_home_pages(opts, approximants, samples, colors, parameters):
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
    parameters: list
        list of parameters that the sampler varies over
    """
    pages = [i for i in approximants]
    pages.append("home")
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    # design the home page
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                  html_page="home")
    html_file.make_header()
    if len(approximants) > 1:
        links = ["home", ["Approximants", [i for i in approximants]+["Comparison"]]]
    else:
        links = ["home", ["Approximants", [i for i in approximants]]]
    html_file.make_navbar(links=links)
    # make summary table of information
    likelihood = []
    subset = []
    for i in samples:
        with h5py.File(i) as f:
            params = [j for j in f["parameter_names"]]
            index = params.index("log_likelihood")
            subset.append([j for j in f["samples"]])
            likelihood.append([j[index] for j in f["samples"]])
            f.close()
    data = [_grab_key_data(i, j, parameters) for i,j in zip(subset, likelihood)]
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
                     "corner", "config", make_navbar_links(parameters)]
        else:
            links = ["home", ["Approximants", [k for k in approximants]],
                     "corner", "config", make_navbar_links(parameters)]
        html_file.make_navbar(links=links)
        # make an array of images that we want inserted in table
        contents = [["{}/plots/1d_posterior_{}_mass_1.png".format(opts.baseurl, i),
                     "{}/plots/1d_posterior_{}_mass_2.png".format(opts.baseurl, i),
                     "{}/plots/1d_posterior_{}_luminosity_distance.png".format(opts.baseurl, i)],
                    ["{}/plots/{}_skymap.png".format(opts.baseurl, i),
                     "{}/plots/{}_waveform.png".format(opts.baseurl, i)],
                    ["{}/plots/1d_posterior_{}_iota.png".format(opts.baseurl, i),
                     "{}/plots/1d_posterior_{}_a_1.png".format(opts.baseurl, i),
                     "{}/plots/1d_posterior_{}_a_2.png".format(opts.baseurl, i)]]
        html_file.make_table_of_images(contents=contents)
        # make table of summary information
        contents = []
        for j in parameters:
            row = []
            row.append(j)
            row.append(np.round(data[num][j]["maxL"], 3))
            row.append(np.round(data[num][j]["mean"], 3))
            row.append(np.round(data[num][j]["median"], 3))
            row.append(np.round(data[num][j]["std"], 3))
            contents.append(row)
        html_file.make_table(headings=[" ", "maxL", "mean", "median", "std"],
                             contents=contents, heading_span=1)
        html_file.make_footer(user=os.environ["USER"], rundir=opts.webdir)
        
def make_1d_histograms_pages(opts, approximants, samples, colors, parameters):
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
    parameters: list
        list of parameters that the sampler varies over
    """
    parameters.append("multiple")
    pages = ["{}_{}".format(i, j) for i in approximants for j in parameters]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for i in parameters:
        for app, col in zip(approximants, colors):
            if len(approximants) > 1:
                links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                         "corner", "config", make_navbar_links(parameters)]
            else:
                links = ["home", ["Approximants", [k for k in approximants]],
                         "corner", "config", make_navbar_links(parameters)]
            html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                          html_page="{}_{}".format(app, i))
            html_file.make_header(title="{} Posterior PDF for {}".format(app, i), background_colour=col,
                                  approximant=app)
            html_file.make_navbar(links=links)
           
            if i != "multiple":
                html_file.insert_image("{}/plots/1d_posterior_{}_{}.png".format(opts.baseurl, app, i))
            else:
                html_file.make_search_bar(popular_options=["mass_1, mass_2",
                                                           "luminosity_distance, iota, ra, dec",
                                                           "iota, phi_12, phi_jl, tilt_1, tilt_2"],
                                          code="combines")
            html_file.make_footer(user=os.environ["USER"], rundir="{}".format(opts.webdir))
                

def make_comparison_pages(opts, approximants, samples, colors, parameters):
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
    parameters: list
        list of parameters that the sampler varies over
    """
    webpage.make_html(web_dir=opts.webdir, pages=["Comparison"])
    html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                   html_page="Comparison")
    html_file.make_header(title="Comparison Summary Page")
    links = ["home", ["Approximant", [i for i in approximants]+["Comparison"]],
                     make_navbar_links(parameters)]
    html_file.make_navbar(links=links)
    contents = [["{}/plots/combined_skymap.png".format(opts.baseurl),
                 "{}/plots/compare_waveforms.png".format(opts.baseurl)]]
    html_file.make_table_of_images(contents=contents)
    if opts.sensitivity:
        html_file.add_content("<div class='row justify-content-center' "
                              "style='margin=top: 2.0em;'>"
                              "<p>To see the sky sensitivity for the following "
                              "networks, click the button</p></div>")
        html_file.add_content("<div class='row justify-content-center' "
                              "style='margin-top: 0.2em;'>"
                              "<button type='button' class='btn btn-info' "
                              "onclick='%s.src=\"%s/plots/combined_skymap.png\"'"
                              "style='margin-left:0.25em; margin-right:0.25em'>Sky Map</button>"
                              "<button type='button' class='btn btn-info' "
                              "onclick='%s.src=\"%s/plots/IMRPhenomPv2_sky_sensitivity_HL.png\"'"
                              "style='margin-left:0.25em; margin-right:0.25em'>HL</button>"
                               %("combined_skymap", opts.baseurl, "combined_skymap", opts.baseurl))
        html_file.add_content("<button type='button' class='btn btn-info' "
                              "onclick='%s.src=\"%s/plots/IMRPhenomPv2_sky_sensitivity_HLV.png\"'"
                              "style='margin-left:0.25em; margin-right:0.25em'>HLV</button></div>\n"
                               %("combined_skymap", opts.baseurl))

    html_file.make_footer(user=os.environ["USER"], rundir=opts.webdir)
    # edit all of the comparison pages
    parameters.append("multiple")
    pages = ["Comparison_{}".format(i) for i in parameters]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for i in parameters:
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="Comparison_{}".format(i))
        html_file.make_header(title="Comparison page for {}".format(i),
                              approximant="Comparison")
        html_file.make_navbar(links=links)
        if i != "multiple":
            html_file.insert_image("{}/plots/combined_posterior_{}.png".format(opts.baseurl, i))
        else:
            html_file.make_search_bar(popular_options=["mass_1, mass_2",
                                                       "luminosity_distance, iota, ra, dec",
                                                       "iota, phi_12, phi_jl, tilt_1, tilt_2"],
                                      code="combines")
        html_file.make_footer(user=os.environ["USER"], rundir=opts.webdir)

def make_corner_pages(opts, approximants, samples, colors, parameters):
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
    parameters: list
        list of parameters that the sampler varies over
    """
    pages = ["{}_corner".format(i) for i in approximants]
    webpage.make_html(web_dir=opts.webdir, pages=pages)
    for app, col in zip(approximants, colors):
        if len(approximants) > 1:
            links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                     "corner", "config", make_navbar_links(parameters)]
        else:
            links = ["home", ["Approximants", [k for k in approximants]],
                     "corner", "config", make_navbar_links(parameters)]
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="{}_corner".format(app))
        html_file.make_header(title="{} Corner plots".format(app), background_colour=col,
                              approximant=app)
        html_file.make_navbar(links=links)
        html_file.make_search_bar(popular_options=["mass_1, mass_2",
                                                   "luminosity_distance, iota, ra, dec",
                                                   "iota, phi_12, phi_jl, tilt_1, tilt_2"])
        html_file.make_footer(user=os.environ["USER"], rundir="{}".format(opts.webdir))

def make_config_pages(opts, approximants, samples, colors, configs, parameters):
    """Make the config pages for all approximants

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
    configs: list
        list of paths to config files you wish to include
    parameters: list
        list of parameters that the sample varies over
    """
    pages = ["{}_config".format(i) for i in approximants]
    webpage.make_html(web_dir=opts.webdir, pages=pages, stylesheets=pages)
    for app, con, col in zip(approximants, configs, colors):
        if len(approximants) > 1:
            links = ["home", ["Approximants", [k for k in approximants+["Comparison"]]],
                     "corner", "config", make_navbar_links(parameters)]
        else:
            links = ["home", ["Approximants", [k for k in approximants]],
                     "corner", "config", make_navbar_links(parameters)]
        html_file = webpage.open_html(web_dir=opts.webdir, base_url=opts.baseurl,
                                      html_page="{}_config".format(app))
        html_file.make_header(title="{} configuration".format(app), background_colour=col,
                              approximant=app)
        html_file.make_navbar(links=links)
        with open(con, 'r') as f:
            contents = f.read()
        styles = html_file.make_code_block(language='ini', contents=contents)
        with open('{0:s}/css/{1:s}_config.css'.format(opts.webdir, app), 'w') as f:
            f.write(styles)
        html_file.make_footer(user=os.environ["USER"], rundir="{}".format(opts.webdir))

def write_html(opts, colors):
    """Generate an html page to show posterior plots

    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    colors: list
        list of colors in hexadecimal format for the different approximants 
    """
    # grab the parameters
    parameters = _grab_parameters(opts.samples[0])
    # make the webpages
    make_home_pages(opts, opts.approximant, opts.samples, colors, parameters)
    make_1d_histograms_pages(opts, opts.approximant, opts.samples, colors,
                             parameters)
    make_corner_pages(opts, opts.approximant, opts.samples, colors, parameters)
    make_config_pages(opts, opts.approximant, opts.samples, colors, opts.config,
                      parameters)
    if len(opts.approximant) != 1:
        make_comparison_pages(opts, opts.approximant, opts.samples, colors,
                              parameters)

def write_html_data_dump(opts, colors):
    """Generate a single html page to show posterior plots
a
    Parameters
    ----------
    opts: argparse
        argument parser object to hold all information from command line
    colors: list
        list of colors in hexadecimal format for the different approximants
    """
    # grab the parameters from the samples
    f = h5py.File(opts.samples[0])
    parameters = [i for i in f["parameters"]]                       
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
        html_file.make_footer(user=os.environ["USER"], rundir=opts.webdir)
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
        html_file.make_footer(user=os.environ["USER"], rundir=opts.webdir)

if __name__ == '__main__':
    # default colors
    colors = ["#a6b3d0", "#baa997", "#FF6347", "#FFA500", "#003366"]
    # get arguments from command line
    parser = command_line()
    opts = parser.parse_args()
    # convert to the standard results file format
    logging.info("Converting files to standard format")
    convert_to_standard_format(opts.samples)
    # check the inputs
    logging.info("Checking the inputs")
    run_checks(opts)
    # make the plots for the webpage
    logging.info("Generating the plots")
    make_plots(opts, colors=colors)
    # copy over the relevant files to the web directory
    logging.info("Copying the files to %s" %(opts.webdir))
    copy_files(opts)
    # check to see if dump option is parsed
    logging.info("Writing HTML pages")
    if opts.dump:
        write_html_data_dump(opts, colors=colors)
    else:
        write_html(opts, colors=colors)
    # send email
    if opts.email:
        try:
            email_notify(opts.email, opts.baseurl+"/home.html")
        except Exception as e:
            print("Unable to send notification email because of error: {}".format(e))
    logging.info("Removing the temporary file that is in standard format")
    for i in opts.samples:
        os.remove(i)
