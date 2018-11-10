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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import webpage

__doc__ == "Parameters to run post_processing.py from the command line"


def command_line():
    """Creates an ArgumentParser object which holds all of the information
    from the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--outpath", dest="outpath",
                        help="make page and plots in DIR", metavar="DIR")
    parser.add_argument("-i", "--inj", dest="injfile",
                        help="SimInsipral injection file", metavar="INJ.XML",
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

    # generate the url of the posplots.html page
    if 'caltech' in host or 'cit' in host:
        url = "https://ldas-jobs.ligo.caltech.edu/"
    elif 'raven' in host:
        url = "https://geo2.arcca.cf.ac.uk/"
    else:
        url = "https://{}/".format(host)
    url += path
    message = "Hi {},\n\nYour output page is ready on {}. You can view the result at {}\n".format(user, host, url)
    cmd = 'echo -e "%s" | mail -s "%s" "%s"' %(message, subject, address)
    ess = subprocess.Popen(cmd, shell=True)
    ess.wait()


def write_html():
    """Generate an html page to show posterior plots
    """
    # make the webpages
    webpage.make_html(web_dir = "/home/c1737564/public_html/LVC/projects/bilby",
                      pages=["corner", "IMRPhenomPv2", "SEOBNRv2", "IMRPhenommass1", "SEOBNRmass1", "home"])
    # edit the home page
    html_file = webpage.open_html(web_dir = "/home/c1737564/public_html/LVC/projects/bilby",
                                  base_url = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby",
                                  html_page="home")
    html_file.make_header()
    html_file.make_navbar(links=[["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]]])
    # edit the home page for IMRPhenomPv2
    html_file = webpage.open_html(web_dir = "/home/c1737564/public_html/LVC/projects/bilby",
                                  base_url = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby",
                                  html_page="IMRPhenomPv2")
    html_file.make_header(title="IMRPhenomPv2 Summary Page", background_colour="#8c6278")
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                 "corner", ["1d_histograms", ["IMRPhenommass1"]]])
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=[["/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png",
                                              "/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png",
                                              "/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png"]])
    html_file.make_footer(user="c1737564", rundir="./")
    # edit the home page for SEOBNRv3
    html_file = webpage.open_html(web_dir = "/home/c1737564/public_html/LVC/projects/bilby",
                                  base_url = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby",
                                  html_page="SEOBNRv3")
    html_file.make_header(title="SEOBNRv3 Summary Page", background_colour="#228B22") 
    html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                 "corner", ["1d_histograms", ["SEOBNRmass1"]]])
    html_file.make_table_of_images(headings=["sky_map", "waveform", "psd"],
                                   contents=[["/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png",
                                              "/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png",
                                              "/home/c1737564/public_html/LVC/projects/bilby/GW150914/plots/GW150914_H1L1_dynesty_mass_1.png"]])
    html_file.make_footer(user="c1737564", rundir="./")
    # edit the mass1 page for both approximants
    for i,j in zip(["IMRPhenommass1", "SEOBNRmass1"], ["#8c6278", "#228B22"]):    
        html_file = webpage.open_html(web_dir="/home/c1737564/public_html/LVC/projects/bilby",
                                      base_url = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby",
                                      html_page=i)
        html_file.make_header(title="Posterior PDF for mass1", background_colour=j)
        html_file.make_navbar(links=["home", ["Approximant", ["IMRPhenomPv2", "SEOBNRv3", "Comparison"]],
                                     "corner", ["1d_histograms", [i]]])
        html_file.make_footer(user="c1737564", rundir="./")

if __name__ == '__main__':
    # get arguments from command line
    parser = command_line()
    opts = parser.parse_args()
    write_html()
    if opts.email:
        try:
            email_notify(opts.email, opts.outpath)
        except Exception as e:
            print("Unable to send notification email because of error: {}".format(e))
