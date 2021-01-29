#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.standard_names import standard_names, lalinference_map
from pesummary.utils.samples_dict import SamplesDict
from pesummary.io import read
from pesummary.core.webpage.main import _WebpageGeneration
from pesummary.utils.utils import logger
import subprocess
import numpy as np
import sys
import shutil
import os
from glob import glob
import pkg_resources

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


REVIEW_TESTS = [
    "all", "core_plots", "ligo.skymap", "spin_disk"
]


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--webdir", dest="webdir", help="make page and plots in DIR",
        metavar="DIR", default="./"
    )
    parser.add_argument(
        "-s", "--samples", dest="samples", help="Posterior samples hdf5 file",
        default=None
    )
    parser.add_argument(
        "-t", "--test", dest="test", help="Review test that you wish to perform",
        choices=REVIEW_TESTS, default="all"
    )
    parser.add_argument(
        "--multi_process", dest="multi_process", default=18,
        help="The number of cores to use when generating plots"
    )
    return parser


def get_executable_path(executable):
    """Return the path to the executable

    Parameters
    ----------
    executable: str
        name of the executable
    """
    try:
        path = subprocess.check_output(["which", "%s" % (executable)])
        path = path.decode("utf-8")
    except Exception:
        path = None
    return path


def make_cli(executable, default_arguments, add_symbol=False):
    """Generate a command line

    Parameters
    ----------
    executable: str
        executable you wish to run
    default_arguments: list
        list of arguments you wish to pass to the executable
    add_symbol: Bool, optional
        if True, prepend the cli with a '$' symbol
    """
    executable_path = get_executable_path(executable)
    if executable_path is None:
        raise Exception(
            "'{}' is not installed in your environment.".format(executable)
        )
    cli = executable + " " + " ".join(default_arguments)
    if add_symbol:
        cli = "$ {}".format(cli)
    return cli


def _launch_lalinference_cli(
    webdir, path_to_results_file, trigger_file=None, add_symbol=False
):
    """Command line to launch the lalinference job

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    trigger_file: str, optional
        path to an xml trigger file.
    add_symbol: Bool, optional
        if True, prepend the cli with a '$' symbol
    """
    webdir += "/lalinference"
    default_arguments = ["--skyres", "0.5", "--outpath", webdir,
                         path_to_results_file]

    if trigger_file is not None:
        default_arguments.append("--trig")
        default_arguments.append(trigger_file)

    cli = make_cli("cbcBayesPostProc", default_arguments, add_symbol=add_symbol)
    return [cli]


def launch_lalinference(webdir, path_to_results_file, trigger_file=None):
    """Launch a subprocess to generate the lalinference pages

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    trigger_file: str, optional
        path to an xml trigger file.
    """
    cli = _launch_lalinference_cli(
        webdir, path_to_results_file, trigger_file=trigger_file, add_symbol=False
    )
    logger.info("Running %s" % (cli))
    ess = subprocess.Popen(
        cli, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ess.wait()


def _launch_skymap_cli(
    webdir, path_to_results_file, add_symbol=False, multi_process=1
):
    """Command lines to launch to the skymap job

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    trigger_file: str, optional
        path to an xml trigger file.
    add_symbol: Bool, optional
        if True, prepend each cli with a '$' symbol
    multi_process: int, optional
        number of cpus to run on
    """
    cli = []
    default_arguments = [
        path_to_results_file, "-j {}".format(multi_process), "--enable-multiresolution",
        "--outdir", webdir
    ]
    cli.append(
        make_cli("ligo-skymap-from-samples", default_arguments, add_symbol=add_symbol)
    )
    default_arguments = [
        "{}/skymap.fits".format(webdir), "--annotate", "--contour 50 90",
        "-o {}/skymap.png".format(webdir)
    ]
    cli.append(
        make_cli("ligo-skymap-plot", default_arguments, add_symbol=add_symbol)
    )
    default_arguments = [
        "{}/skymap.fits".format(webdir), "-p 50 90",
        "-o {}/skymap.txt".format(webdir)
    ]
    cli.append(
        make_cli("ligo-skymap-stats", default_arguments, add_symbol=add_symbol)
    )
    return cli


def launch_skymap(webdir, path_to_results_file, multi_process=1):
    """Generate a skymap with the `ligo.skymap` package.

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    multi_process: int, optional
        number of cpus to run on
    """
    webdir += "/ligo_skymap"
    cli = _launch_skymap_cli(webdir, path_to_results_file, multi_process=multi_process)
    for _cli in cli:
        ess = subprocess.run(_cli, shell=True)


def _launch_pesummary_cli(
    webdir, path_to_results_file, trigger_file=None, add_symbol=False
):
    """Command lines to launch the pesummary job

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    trigger_file: str, optional
        path to an xml trigger file.
    multi_process: int, optional
        number of cpus to run on
    """
    executable = "summarypages"
    executable_path = get_executable_path(executable)

    if executable_path is None:
        raise Exception(
            "'summarypages' is not installed in your environment. failed "
            "to generate PESummary pages")

    webdir += "/pesummary"
    default_arguments = ["--webdir", webdir, "--samples", path_to_results_file,
                         "--approximant", "IMRPhenomPv2", "--gw",
                         "--labels", "pesummary", "--cosmology", "planck15_lal",
                         "--multi_process 18"]

    if trigger_file is not None:
        default_arguments.append("--trig_file")
        default_arguments.append(trigger_file)

    cli = executable + " " + " ".join(default_arguments)
    if add_symbol:
        cli = "$ {}".format(cli)
    return [cli]


def launch_pesummary(
    webdir, path_to_results_file, trigger_file=None, multi_process=1
):
    """Launch a subprocess to generate the PESummary pages

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    path_to_results_file: str
        path to the results file. Must be compatible with the LALInference
        pipeline
    trigger_file: str, optional
        path to an xml trigger file.
    multi_process: int, optional
        number of cpus to run on
    """
    cli = _launch_pesummary_cli(
        webdir, path_to_results_file, trigger_file=trigger_file
    )[0]
    executable = "summarypages"
    executable_path = get_executable_path(executable)

    if executable_path is None:
        raise Exception(
            "'summarypages' is not installed in your environment. failed "
            "to generate PESummary pages")

    webdir += "/pesummary"
    default_arguments = ["--webdir", webdir, "--samples", path_to_results_file,
                         "--approximant", "IMRPhenomPv2", "--gw",
                         "--labels", "pesummary", "--cosmology", "planck15_lal",
                         "--multi_process {}".format(multi_process)]

    if trigger_file is not None:
        default_arguments.append("--trig_file")
        default_arguments.append(trigger_file)

    cli = executable + " " + " ".join(default_arguments)
    logger.info("Running %s" % (cli))
    ess = subprocess.run(cli, shell=True)


def _launch_spin_disk_script():
    """Script to run which generates a spin disk using pesummary and a spin
    disk using the script used in GWTC1
    """
    script = """
    import h5py
    from pesummary.gw.fetch import fetch

    base = "https://ldas-jobs.ligo.caltech.edu/~charlie.hoy/projects/GWTC1/"
    f = fetch(
        base + "plot_spin_disks.py", read_file=False, delete_on_exit=True,
        outdir=pkg_resources.resource_filename("pesummary", "cli")
    )
    f = fetch(
        base + "bounded_2d_kde.py", read_file=False, delete_on_exit=True,
        outdir=pkg_resources.resource_filename("pesummary", "cli")
    )
    from .plot_spin_disks import plot_spindisk_with_colorbar
    path_to_pesummary_file = os.path.join(
        webdir, "pesummary", "samples", "posterior_samples.h5"
    )
    try:
        os.makedirs(os.path.join(webdir, "spin_disk"))
    except FileExistsError:
        pass
    f = read(path_to_pesummary_file)
    samples = f.samples_dict["pesummary"]
    fig = samples.plot(type="spin_disk", colorbar=True, cmap="Blues")
    fig.savefig("{}/spin_disk/pesummary.png".format(webdir))
    fig.close()
    f = h5py.File(path_to_results_file, 'r')
    keys = list(f["lalinference"].keys())
    samples = f["lalinference"][keys[0]]["posterior_samples"]
    if "costilt1" in samples.dtype.names:
        pass
    else:
        samples = SamplesDict({
            "a1": samples["a1"], "a2": samples["a2"],
            "costilt1": np.cos(samples["tilt1"]), "costilt2": np.cos(samples["tilt2"])
        }).to_structured_array()
    fig = plot_spindisk_with_colorbar(
        samples, None, "Blues", plot_event_label=False, Na=25, Nt=25,
        return_maxP_no_plot=False, threshold=True
    )
    fig.savefig("{}/spin_disk/GWTC1.png".format(webdir))
    f.close()
    """
    return "\n".join([line[4:] for line in script.split("\n")])


def launch_spin_disk(webdir, path_to_results_file):
    """Generate a spin disk plot using both pesummary and code from GWTC1
    """
    exec(_launch_spin_disk_script())


def get_list_of_plots(webdir, lalinference=False, pesummary=False):
    """Get a list of plots generated by either lalinference or pesummary

    Parameters
    ----------
    webdir: str
        path to the directory to store the output pages
    lalinference: Bool
        if True, return the plots generated by LALInference. Default False
    pesummary: Bool
        if True, return the plots generated by PESummary. Default False
    """
    from glob import glob

    if lalinference:
        histogram_plots = sorted(
            glob(webdir + "/lalinference/1Dpdf/*.png"))
        params = [
            i.split("/")[-1].split(".png")[0] for i in histogram_plots]
    elif pesummary:
        histogram_plots = sorted(
            glob(webdir + "/pesummary/plots/*_1d_posterior*.png"))
        params = [
            i.split("/")[-1].split("_1d_posterior_")[1].split(".png")[0] for i
            in histogram_plots]
    return histogram_plots, params


class WebpageGeneration(_WebpageGeneration):
    """Class to handle webpage generation displaying the outcome from the review

    Parameters
    ----------
    webdir: str
        the web directory of the run
    path_to_results_file: str
        the path to the lalinference h5 file
    *args: tuple
        all args passed to the _WebpageGeneration class
    **kwargs: dict
        all kwargs passed to the _WebpageGeneration class
    """
    def __init__(self, webdir, path_to_results_file, *args, test="all", **kwargs):
        self.path_to_results_file = os.path.abspath(path_to_results_file)
        self.test = test
        self.lalinference_path = os.path.abspath(os.path.join(
            webdir, "lalinference", "posterior_samples.dat"
        ))
        self.pesummary_path = os.path.abspath(os.path.join(
            webdir, "pesummary", "samples", "posterior_samples.h5"
        ))
        lalinference = np.genfromtxt(self.lalinference_path, names=True)
        pesummary_dict = read(self.pesummary_path).samples_dict["pesummary"]
        _lalinference_dict = {
            param: lalinference[param] for param in lalinference.dtype.names
        }
        params = _lalinference_dict.keys()
        lalinference_dict = {}
        for param, data in _lalinference_dict.items():
            if param in standard_names.keys():
                lalinference_dict[standard_names[param]] = data
            elif param in [
                    _param.lower() for _param in pesummary_dict.keys()
            ]:
                pp = [
                    _param for _param in pesummary_dict.keys() if param == _param.lower()
                ][0]
                lalinference_dict[pp] = data
            else:
                lalinference_dict[param] = data
        samples = {
            "lalinference": SamplesDict(lalinference_dict),
            "pesummary": read(
                os.path.join(webdir, "pesummary", "samples", "posterior_samples.h5")
            ).samples_dict["pesummary"]
        }
        super(WebpageGeneration, self).__init__(
            *args, webdir=webdir, labels=list(samples.keys()), samples=samples,
            user=os.environ["USER"], **kwargs
        )
        self.abswebdir = os.path.abspath(self.webdir)
        parameters = [list(self.samples[key].keys()) for key in self.samples.keys()]
        params = list(set.intersection(*[set(l) for l in parameters]))
        self.same_parameters = params
        self.copy_css_and_js_scripts()

    def copy_css_and_js_scripts(self):
        """Copy css and js scripts from the package to the web directory
        """
        files_to_copy = []
        path = pkg_resources.resource_filename("pesummary", "core")
        scripts = glob(os.path.join(path, "js", "*.js"))
        for i in scripts:
            files_to_copy.append(
                [i, os.path.join(self.webdir, "js", os.path.basename(i))]
            )
        scripts = glob(os.path.join(path, "css", "*.css"))
        for i in scripts:
            files_to_copy.append(
                [i, os.path.join(self.webdir, "css", os.path.basename(i))]
            )
        for _dir in ["js", "css"]:
            try:
                os.mkdir(os.path.join(self.webdir, _dir))
            except FileExistsError:
                pass
        for ff in files_to_copy:
            shutil.copy(ff[0], ff[1])

    def generate_webpages(self):
        """Generate all webpages for all result files passed
        """
        self.make_home_pages()
        if self.test == "core_plots" or self.test == "all":
            self.make_comparison_core_plots()
            self.make_comparison_samples()
        if self.test == "skymap" or self.test == "all":
            self.make_comparison_skymap()
        if self.test == "spin_disk" or self.test == "all":
            self.make_comparison_spin_disk()
        self.generate_specific_javascript()

    def make_navbar_for_comparison_page(self):
        """Make a navbar for the comparison homepage
        """
        links = ["home"]
        comparison = ["Comparison"]
        sub_comparison = []
        if self.test == "core_plots" or self.test == "all":
            sub_comparison.append("core_plots")
            sub_comparison.append("samples")
        if self.test == "skymap" or self.test == "all":
            sub_comparison.append("skymap")
        if self.test == "spin_disk" or self.test == "all":
            sub_comparison.append("spin_disk")
        comparison.append(sub_comparison)
        links.append(comparison)
        return links

    def _make_home_pages(self, pages):
        """Make the home pages

        Parameters
        ----------
        pages: list
            list of pages that you wish to create
        """
        from pesummary import __version__

        html_file = self.setup_page(
            "home", self.navbar["comparison"], title="Summary"
        )
        html_file.make_div(indent=2, _class='banner', _style="text-align: center")
        html_file.add_content("Review completed with:")
        html_file.end_div()
        html_file.make_div(
            indent=2, _class='banner', _style="text-align: center; color:red"
        )
        html_file.add_content("pesummary=={}".format(__version__))
        html_file.end_div()
        html_file.make_div(indent=2, _class='banner', _style="font-size: 14px")
        html_file.add_content(
            "This page compares the output between pesummary and lalinference/"
            "ligo.skymap."
        )
        html_file.end_div()
        html_file.make_banner(
            approximant="On the command-line", key="command_line",
            _style="font-size: 26px;", link=os.getcwd()
        )
        command = ""
        for i in sys.argv:
            command += " "
            if i[0] == "-":
                command += "\n"
            command += "{}".format(i)
        html_file.make_container()
        styles = html_file.make_code_block(language="shell", contents=command)
        with open('{0:s}/css/home.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        html_file.export(
            "pesummary.sh", csv=False, json=False, shell=True,
            margin_top="-4em"
        )
        html_file.end_div()

    def make_comparison_spin_disk(self):
        """Make a page which compares the spin disk produced using code in
        pesummary and the spin disk produced using code from the GWTC1 catalog
        """
        pages = ["spin_disk"]
        self.create_blank_html_pages(pages, stylesheets="spin_disk")
        html_file = self.setup_page(
            pages[0], self.navbar["comparison"], title="Spin disk comparison"
        )
        html_file.make_div(indent=2, _class='banner', _style=None)
        html_file.add_content("Spin disk comparison")
        html_file.end_div()
        html_file.make_div(indent=2, _class='paragraph', _style="max-width:1400px")
        html_file.add_content(
            "Below we compare the spin disk plot produced using code within pesummary "
            "to the spin disk produced using code from the GWTC1 catalog."
        )
        html_file.end_div()
        html_file.make_container()
        styles = html_file.make_code_block(
            language="python", contents=_launch_spin_disk_script()
        )
        with open('{0:s}/css/spin_disk.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        contents = [[
            "../spin_disk/pesummary.png", "../spin_disk/GWTC1.png"
        ]]
        html_file = self.make_modal_carousel(html_file, contents)

    def make_comparison_core_plots(self):
        """Make a page which compares the core 1d_histogram, autocorrelation
        and sample trace plots in pesummary to the ones produced with
        `cbcBayesPostProc`
        """
        pages = ["core_plots"]
        self.create_blank_html_pages(pages)
        html_file = self.setup_page(
            pages[0], self.navbar["comparison"], title="Core plot comparison"
        )
        html_file.make_div(indent=2, _class='banner', _style=None)
        html_file.add_content("Core plot comparison")
        html_file.end_div()
        html_file.make_div(indent=2, _class='paragraph', _style="max-width:1400px")
        html_file.add_content(
            "Below we compare the 1d histograms, autocorrelation and sample trace "
            "plots produced by pesummary and `cbcBayesPostProc`."
        )
        html_file.end_div()
        cli = "\n".join(
            _launch_lalinference_cli(
                self.abswebdir, self.path_to_results_file, add_symbol=True
            )
        )
        cli += "\n"
        cli += "\n".join(
            _launch_pesummary_cli(
                self.abswebdir, self.path_to_results_file, add_symbol=True
            )
        )
        html_file.make_container()
        styles = html_file.make_code_block(language="shell", contents=cli)
        with open('{0:s}/css/Comparison_core_plots.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        _reverse_dict = {val: key for key, val in lalinference_map.items()}
        _, lal_params = get_list_of_plots(self.webdir, lalinference=True)
        not_included = []
        _lalinf_base = "../lalinference"
        _pes_base = "../pesummary"
        for param in lal_params:
            if param in standard_names.keys():
                html_file.make_banner(
                    approximant=param, _style="font-size: 26px;"
                )
                contents = [[
                    "{}/1Dpdf/{}.png".format(_lalinf_base, param),
                    "{}/plots/pesummary_1d_posterior_{}.png".format(
                        _pes_base, standard_names[param]
                    )
                ], [
                    "{}/1Dsamps/{}_acf.png".format(_lalinf_base, param),
                    "{}/plots/pesummary_autocorrelation_{}.png".format(
                        _pes_base, standard_names[param]
                    )
                ], [
                    "{}/1Dsamps/{}_samps.png".format(_lalinf_base, param),
                    "{}/plots/pesummary_sample_evolution_{}.png".format(
                        _pes_base, standard_names[param]
                    )
                ]]
                html_file.make_table_of_images(contents=contents, autoscale=True)
                try:
                    _lalinf_samples = self.samples["lalinference"][standard_names[param]]
                    _pes_samples = self.samples["pesummary"][standard_names[param]]
                    html_file.make_div(
                        _class="banner", _style="font-size:26px; color:red"
                    )
                    html_file.add_content(
                        "max difference in %s samples: %s" % (
                            standard_names[param], np.max(
                                np.abs(_lalinf_samples - _pes_samples)
                            )
                        )
                    )
                    html_file.end_div()
                except Exception as e:
                    print(e)
            else:
                not_included.append(param)
        html_file.make_div(
            indent=2, _class='banner',
            _style="font-size: 14px; max-width:1400px; margin-bottom: 2em"
        )
        html_file.add_content(
            "parameters not included: {}".format(", ".join(not_included))
        )
        html_file.end_div()
        html_file.make_footer()

    def make_comparison_skymap(self):
        """Make a page which compares the skymap produced with `ligo.skymap`
        and pesummary.
        """
        pages = ["skymap"]
        self.create_blank_html_pages(pages)
        html_file = self.setup_page(
            pages[0], self.navbar["comparison"], title="Skymap comparison",
        )
        html_file.make_div(indent=2, _class='banner', _style=None)
        html_file.add_content("Skymap comparison")
        html_file.end_div()
        html_file.make_div(indent=2, _class='paragraph', _style="max-width:1400px")
        html_file.add_content(
            "Below we compare the skymap produced by ligo.skymap and the skymap "
            "produced by pesummary (wrapper for ligo.skymap). We compare the "
            "skymap plots and skymap stats."
        )
        html_file.end_div()
        _webdir = self.abswebdir + "/ligo_skymap"
        cli = "\n".join(
            _launch_skymap_cli(_webdir, self.path_to_results_file, add_symbol=True)
        )
        cli += "\n\n"
        cli += "\n".join(
            _launch_pesummary_cli(_webdir, self.path_to_results_file, add_symbol=True)
        )
        html_file.make_container()
        styles = html_file.make_code_block(language="shell", contents=cli)
        with open('{0:s}/css/Comparison_skymap.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        lal_skymap = glob(
            os.path.join(self.webdir, "ligo_skymap", "skymap.png")
        )[0].replace(self.webdir, "")
        pes_skymap = glob(
            os.path.join(self.webdir, "pesummary", "plots", "pesummary_skymap.png")
        )[0].replace(self.webdir, "")
        contents = [
            ["../{}".format(lal_skymap)], ["../{}".format(pes_skymap)]
        ]
        html_file = self.make_modal_carousel(html_file, contents)
        html_file.make_div(indent=2, _class='banner', _style="font-size: 26px")
        html_file.add_content("Stat comparison")
        html_file.end_div()
        lal_stats = np.genfromtxt(
            os.path.join(self.webdir, "ligo_skymap", "skymap.txt"), names=True,
            skip_header=True
        )
        pes_stats = np.genfromtxt(
            os.path.join(
                self.webdir, "pesummary", "samples", "pesummary_skymap_stats.dat"
            ), names=True, skip_header=True
        )
        keys = lal_stats.dtype.names
        headings = [" ", "ligo.skymap", "pesummary", "difference (%)"]
        contents = []
        for key in keys:
            row = [key, lal_stats[key], pes_stats[key]]
            row += [np.abs(lal_stats[key] - pes_stats[key]) * 100]
            contents.append(row)
        html_file.make_table(
            headings=headings, contents=contents, heading_span=1,
            accordian=False, format="table-hover header-fixed",
            sticky_header=True
        )
        html_file.make_div(_style="margin-bottom: 2em")
        html_file.end_div()
        html_file.make_footer()

    def make_comparison_samples(self):
        """Make a page which compares the samples produced by `cbcBayesPostProc`
        and pesummary.
        """
        pages = ["samples"]
        self.create_blank_html_pages(pages)
        html_file = self.setup_page(
            pages[0], self.navbar["comparison"], title="Sample by sample comparison",
        )
        html_file.make_div(indent=2, _class='banner', _style=None)
        html_file.add_content("Sample comparison")
        html_file.end_div()
        html_file.make_div(indent=2, _class='paragraph', _style="max-width:1400px")
        html_file.add_content(
            "Below we compare the samples stored in the lalinference "
            "posterior_samples.dat (produced with `cbcBayesPostProc`) and the "
            "samples stored in the `posterior_samples.h5` file (produced with "
            "`pesummary`). We compare the medians, 90% confidence intervals."
        )
        html_file.end_div()
        cli = "\n".join(
            _launch_lalinference_cli(
                self.abswebdir, self.path_to_results_file, add_symbol=True
            )
        )
        cli += "\n"
        cli += "\n".join(
            _launch_pesummary_cli(
                self.abswebdir, self.path_to_results_file, add_symbol=True
            )
        )
        cli += "\n\n"
        cli += "Files compared: \nlalinference: {}\npesummary: {}".format(
            self.lalinference_path, self.pesummary_path
        )
        html_file.make_container()
        styles = html_file.make_code_block(language="shell", contents=cli)
        with open('{0:s}/css/Comparison_samples.css'.format(self.webdir), 'w') as g:
            g.write(styles)
        html_file.end_container()
        headings = [" ", "lalinference", "pesummary", "difference (%)"]
        contents = []
        for _type in ["median", "5th percentile", "95th percentile"]:
            if _type == "median":
                func = np.median
            elif _type == "5th percentile":
                func = lambda array: np.percentile(array, 5)
            elif _type == "95th percentile":
                func = lambda array: np.percentile(array, 95)
            for param in self.same_parameters:
                row = []
                lal = func(self.samples["lalinference"][param])
                pes = func(self.samples["pesummary"][param])
                row = [
                    param, lal, pes, np.abs(pes - lal) * 100
                ]
                contents.append(row)
            html_file.make_div(
                indent=2, _class='banner', _style="font-size: 26px; margin-top: 1em;"
            )
            html_file.add_content(_type)
            html_file.end_div()
            html_file.make_table(
                headings=headings, contents=contents, heading_span=1,
                accordian=False, format="table-hover header-fixed",
                sticky_header=True
            )
        for analysis in ["lalinference", "pesummary"]:
            html_file.make_div(
                indent=2, _class='banner',
                _style="font-size: 14px; max-width:1400px; margin-bottom: 2em"
            )
            html_file.add_content(
                "{} parameters not included: {}".format(
                    analysis, ", ".join(
                        [
                            param for param in self.samples[analysis].keys()
                            if param not in self.same_parameters
                        ]
                    )
                )
            )
            html_file.end_div()
        html_file.make_footer()


def main(args=None):
    """Top level interface for `summaryreview`
    """
    parser = command_line()
    opts = parser.parse_args(args=args)
    if opts.test == "core_plots" or opts.test == "all":
        logger.info("Starting to generate plots using 'cbcBayesPostProc'")
        launch_lalinference(opts.webdir, opts.samples)
        logger.info("Starting to generate plots using 'summarypages'")
        launch_pesummary(opts.webdir, opts.samples, multi_process=opts.multi_process)
    if opts.test == "skymap" or opts.test == "all":
        logger.info("Generating skymap with `ligo.skymap`")
        launch_skymap(opts.webdir, opts.samples, multi_process=opts.multi_process)
    if opts.test == "spin_disk" or opts.test == "all":
        logger.info("Starting to make spin disk plots")
        launch_spin_disk(opts.webdir, opts.samples)
    logger.info("Making webpages to display tests")
    try:
        w = WebpageGeneration(opts.webdir, opts.samples, test=opts.test)
        w.generate_webpages()
    except Exception as e:
        logger.warn("Unable to generate webpages because {}".format(e))


if __name__ == "__main__":
    main()
