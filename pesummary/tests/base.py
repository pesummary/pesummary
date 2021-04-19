# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pathlib import Path
from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from pesummary.core.inputs import Input

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def namespace(args):
    """Generate a namespace for testing purposes

    Parameters
    ----------
    args: dict
        dictionary of arguments
    """
    base = Namespace()
    for key in list(args.keys()):
        setattr(base, key, args[key])
    return base


def gw_parameters():
    parameters = [
        'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl',
        'phi_12', 'psi', 'theta_jn', 'ra', 'dec', 'luminosity_distance',
        'geocent_time', 'log_likelihood', 'mass_ratio', 'total_mass',
        'chirp_mass', 'symmetric_mass_ratio', 'iota', 'spin_1x', 'spin_1y',
        'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'chi_p', 'chi_eff',
        'cos_tilt_1', 'cos_tilt_2', 'redshift', 'comoving_distance',
        'mass_1_source', 'mass_2_source', 'total_mass_source',
        'chirp_mass_source', 'phi_1', 'phi_2', 'cos_theta_jn', 'cos_iota',
        'peak_luminosity_non_evolved', 'final_spin_non_evolved',
        'final_mass_non_evolved', 'final_mass_source_non_evolved',
        'radiated_energy_non_evolved', 'inverted_mass_ratio',
        'viewing_angle', 'chi_p_2spin'
    ]
    return parameters


def get_list_of_files(gw=False, number=1, existing_plot=False):
    """Return a list of files that should be generated from a typical workflow
    """
    if not gw:
        import string

        parameters = list(string.ascii_lowercase)[:17] + ["log_likelihood"]
        label = "core"
    else:
        parameters = gw_parameters()
        label = "gw"
    html = [
        "./.outdir/html/error.html",
        "./.outdir/html/Version.html",
        "./.outdir/html/Logging.html",
        "./.outdir/html/About.html",
        "./.outdir/html/Downloads.html"]
    if gw:
        sections = [
            "spins", "spin_angles", "timings", "source", "remnant", "others",
            "masses", "location", "inclination", "energy"
        ]
    else:
        sections = ["A-D", "E-F", "I-L", "M-P", "Q-T"]
    for num in range(number):
        html.append("./.outdir/html/%s%s_%s%s.html" % (label, num, label, num))
        if gw:
            html.append("./.outdir/html/%s%s_%s%s_Classification.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_Corner.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_Config.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_Custom.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_All.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_Interactive_Corner.html" % (
            label, num, label, num
        ))
        for section in sections:
            html.append("./.outdir/html/%s%s_%s%s_%s_all.html" % (label, num, label, num, section))
        for j in parameters:
            html.append("./.outdir/html/%s%s_%s%s_%s.html" % (label, num, label, num, j))
        if existing_plot:
            html.append("./.outdir/html/%s%s_%s%s_Additional.html" % (label, num, label, num))

    if number > 1:
        html.append("./.outdir/html/Comparison.html")
        html.append("./.outdir/html/Comparison_Custom.html")
        html.append("./.outdir/html/Comparison_All.html")
        html.append("./.outdir/html/Comparison_Interactive_Ridgeline.html")
        for j in parameters:
            if j != "classification":
                html.append("./.outdir/html/Comparison_%s.html" % (j))
        for section in sections:
            html.append("./.outdir/html/Comparison_%s_all.html" % (section))
    return sorted(html)


def get_list_of_plots(
    gw=False, number=1, mcmc=False, label=None, outdir=".outdir",
    comparison=True, psd=False, calibration=False, existing_plot=False,
    expert=False
):
    """Return a list of plots that should be generated from a typical workflow
    """
    if not gw:
        import string

        parameters = list(string.ascii_lowercase)[:17] + ["log_likelihood"]
        if label is None:
            label = "core"
    else:
        parameters = gw_parameters()
        if label is None:
            label = "gw"

    plots = []
    for num in range(number):
        for i in ["sample_evolution", "autocorrelation", "1d_posterior", "cdf"]:
            for j in parameters:
                plots.append("./%s/plots/%s%s_%s_%s.png" % (outdir, label, num, i, j))
        if mcmc:
            for j in parameters:
                plots.append("./%s/plots/%s%s_1d_posterior_%s_combined.png" % (outdir, label, num, j))
        if psd:
            plots.append("./%s/plots/%s%s_psd_plot.png" % (outdir, label, num))
        if calibration:
            plots.append("./%s/plots/%s%s_calibration_plot.png" % (outdir, label, num))
        if existing_plot:
            plots.append("./%s/plots/test.png" % (outdir))
        if expert:
            for j in parameters:
                if j != "log_likelihood":
                    plots.append("./%s/plots/%s%s_2d_contour_%s_log_likelihood.png" % (outdir, label, num, j))
                plots.append("./%s/plots/%s%s_1d_posterior_%s_bootstrap.png" % (outdir, label, num, j))
                plots.append("./%s/plots/%s%s_sample_evolution_%s_log_likelihood_colored.png" % (outdir, label, num, j))
    if number > 1 and comparison:
        for i in ["1d_posterior", "boxplot", "cdf"]:
            for j in parameters:
                plots.append("./%s/plots/combined_%s_%s.png" % (outdir, i, j))

    if gw:
        for num in range(number):
            plots.append("./%s/plots/%s%s_skymap.png" % (outdir, label, num))
            plots.append("./%s/plots/%s%s_default_pepredicates.png" % (outdir, label, num))
            plots.append("./%s/plots/%s%s_default_pepredicates_bar.png" % (outdir, label, num))
            plots.append("./%s/plots/%s%s_population_pepredicates.png" % (outdir, label, num))
            plots.append("./%s/plots/%s%s_population_pepredicates_bar.png" % (outdir, label, num))
        if number > 1 and comparison:
            plots.append("./%s/plots/combined_skymap.png" % (outdir))
        
    return sorted(plots)


def make_argparse(gw=True, extension="json", bilby=False, lalinference=False,
                  number=1, existing=False, disable_expert=True):
    """
    """
    parser = command_line()
    default_args = []
    if gw:
        insert_gwspecific_option_group(parser)
        default_args.append("--gw")
        default_args.append("--nsamples_for_skymap")
        default_args.append("10")
    params, data = make_result_file(
        extension=extension, gw=gw, bilby=bilby, lalinference=lalinference)
    if not existing:
        default_args.append("--webdir")
    else:
        default_args.append("--existing_webdir")
    default_args.append(".outdir")
    default_args.append("--samples")
    for i in range(number):
        default_args.append("./.outdir/test.%s" % (extension))
    default_args.append("--labels")
    if not existing:
        for i in range(number):
            if not gw:
                default_args.append("core%s" % (i))
            else:
                default_args.append("gw%s" % (i))
    else:
        if not gw:
            default_args.append("core1")
        else:
            default_args.append("gw1")
    default_args.append("--config")
    for i in range(number):
        default_args.append(testing_dir + "/example_config.ini")
    if disable_expert:
        default_args.append("--disable_expert")
    opts = parser.parse_args(default_args)
    if gw:
        func = GWInput
    else:
        func = Input
    return opts, func(opts)


def read_result_file(outdir="./.outdir", extension="json", bilby=False,
                     lalinference=False, pesummary=False):
    """
    """
    if bilby:
        from bilby.core.result import read_in_result

        if extension == "json":
            f = read_in_result(outdir + "/test.json")
        elif extension == "h5" or extension == "hdf5":
            f = read_in_result(outdir + "/test.h5")
        posterior = f.posterior
        posterior = posterior.select_dtypes(include=[float, int])
        samples = {key: val for key, val in posterior.items()}
    elif lalinference:
        import h5py

        f = h5py.File(outdir + "/test.hdf5", "r")

        posterior = f["lalinference"]["lalinference_nest"]["posterior_samples"]
        samples = {
            i: [j[num] for j in posterior] for num, i in enumerate(
                posterior.dtype.names
            )
        }
    elif pesummary:
        from pesummary.gw.file.read import read

        if extension == "json":
            f = read(outdir + "/test.json")
        else:
            f = read(outdir + "/test.h5")
        data = f.samples_dict
        labels = f.labels
        samples = data[labels[0]]
    elif extension == "dat":
        import numpy as np

        data = np.genfromtxt(outdir + "/test.dat", names=True)
        samples = {
            i: [j[num] for j in data] for num, i in enumerate(data.dtype.names)
        }
    else:
        samples = {}
    return samples


def make_psd(outdir="./.outdir"):
    """Make a psd file
    """
    frequencies = np.linspace(0, 1024, 1000)
    strains = np.random.uniform(10, 0.1, 1000)
    data = np.vstack([frequencies, strains]).T
    np.savetxt(
        "{}/psd.dat".format(outdir), data, delimiter="\t",
        header="\t".join(["frequencies", "strain"])
    )


def make_calibration(outdir="./.outdir"):
    """Make a calibration file
    """
    frequencies = np.linspace(0, 1024, 1000)
    columns = [np.random.uniform(10, 0.1, 1000) for _ in range(6)]
    data = np.vstack([frequencies] + columns).T
    np.savetxt(
        "{}/calibration.dat".format(outdir), data, delimiter="\t",
        header="\t".join(["frequencies", "a", "b", "c", "d", "e", "f"])
    )


def make_injection_file(
    outdir="./.outdir", extension="json", return_filename=True,
    return_injection_dict=True
):
    """
    """
    import os
    from pesummary.io import write

    filename = os.path.join(outdir, "injection.{}".format(extension))
    parameters = gw_parameters()
    samples = np.array([[np.random.random()] for i in range(len(parameters))]).T
    write(parameters, samples, filename=filename, file_format=extension)
    args = []
    if return_filename:
        args.append(filename)
    if return_injection_dict:
        args.append({param: samples[0][num] for num, param in enumerate(parameters)})
    return args


def make_result_file(outdir="./.outdir/", extension="json", gw=True, bilby=False,
                     lalinference=False, pesummary=False, pesummary_label="label",
                     config=None, psd=None, calibration=None, random_seed=None,
                     n_samples=1000):
    """Make a result file that can be read in by PESummary

    Parameters
    ----------
    outdir: str
        directory where you would like to store the result file
    extension: str
        the file extension of the result file
    gw: Bool
        if True, gw parameters will be used
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    print(extension, gw, bilby, lalinference, pesummary)
    data = np.array([np.random.random(18) for i in range(n_samples)])
    if gw:
        parameters = ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2",
                      "phi_jl", "phi_12", "psi", "theta_jn", "ra", "dec",
                      "luminosity_distance", "geocent_time", "redshift",
                      "mass_1_source", "mass_2_source", "log_likelihood"]
        distance = np.random.random(n_samples) * 500
        mass_1 = np.random.random(n_samples) * 100
        q = np.random.random(n_samples) * 100
        a_1 = np.random.uniform(0, 0.99, n_samples)
        a_2 = np.random.uniform(0, 0.99, n_samples)
        for num, i in enumerate(data):
            data[num][12] = distance[num]
            data[num][0] = mass_1[num]
            data[num][1] = mass_1[num] * q[num]
            data[num][2] = a_1[num]
            data[num][3] = a_2[num]
    else:
        import string

        parameters = list(string.ascii_lowercase)[:17] + ["log_likelihood"]
    if extension == "dat":
        np.savetxt(outdir + "test.dat", data, delimiter=" ",
                   header=" ".join(parameters), comments="")
    elif extension == "csv":
        np.savetxt(outdir + "test.csv", data, delimiter=",",
                   header=",".join(parameters), comments="")
    elif extension == "npy":
        from pesummary.utils.samples_dict import SamplesDict
        samples = SamplesDict(parameters, np.array(data).T).to_structured_array()
        np.save(outdir + "test.npy", samples)
    elif extension == "json" and not bilby and not pesummary and not lalinference:
        import json

        dictionary = {"NameOfCode": {"posterior_samples": {key:
                      [i[num] for i in data] for num, key in
                      enumerate(parameters)}}}
        with open(outdir + "test.json", "w") as f:
            json.dump(dictionary, f, indent=4, sort_keys=True)
    elif (extension == "hdf5" or extension == "h5") and not bilby and not pesummary and not lalinference:
            import h5py

            h5py_data = np.array(
                [tuple(i) for i in data], dtype=[tuple([i, 'float64']) for i in parameters])
            f = h5py.File(outdir + "test.h5", "w")
            name = f.create_group("NameOfCode")
            samples = f.create_dataset("posterior_samples", data=h5py_data)
            f.close()
    elif bilby and not pesummary and not lalinference:
        import bilby
        from bilby.core.result import Result
        from bilby.core.prior import PriorDict
        from pandas import DataFrame

        priors = PriorDict()
        priors.update({"%s" % (i): bilby.core.prior.Uniform(0.1, 0.5, 0) for i in parameters})
        posterior_data_frame = DataFrame(data, columns=parameters)
        injection_parameters = {par: 1. for par in parameters}
        bilby_object = Result(
            search_parameter_keys=parameters, samples=data,
            posterior=posterior_data_frame, label="test",
            injection_parameters=injection_parameters,
            priors=priors,
            log_bayes_factor=0.5, log_evidence_err=0.1, log_noise_evidence=0.1,
            log_evidence=0.2, version=["bilby=0.5.3:"],
            meta_data={"likelihood": {"time_marginalization": "True"}})
        if extension == "json":
            bilby_object.save_to_file(
                filename=outdir + "test.json", extension="json")
        elif extension == "hdf5" or extension == "h5":
            bilby_object.save_to_file(
                filename=outdir + "test.h5", extension="hdf5")
    elif lalinference and not bilby and not pesummary:
        import h5py

        h5py_data = np.array(
            [tuple(i) for i in data], dtype=[tuple([i, 'float64']) for i in parameters])
        f = h5py.File(outdir + "test.hdf5", "w")
        lalinference = f.create_group("lalinference")
        nest = lalinference.create_group("lalinference_nest")
        samples = nest.create_dataset("posterior_samples", data=h5py_data)
        f.close()
    elif pesummary and not lalinference and not bilby:
        dictionary = {
            pesummary_label: {
                "posterior_samples": {
                    "parameter_names": parameters,
                    "samples": [list(i) for i in data]
                },
                "injection_data": {
                    "injection_values": [float("nan") for i in range(len(parameters))]
                },
                "version": ["No version information found"],
                "meta_data": {
                    "sampler": {"log_evidence": 0.5},
                    "meta_data": {}
                }
            },
            "version": {
                "pesummary": ["v0.1.7"]
            }
        }
        if config is not None:
            dictionary[pesummary_label]["config_file"] = config
        if psd is not None:
            dictionary[pesummary_label]["psds"] = psds
        if calibration is not None:
            dictionary[pesummary_label]["calibration_envelope"] = calibration

        if extension == "json":
            import json

            with open(outdir + "test.json", "w") as f:
                json.dump(dictionary, f, indent=4, sort_keys=True)
        elif extension == "hdf5" or extension == "h5":
            import h5py
            from pesummary.core.file.meta_file import recursively_save_dictionary_to_hdf5_file

            f = h5py.File(outdir + "test.h5", "w")
            recursively_save_dictionary_to_hdf5_file(
                f, dictionary, extra_keys=list(dictionary.keys())
            )
            f.close()
    return parameters, data


testing_dir = str(Path(__file__).parent)
data_dir = str(Path(__file__).parent / "files")
