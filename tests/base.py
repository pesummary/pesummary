import numpy as np
import os
from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from pesummary.core.inputs import Input

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


def get_list_of_files(gw=False, number=1):
    """Return a list of files that should be generated from a typical workflow
    """
    if not gw:
        import string

        parameters = list(string.ascii_lowercase)[:14] + ["log_likelihood"]
        label = "core"
    else:
        parameters = [
            'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl',
            'phi_12', 'psi', 'theta_jn', 'ra', 'dec', 'luminosity_distance',
            'geocent_time', 'log_likelihood', 'mass_ratio', 'total_mass',
            'chirp_mass', 'symmetric_mass_ratio', 'iota', 'spin_1x', 'spin_1y',
            'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'chi_p', 'chi_eff',
            'cos_tilt_1', 'cos_tilt_2', 'redshift', 'comoving_distance',
            'mass_1_source', 'mass_2_source', 'total_mass_source',
            'chirp_mass_source', 'phi_1', 'phi_2', 'cos_theta_jn', 'cos_iota']
        label = "gw"
    html = [
        "./.outdir/html/error.html",
        "./.outdir/html/version.html",
        "./.outdir/html/logging.html"]
    for num in range(number):
        html.append("./.outdir/html/%s%s_%s%s.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_corner.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_config.html" % (label, num, label, num))
        html.append("./.outdir/html/%s%s_%s%s_multiple.html" % (label, num, label, num))
        for j in parameters:
            html.append("./.outdir/html/%s%s_%s%s_%s.html" % (label, num, label, num, j))

    if number > 1:
        html.append("./.outdir/html/Comparison.html")
        html.append("./.outdir/html/Comparison_multiple.html")
        for j in parameters:
            if j != "classification":
                html.append("./.outdir/html/Comparison_%s.html" % (j))
    return sorted(html)


def get_list_of_plots(gw=False, number=1):
    """Return a list of plots that should be generated from a typical workflow
    """
    if not gw:
        import string

        parameters = list(string.ascii_lowercase)[:14] + ["log_likelihood"]
        label = "core"
    else:
        parameters = [
            'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl',
            'phi_12', 'psi', 'theta_jn', 'ra', 'dec', 'luminosity_distance',
            'geocent_time', 'log_likelihood', 'mass_ratio', 'total_mass',
            'chirp_mass', 'symmetric_mass_ratio', 'iota', 'spin_1x', 'spin_1y',
            'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'chi_p', 'chi_eff',
            'cos_tilt_1', 'cos_tilt_2', 'redshift', 'comoving_distance',
            'mass_1_source', 'mass_2_source', 'total_mass_source',
            'chirp_mass_source', 'phi_1', 'phi_2', 'cos_theta_jn', 'cos_iota']
        label = "gw"

    plots = []
    for num in range(number):
        for i in ["sample_evolution", "autocorrelation", "1d_posterior", "cdf"]:
            for j in parameters:
                plots.append("./.outdir/plots/%s%s_%s_%s.png" % (label, num, i, j))
    if number > 1:
        for i in ["1d_posterior", "boxplot", "cdf"]:
            for j in parameters:
                plots.append("./.outdir/plots/combined_%s_%s.png" % (i, j))

    if gw:
        for num in range(number):
            plots.append("./.outdir/plots/gw%s_skymap.png" % (num))
        if number > 1:
            plots.append("./.outdir/plots/combined_skymap.png")
        
    return sorted(plots)


def make_argparse(gw=True, extension="json", bilby=False, lalinference=False,
                  number=1, existing=False):
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
        default_args.append("./tests/example_config.ini")
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


def make_result_file(outdir="./.outdir/", extension="json", gw=True, bilby=False,
                     lalinference=False, pesummary=False):
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
    print(extension, gw, bilby, lalinference, pesummary)
    data = np.array([np.random.random(15) for i in range(1000)])
    if gw:
        parameters = ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2",
                      "phi_jl", "phi_12", "psi", "theta_jn", "ra", "dec",
                      "luminosity_distance", "geocent_time", "log_likelihood"]
        distance = np.random.random(1000) * 500
        for num, i in enumerate(data):
            data[num][12] = distance[num]
        mass_1 = np.random.random(1000) * 100
        for num, i in enumerate(data):
            data[num][0] = mass_1[num]
        for num, i in enumerate(data):
            data[num][1] = mass_1[num]
    else:
        import string

        parameters = list(string.ascii_lowercase)[:14] + ["log_likelihood"]
    if extension == "dat":
            np.savetxt(outdir + "test.dat", data, delimiter=" ",
                       header=" ".join(parameters), comments="")
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
            version="bilby=0.5.3:", priors=priors,
            log_bayes_factor=0.5, log_evidence_err=0.1, log_noise_evidence=0.1,
            log_evidence=0.2,
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
            "posterior_samples":
                {"label": 
                    {"parameter_names": parameters,
                     "samples": [list(i) for i in data]
                    }
                },
            "injection_data":
                {"label":
                    {"injection_values": [float("nan") for i in range(len(parameters))]
                    }
                },
            "version":
                {"label": ["No version information found"],
                 "pesummary": ["v0.1.7"]
                },
            "meta_data":
                {"label":
                    {"sampler": {"log_evidence": 0.5},
                     "meta_data": {}}
                }
            }

        if extension == "json":
            import json

            with open(outdir + "test.json", "w") as f:
                json.dump(dictionary, f, indent=4, sort_keys=True)
        elif extension == "hdf5" or extension == "h5":
            import h5py
            from pesummary.core.file.meta_file import _recursively_save_dictionary_to_hdf5_file

            f = h5py.File(outdir + "test.h5", "w")
            _recursively_save_dictionary_to_hdf5_file(f, dictionary)
            f.close()
    return parameters, data
