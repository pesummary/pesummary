# Licensed under an MIT style license -- see LICENSE.md

from ...core.cli.parser import ArgumentParser as _ArgumentParser
from ...core.cli.actions import DictionaryAction, DeprecatedStoreTrueAction

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class ArgumentParser(_ArgumentParser):
    """Extension of the pesummary.core.cli.parser.ArgumentParser object to handle
    gw specific command line arguments.

    Properties
    ----------
    fallback_options: dict
        dictionary of default kwargs
    pesummary_options: dict
        dictionary giving pesummary options
    command_line: str
        command line run
    command_line_arguments: list
        list giving command line arguments
    dynamic_argparse: list
        list of dynamic argparse functions that you wish to add to the
        argparse.Namespace object
    """
    @property
    def dynamic_argparse(self):
        return [
            add_dynamic_PSD_to_namespace,
            add_dynamic_calibration_to_namespace
        ]

    @property
    def gw_options(self):
        parser = ArgumentParser()
        parser.add_known_options_to_parser_from_key(parser, "gw")
        parser.add_known_options_to_parser_from_key(parser, "remnant")
        opts = [i.dest for i in vars(parser)["_actions"]]
        defaults = [i.default for i in vars(parser)["_actions"]]
        return opts, defaults

    def _pesummary_options(self):
        core_options = super(ArgumentParser, self)._pesummary_options()
        options = core_options.copy()
        gw_options = {
            "--disable_remnant": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Prevent remnant quantities from being calculated when the "
                    "conversions module is used"
                ),
                "key": "remnant",
            },
            "--force_BBH_remnant_computation": {
                "default": False,
                "action": "store_true",
                "help": (
                    "Use BBH fits to calculate remnant quantities for systems "
                    "that include tidal deformability parameters"
                ),
                "key": "remnant",
            },
            "--force_BH_spin_evolution": {
                "default": False,
                "action": "store_true",
                "help": (
                    "Use BH spin evolution methods to evolve spins in systems "
                    "that include tidal deformability parameters"
                ),
                "key": "remnant",
            },
            "--evolve_spins": {
                "dest": "evolve_spins_forwards",
                "action": DeprecatedStoreTrueAction(
                    new_option="--evolve_spins_forwards"
                ),
                "help": (
                    "Evolve the spins up to the Schwarzschild ISCO frequency "
                    "for remnant fits evaluation"
                ),
                "default": False,
                "key": "remnant",
            },
            "--evolve_spins_forwards": {
                "action": "store_true",
                "help": (
                    "Evolve the spins up to the Schwarzschild ISCO frequency "
                    "for remnant fits evaluation"
                ),
                "default": False,
                "key": "remnant",
            },
            "--evolve_spins_backwards": {
                "nargs": "?",
                "dest": "evolve_spins_backwards",
                "choices": ["precession_averaged", "hybrid_orbit_averaged"],
                "default": False,
                "help": (
                    "Method to use when evolving spins backwards to infinite "
                    "separation. Default 'precession_averaged'."
                ),
                "key": "remnant",
            },
            "--NRSur_fits": {
                "nargs": "?",
                "dest": "NRSur_fits",
                "default": False,
                "help": (
                    "The NRSurrogate you wish to use to calculate the remnant "
                    "quantities from your posterior samples. If not passed, "
                    "the average NR fits are used"
                ),
                "key": "remnant",
            },
            "--waveform_fits": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Use the provided approximant (either from command line or "
                    "stored in the result file) to calculate the remnant "
                    "quantities from your posterior samples. If not passed, "
                    "the average NR fits are used"
                ),
                "key": "remnant",
            },
            "--approximant": {
                "dest": "approximant",
                "help": "waveform approximant used to generate samples",
                "nargs": "+",
                "short": "-a",
                "key": "gw",
            },
            "--sensitivity": {
                "action": "store_true",
                "default": False,
                "help": "generate sky sensitivities for HL, HLV",
                "key": "gw",
            },
            "--gracedb": {
                "dest": "gracedb",
                "help": "gracedb of the event",
                "key": "gw",
            },
            "--gracedb_server": {
                "dest": "gracedb_server",
                "help": "service url to use when accessing gracedb",
                "key": "gw",
            },
            "--gracedb_data": {
                "dest": "gracedb_data",
                "nargs": "+",
                "default": ["t_0", "far", "created"],
                "help": (
                    "data you wish to download from gracedb and store in the "
                    "metafile"
                ),
                "key": "gw",
            },
            "--psd": {
                "dest": "psd",
                "action": DictionaryAction,
                "help": "psd files used",
                "nargs": "+",
                "default": {},
                "key": "gw",
            },
            "--{}_psd": {
                "dest": "example_psd",
                "metavar": "IFO:PATH_to_PSD.dat",
                "help": (
                    "psd files used for a specific label. '{}' should be "
                    "replaced with the label of interest. For example "
                    "--IMRPhenomPv3_psd H1:IF0_psd.dat"
                ),
                "key": "gw",
            },
            "--calibration": {
                "dest": "calibration",
                "help": "files for the calibration envelope",
                "nargs": "+",
                "default": {},
                "action": DictionaryAction,
                "key": "gw",
            },
            "--{}_calibration": {
                "dest": "example_calibration",
                "metavar": "IFO:PATH_to_CAL.txt",
                "help": (
                    "calibration files used for a specific label. '{}' should "
                    "be replaced with the label of interest. For example "
                    "--IMRPhenomPv3_calibration H1:IF0_cal.dat"
                ),
                "key": "gw",
            },
            "--trigfile": {
                "dest": "inj_file",
                "help": "xml file containing the trigger values",
                "nargs": "+",
                "key": "gw",
            },
            "--gwdata": {
                "dest": "gwdata",
                "help": "channels and paths to strain cache files",
                "action": DictionaryAction,
                "metavar": "CHANNEL:CACHEFILE or PICKLEFILE",
                "nargs": "+",
                "key": "gw",
            },
            "--multi_threading_for_skymap": {
                "action": "store_true",
                "default": False,
                "help": (
                    "use multi-threading to speed up generation of ligo.skymap"
                ),
                "key": "gw"
            },
            "--nsamples_for_skymap": {
                "dest": "nsamples_for_skymap",
                "help": (
                    "The number of samples used to generate the ligo.skymap. "
                    "These samples will be randomly drawn from the posterior "
                    "distributions"
                ),
                "key": "gw",
            },
            "--calculate_multipole_snr": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Calculate the SNR in the (ell, m) = [(2, 1), (3, 3), "
                    "(4, 4)] subdominant multipoles based on the posterior "
                    "samples"
                ),
                "key": "gw",
            },
            "--calculate_precessing_snr": {
                "action": "store_true",
                "default": False,
                "help": (
                    "Calculate the precessing SNR based on the posterior "
                    "samples"
                ),
                "key": "gw",
            },
            "--psd_default": {
                "dest": "psd_default",
                "default": "aLIGOZeroDetHighPower",
                "help": (
                    "The PSD to use for conversions when no psd file is "
                    "provided. Default aLIGOZeroDetHighPower"
                ),
                "key": "gw",
            },
            "--f_low": {
                "dest": "f_low",
                "nargs": "+",
                "help": "Low frequency cutoff used to generate the samples",
                "key": "gw",
            },
            "--f_ref": {
                "dest": "f_ref",
                "help": "Reference frequency used to generate the samples",
                "nargs": "+",
                "key": "gw",
            },
            "--f_final": {
                "dest": "f_final",
                "nargs": "+",
                "type": float,
                "help": (
                    "Final frequency to use when calculating the precessing snr"
                ),
                "key": "gw"
            },
            "--delta_f": {
                "dest": "delta_f",
                "help": (
                    "Difference in frequency samples when calculating the "
                    "precessing snr"
                ),
                "nargs": "+",
                "type": float,
                "key": "gw",
            },
            "--no_ligo_skymap": {
                "action": "store_true",
                "default": False,
                "help": "do not generate a skymap with ligo.skymap",
                "key": "gw",
            },
            "--gw": {
                "action": "store_true",
                "help": "run with the gravitational wave pipeline",
                "default": False,
                "key": "gw",
            },
            "--public": {
                "action": "store_true",
                "help": "generate public facing summary pages",
                "default": False,
                "key": "gw"
            },
            "--redshift_method": {
                "dest": "redshift_method",
                "help": "The method to use when estimating the redshift",
                "choices": ["approx", "exact"],
                "default": "approx",
                "key": "gw",
            },
            "--cosmology": {
                "dest": "cosmology",
                "help": "The cosmology to use when calculating the redshift",
                "default": "Planck15",
                "key": "gw"
            },
            "--no_conversion": {
                "action": "store_true",
                "default": False,
                "help": "Do not generate any conversions",
                "key": "gw"
            }
        }
        options.update(gw_options)
        return options

    def add_gw_group(self):
        gw_group = self.add_argument_group(
            "\n\n=====================================================\n"
            "Options specific for gravitational wave results files\n"
            "====================================================="
        )
        return self.add_known_options_to_parser_from_key(gw_group, "gw")

    def add_remnant_group(self):
        remnant_group = self.add_argument_group(
            "Options specific for calculating the remnant properties\n"
            "-------------------------------------------------------"
        )
        return self.add_known_options_to_parser_from_key(
            remnant_group, "remnant"
        )

    def add_all_groups_to_parser(self):
        super(ArgumentParser, self).add_all_groups_to_parser()
        self.add_gw_group()
        self.add_remnant_group()


class TGRArgumentParser(ArgumentParser):
    """Extension of the pesummary.gw.cli.parser.ArgumentParser object to handle
    TGR specific command line arguments.

    Properties
    ----------
    fallback_options: dict
        dictionary of default kwargs
    pesummary_options: dict
        dictionary giving pesummary options
    command_line: str
        command line run
    command_line_arguments: list
        list giving command line arguments
    dynamic_argparse: list
        list of dynamic argparse functions that you wish to add to the
        argparse.Namespace object
    """
    @property
    def dynamic_argparse(self):
        return [add_dynamic_tgr_kwargs_to_namespace]

    def _pesummary_options(self):
        TESTS = ["imrct"]
        _options = super(TGRArgumentParser, self)._pesummary_options()
        options = {
            "--test": {
                "short": "-t",
                "help": (
                    "What test do you want to run? Currently only supports "
                    "{}".format(", ".join(TESTS))
                ),
                "required": True,
                "choices": TESTS,
            },
            "--{test}_kwargs": {
                "dest": "example_test_kwargs",
                "help": (
                    "Kwargs you wish to use when postprocessing the results. "
                    "Kwargs should be provided as a dictionary 'kwarg:value'. "
                    "For example `--imrct_kwargs N_bins:201 multi_process:4` "
                    "would pass the kwargs N_bins=201, multi_process=4 to the "
                    "IMRCT function. The test name '{test}' should match the "
                    "test provided with the --test flag"
                ),
            },
            "--labels": {
                "help": (
                    "Labels used to distinguish runs. The label format is "
                    "dependent on the TGR test you wish to use. For the IMRCT "
                    "test, labels need to be inspiral and postinspiral if "
                    "analysing a single event or {label1}:inspiral,"
                    "{label1}:postinspiral,{label2}:inspiral,"
                    "{label2}:postinspiral,... if analysing two or more events "
                    "(where label1/label2 is a unique string to distinguish "
                    "files from a single event). If a metafile is provided, "
                    "labels need to be {inspiral_label}:inspiral "
                    "{postinspiral_label}:postinspiral where inspiral_label "
                    "and postinspiral_label are the pesummary labels for the "
                    "inspiral and postinspiral analyses respectively."
                ),
                "nargs": "+",
            },
            "--cutoff_frequency": {
                "type": float,
                "nargs": "+",
                "help": (
                    "Cutoff Frequency for IMRCT. Overrides any cutoff "
                    "frequency present in the supplied files. The supplied "
                    "cutoff frequency will only be used as metadata and "
                    "does not affect the cutoff frequency used in the "
                    "analysis. If only one number is supplied, the inspiral "
                    "maximum frequency and the postinspiral maximum frequency "
                    "are set to the same number. If a list of length 2 is "
                    "supplied, this assumes that the one corresponding to the "
                    "inspiral label is the maximum frequency for the inspiral "
                    "and that corresponding to the postinspiral label is "
                    "the minimum frequency for the postinspiral"
                )
            },
            "--links_to_pe_pages": {
                "help": "URLs for PE results pages separated by spaces.",
                "nargs": "+",
                "default": [],
            },
            "--disable_pe_page_generation": {
                "action": "store_true",
                "help": (
                    "Disable PE page generation for the input samples. This "
                    "option is only relevant if no URLs for PE results pages "
                    "are provided using --links_to_pe_pages."
                )
            },
            "--pe_page_options": {
                "type": str,
                "default": "",
                "help": (
                    "Additional options to pass to 'summarypages' when "
                    "generating PE webpages. All options should be wrapped in "
                    "quotation marks like, --pe_page_options='--no_ligo_skymap "
                    "--nsamples 1000 --psd...'. See 'summarypages --help' for "
                    "details. These options are added to base executable: "
                    "'summarypages --webdir {} --samples {} --labels {} --gw'"
                ),
            },
            "--make_diagnostic_plots": {
                "action": "store_true",
                "help": "Make extra diagnostic plots"
            }
        }
        extra_options = [
            "--webdir", "--approximant", "--evolve_spins_forwards", "--f_low",
            "--samples"
        ]
        for key in extra_options:
            options[key] = _options[key]
        return options


def add_dynamic_argparse(
        existing_namespace, pattern, example="--{}_psd", default={},
        nargs='+', action=DictionaryAction, command_line=None
):
    """Add a dynamic argparse argument and add it to an existing
    argparse.Namespace object

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    pattern: str
        generic pattern for customg argparse. For example '--*_psd'
    example: str, optional
        example string to demonstrate usage
    default: obj, optional
        the default argument for the dynamic argparse object
    nargs: str
    action: argparse.Action
        argparse action to use for the dynamic argparse
    command_line: str, optional
        command line you wish to pass. If None, command line taken from
        sys.argv
    """
    import fnmatch
    import collections
    import argparse
    if command_line is None:
        from pesummary.utils.utils import command_line_arguments
        command_line = command_line_arguments()
    commands = fnmatch.filter(command_line, pattern)
    duplicates = [
        item for item, count in collections.Counter(commands).items() if
        count > 1
    ]
    if example in commands:
        commands.remove(example)
    if len(duplicates) > 0:
        raise Exception(
            "'{}' has been repeated. Please give a unique argument".format(
                duplicates[0]
            )
        )
    parser = argparse.ArgumentParser()
    for i in commands:
        parser.add_argument(
            i, help=argparse.SUPPRESS, action=action, nargs=nargs,
            default=default
        )
    args, unknown = parser.parse_known_args(args=command_line)
    existing_namespace.__dict__.update(vars(args))
    return args, unknown


def add_dynamic_PSD_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic PSD argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    command_line: str, optional
        The command line which you are passing. Default None
    """
    return add_dynamic_argparse(
        existing_namespace, "--*_psd", command_line=command_line
    )


def add_dynamic_calibration_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic calibration argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments too
    command_line: str, optional
        The command line which you are passing. Default None
    """
    return add_dynamic_argparse(
        existing_namespace, "--*_calibration", example="--{}_calibration",
        command_line=command_line
    )


def add_dynamic_tgr_kwargs_to_namespace(existing_namespace, command_line=None):
    """Add a dynamic TGR kwargs argument to the argparse namespace

    Parameters
    ----------
    existing_namespace: argparse.Namespace
        existing namespace you wish to add the dynamic arguments to
    command_line: str, optional
        The command line which you are passing. Default None
    """
    return add_dynamic_argparse(
        existing_namespace, "--*_kwargs", example="--{}_kwargs",
        command_line=command_line
    )
