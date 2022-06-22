# Licensed under an MIT style license -- see LICENSE.md

from ...core.cli.parser import parser as core_parser
from ...core.cli.actions import DictionaryAction
from .command_line import insert_gwspecific_option_group

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class parser(core_parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """
    def __init__(self, existing_parser=None):
        super(parser, self).__init__(existing_parser=existing_parser)
        if existing_parser is None:
            insert_gwspecific_option_group(self._parser)

    @property
    def dynamic_argparse(self):
        return [
            add_dynamic_PSD_to_namespace,
            add_dynamic_calibration_to_namespace
        ]


class TGRparser(parser):
    """Class to handle parsing command line arguments

    Attributes
    ----------
    dynamic_argparse: list
        list of dynamic argparse methods
    """

    def __init__(self, existing_parser=None):
        super(parser, self).__init__(existing_parser=existing_parser)

    @property
    def dynamic_argparse(self):
        return [add_dynamic_tgr_kwargs_to_namespace]


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
