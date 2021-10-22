#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

import argparse
from pesummary.utils.exceptions import InputError
from pesummary.utils.utils import logger
from pesummary.core.inputs import Input
from pesummary.gw.inputs import GWInput
from pesummary.core.webpage.main import _WebpageGeneration

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is a lightweight version of summarypages. It
allows you to customise which parameters you wish to view rather than plotting
every single parameter in the result file"""


def command_line(parser=None):
    """Generate an Argument Parser object to control the command line options

    Parameters
    ----------
    """
    if parser is None:
        from pesummary.gw.parser import parser as _parser
        parser = _parser()

    parser.add_argument("--parameters", dest="parameters", nargs='+',
                        help=("list of parameters you wish to include in the "
                              "summarypages"),
                        default=None)
    return parser


class LWInput(Input):
    """
    """
    def __init__(self, *args, **kwargs):
        super(LWInput, self).__init__(*args, **kwargs)
        self.parameters_to_include = self.opts.parameters

    @property
    def parameters_to_include(self):
        return self._parameters_to_include

    @parameters_to_include.setter
    def parameters_to_include(self, parameters_to_include):
        self._parameters_to_include = parameters_to_include
        if parameters_to_include is None:
            raise InputError(
                "Please provide a list of parameters you wish to plot"
            )
        removed_labels = []
        for num, label in enumerate(self.labels):
            params = [
                i for i in parameters_to_include if i in
                list(self.samples[label].keys())
            ]
            not_included_params = [
                i for i in parameters_to_include if i not in
                list(self.samples[label].keys())
            ]
            if len(not_included_params) == len(parameters_to_include):
                logger.warning(
                    "Unable to find specified parameters {} in the file {}. "
                    "The file {} will not be included in the final "
                    "pages".format(
                        ", ".join(parameters_to_include), self.result_files[num],
                        self.result_files[num]
                    )
                )
                removed_labels.append(label)
            elif len(not_included_params) != 0:
                logger.warning(
                    "The parameters {} are not in the file {}. They will not "
                    "be included in the final pages".format(
                        ", ".join(not_included_params), self.result_files[num]
                    )
                )
            original = list(self.samples[label].keys())
            for i in original:
                if i not in params:
                    self.samples[label].pop(i)
        if len(removed_labels) == len(self.labels):
            raise InputError("Please provide a results file")
        for label in removed_labels:
            self.samples.pop(label)
            self.labels.remove(label)


class GWLWInput(GWInput, LWInput):
    pass


def main(args=None):
    """The main interface to `summarypageslw`
    """
    from .summarypages import main as _main
    _parser = command_line()
    _main(
        args=args, _parser=_parser, _core_input_cls=LWInput,
        _gw_input_cls=GWLWInput
    )


if __name__ == "__main__":
    main()
