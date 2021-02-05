#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary import __version__

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
__doc__ = """This executable is used to display the version of PESummary that
is currently being used"""


def main():
    """Top level interface for `summaryversion`
    """
    print(__version__)
    return
