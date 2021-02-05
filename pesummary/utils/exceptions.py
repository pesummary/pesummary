# Licensed under an MIT style license -- see LICENSE.md

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class InputError(Exception):
    """
    """
    def __init__(self, message):
        super(InputError, self).__init__(message)


class PlotError(Exception):
    """
    """
    def __init__(self, message):
        super(PlotError, self).__init__(message)


class EvolveSpinError(Exception):
    """
    """
    def __init__(self, message):
        super(EvolveSpinError, self).__init__(message)
