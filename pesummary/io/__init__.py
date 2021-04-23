# Licensed under an MIT style license -- see LICENSE.md

from .read import read
from .write import write

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def available_formats():
    """Return the available formats for reading and writing

    Returns
    -------
    tuple: tuple of sets. First set are the available formats for reading.
    Second set are the available sets for writing.
    """
    import pesummary.core.file.formats
    import pesummary.gw.file.formats
    import pkgutil
    import importlib

    read_formats, write_formats = [], []
    modules = {
        "gw": pesummary.gw.file.formats, "core": pesummary.core.file.formats
    }
    for package in ["core", "gw"]:
        formats = [
            a for _, a, _ in pkgutil.walk_packages(path=modules[package].__path__)
        ]
        for _format in formats:
            _submodule = importlib.import_module(
                "pesummary.{}.file.formats.{}".format(package, _format)
            )
            if hasattr(_submodule, "write_{}".format(_format)):
                write_formats.append(_format)
            if hasattr(_submodule, "read_{}".format(_format)):
                read_formats.append(_format)
    return set(read_formats), set(write_formats)
