# Licensed under an MIT style license -- see LICENSE.md

from pesummary.gw.file.strain import StrainData
from glue.lal import Cache

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_lcf(path, channel=None, **kwargs):
    """Grab the data stored in a .lcf file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    channel: str
        channel to use when reading in the data
    """
    if channel is None:
        raise ValueError("Please provider a channel for reading the data")

    with open(path, "r") as f:
        data = Cache.fromfile(f)

    return StrainData.read(data, channel, **kwargs)
