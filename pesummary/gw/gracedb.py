# Licensed under an MIT style license -- see LICENSE.md

from ligo.gracedb.rest import GraceDb
from ligo.gracedb.exceptions import HTTPError
from pesummary.utils.utils import logger
from pesummary import conf

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def get_gracedb_data(
    gracedb_id, superevent=False, info=None, json=None,
    service_url=conf.gracedb_server
):
    """Grab data from GraceDB for a specific event.

    Parameters
    ----------
    gracedb_id: str
        the GraceDB id of the event you wish to retrieve the data for
    superevent: Bool, optional
        True if the gracedb_id you are providing is a superevent
    info: str/list, optional
        either a string or list of strings for information you wish to
        retrieve
    json: dict, optional
        data that you have already downloaded from gracedb
    service_url: str, optional
        service url you wish to use when accessing data from GraceDB
    """
    client = GraceDb(service_url=service_url)
    if json is None and superevent:
        json = client.superevent(gracedb_id).json()
    elif json is None:
        try:
            json = client.superevent(gracedb_id).json()
        except HTTPError:
            json = client.event(gracedb_id).json()

    if isinstance(info, str) and info in json.keys():
        return str(json[info])
    elif isinstance(info, str):
        raise AttributeError(
            "Could not find '{}' in the gracedb dictionary. Available entries "
            "are: {}".format(info, ", ".join(json.keys()))
        )
    elif isinstance(info, list):
        data = {}
        for _info in info:
            if _info in json.keys():
                data[_info] = json[_info]
            else:
                logger.warning(
                    "Unable to find any information for '{}'".format(_info)
                )
        return data
    elif info is None:
        return json
    else:
        raise ValueError(
            "info data not understood. Please provide either a list or string"
        )
