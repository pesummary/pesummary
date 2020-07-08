# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org> This program is free
# software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from ligo.gracedb.rest import GraceDb
from ligo.gracedb.exceptions import HTTPError
from pesummary.utils.utils import logger
from pesummary import conf


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
