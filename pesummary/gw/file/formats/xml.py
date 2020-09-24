# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from gwpy.table import Table
import numpy as np


def read_xml(
    path, format="ligolw", tablename="sim_inspiral", num=None, cls=None
):
    """Grab the data from an xml file

    Parameters
    ----------
    path: str
        Path to the injection file you wish to read
    format: str, optional
        The format of your xml. Default is 'ligolw'
    tablename: str, optional
        Name of the table you wish to load. Default is 'sim_inspiral'
    num: int, optional
        The row you wish to load. Default is None
    """
    from pesummary.gw.file.standard_names import standard_names

    table = Table.read(path, format=format, tablename=tablename)
    injection = {
        standard_names[key]: [table[key][num]] if num is not None else
        list(table[key]) for key in table.colnames if key in
        standard_names.keys()
    }
    if cls is not None:
        return cls(injection)
    parameters = list(injection.keys())
    samples = np.array([injection[param] for param in parameters])
    return parameters, samples.T.tolist()
