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

import numpy as np
from pesummary.core.file.formats.base_read import Read
from pesummary.core.file.injection import Injection


class GWInjection(Injection):
    """Class to handle injection information

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter
    conversion: Bool, optional
        If True, convert all injection parameters
    conversion_kwargs: dict, optional
        kwargs that are passed to the
        `pesummary.gw.file.conversions._Conversion` class
    """
    def __init__(self, *args, conversion=True, conversion_kwargs={}, **kwargs):
        super(GWInjection, self).__init__(*args, **kwargs)
        if conversion:
            from pesummary.gw.file.conversions import _Conversion

            converted = _Conversion(self, **conversion_kwargs)
            if conversion_kwargs.get("return_kwargs", False):
                converted = converted[0]
            for key, value in converted.items():
                self[key] = value

    @classmethod
    def read(cls, path, conversion=True, conversion_kwargs={}, **kwargs):
        """Read an injection file and initalize the Injection class

        Parameters
        ----------
        path: str
            Path to the injection file you wish to read
        conversion: Bool, optional
            If True, convert all injection parameters
        conversion_kwargs: dict, optional
            kwargs that are passed to the
            `pesummary.gw.file.conversions._Conversion` class
        **kwargs: dict
            All kwargs passed to the format specific read function
        """
        if Read.extension_from_path(path) == "xml":
            data = GWInjection._grab_injection_from_xml_file(path, **kwargs)
            return cls(
                data, conversion=conversion, conversion_kwargs=conversion_kwargs
            )
        else:
            return super(GWInjection, cls).read(
                path, conversion=conversion, conversion_kwargs=conversion_kwargs
            )

    @staticmethod
    def _grab_injection_from_xml_file(
        injection_file, format="ligolw", tablename="sim_inspiral", num=0
    ):
        """Grab the data from an xml injection file

        Parameters
        ----------
        injection_file: str
            Path to the injection file you wish to read
        format: str, optional
            The format of your xml. Default is 'ligolw'
        tablename: str, optional
            Name of the table you wish to load. Default is 'sim_inspiral'
        num: int, optional
            The injection row you wish to load. Default is 0
        """
        from gwpy.table import Table
        from pesummary.gw.file.standard_names import standard_names

        table = Table.read(
            injection_file, format=format, tablename=tablename
        )
        injection = {
            standard_names[key]: [table[key][num]] for key in table.colnames
            if key in standard_names.keys()
        }
        return injection
