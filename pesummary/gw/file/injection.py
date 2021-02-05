# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.core.file.formats.base_read import Read
from pesummary.core.file.injection import Injection

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        kwargs that are passed to the `pesummary.gw.conversions.convert`
        function
    """
    def __init__(self, *args, conversion=True, conversion_kwargs={}, **kwargs):
        super(GWInjection, self).__init__(*args, **kwargs)
        if conversion:
            from pesummary.gw.conversions import convert

            converted = convert(self, **conversion_kwargs)
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
            kwargs that are passed to the `pesummary.gw.conversions.convert`
            function
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
