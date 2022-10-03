# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.injection import Injection
from pesummary.gw.file.standard_names import standard_names

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
    def __init__(
        self, *args, mapping=standard_names, conversion=True,
        conversion_kwargs={}, **kwargs
    ):
        super(GWInjection, self).__init__(*args, mapping=mapping, **kwargs)
        if conversion:
            from pesummary.gw.conversions import convert

            converted = convert(self, **conversion_kwargs)
            if conversion_kwargs.get("return_kwargs", False):
                converted = converted[0]
            self.update(converted)

    @classmethod
    def read(
        cls, path, conversion=True, conversion_kwargs={}, additional_xml_map={},
        **kwargs
    ):
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
        return super(GWInjection, cls).read(
            path, conversion=conversion, conversion_kwargs=conversion_kwargs,
            mapping=standard_names, num=0, read_kwargs={
                "additional_xml_map": additional_xml_map,
                "add_zero_likelihood": False
            }
        )
