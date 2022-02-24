# Licensed under an MIT style license -- see LICENSE.md

from gwpy.table import Table
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_xml(
    path, format="ligolw", tablename="sim_inspiral", num=None, cls=None,
    additional_xml_map={}, ignore_params=["beta"], **kwargs
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
    additional_xml_map: dict, optional
        Additional mapping of non standard names. Key is the
        standard parameter name and item is the name stored in
        the xml file
    """
    from pesummary.gw.file.standard_names import standard_names

    table = Table.read(path, format=format, tablename=tablename)
    injection = {
        standard_names[key]: [table[key][num]] if num is not None else
        list(table[key]) for key in table.colnames if key in
        standard_names.keys()
    }
    for key, item in additional_xml_map.items():
        injection[key] = (
            [table[item][num]] if num is not None else list(table[item])
        )
    for param in ignore_params:
        if param in injection.keys():
            _ = injection.pop(param)
    if cls is not None:
        return cls(injection)
    parameters = list(injection.keys())
    samples = np.array([injection[param] for param in parameters])
    return parameters, samples.T.tolist()
