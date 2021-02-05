# Licensed under an MIT style license -- see LICENSE.md

import configparser
import os
from pesummary.utils.utils import check_filename, logger
from pesummary.utils.decorators import open_config
from pesummary.utils.dict import convert_value_to_string

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def save_config_dictionary_to_file(
    config_dict, outdir="./", filename=None, overwrite=False, label=None,
    _raise=True, **kwargs
):
    """Save a dictionary containing the configuration settings to a file

    Parameters
    ----------
    config_dict: dict
        dictionary containing the configuration settings
    outdir: str, optional
        path indicating where you would like to configuration file to be
        saved. Default is current working directory
    filename: str, optional
        rhe name of the file you wish to write to
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    """
    _filename = check_filename(
        default_filename="pesummary_{}.ini", outdir=outdir, label=label,
        filename=filename, overwrite=overwrite
    )
    config = configparser.ConfigParser()
    config.optionxform = str
    if config_dict is None:
        if _raise:
            raise ValueError("No config data found. Unable to write to file")
        logger.warn("No config data found. Unable to write to file")
        return

    for key in config_dict.keys():
        config[key] = convert_value_to_string(config_dict[key])

    with open(_filename, "w") as configfile:
        config.write(configfile)
    return _filename


@open_config(index=0)
def read_ini(path):
    """Return the config data as a dictionary

    Parameters
    ----------
    path: str
        path to the configuration file
    """
    config = path
    if config.error:
        raise ValueError(
            "Unable to open %s with configparser because %s" % (
                config.path_to_file, config.error
            )
        )
    sections = config.sections()
    data = {}
    if sections != []:
        for i in sections:
            data[i] = {}
            for key in config["%s" % (i)]:
                data[i][key] = config["%s" % (i)]["%s" % (key)]
    return data


def write_ini(
    config_dictionary, outdir="./", label=None, filename=None, overwrite=False,
    **kwargs
):
    """Write dictonary containing the configuration settings to an ini file

    Parameters
    ----------
    config_dict: dict
        dictionary containing the configuration settings
    outdir: str, optional
        path indicating where you would like to configuration file to be
        saved. Default is current working directory
    label: str, optional
        The label of the analysis. This is used in the filename if a filename
        if not specified
    filename: str, optional
        rhe name of the file you wish to write to
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    """
    return save_config_dictionary_to_file(
        config_dictionary, outdir=outdir, label=label, filename=filename,
        overwrite=overwrite, **kwargs
    )
