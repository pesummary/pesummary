# Licensed under an MIT style license -- see LICENSE.md

import sqlite3
import numpy as np
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict, SamplesDict
from pesummary.utils.utils import logger, check_filename

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def read_sql(path, path_to_samples=None, remove_row_column="ROW", **kwargs):
    """Grab the parameters and samples in an sql database file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    path_to_samples: str/list, optional
        table or list of tables that you wish to load
    remove_row_column: str, optional
        remove the column with name 'remove_row_column' which indicates the row.
        Default 'ROW'
    """
    db = sqlite3.connect(path)
    d = db.cursor()
    d.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [i[0] for i in d.fetchall()]
    if path_to_samples is not None:
        if isinstance(path_to_samples, str):
            if path_to_samples in tables:
                tables = [path_to_samples]
            else:
                raise ValueError(
                    "{} not in list of tables".format(path_to_samples)
                )
        elif isinstance(path_to_samples, (np.ndarray, list)):
            if not all(name in tables for name in path_to_samples):
                names = [
                    name for name in path_to_samples if name not in tables
                ]
                raise ValueError(
                    "The tables: {} are not in the sql database".format(
                        ", ".join(names)
                    )
                )
            else:
                tables = path_to_samples
        else:
            raise ValueError("{} not understood".format(path_to_samples))
    parameters, samples = [], []
    for table in tables:
        d.execute(
            "SELECT * FROM {}".format(table)
        )
        samples.append(np.array(d.fetchall()))
        parameters.append([i[0] for i in d.description])
    for num, (_parameters, _samples) in enumerate(zip(parameters, samples)):
        if remove_row_column in _parameters:
            ind = _parameters.index(remove_row_column)
            _parameters.remove(remove_row_column)
            mask = np.ones(len(_samples.T), dtype=bool)
            mask[ind] = False
            samples[num] = _samples[:, mask]
    if len(tables) == 1:
        return parameters[0], np.array(samples[0]).tolist(), tables
    return parameters, np.array(samples).tolist(), tables


def write_sql(
    *args, table_name="MYTABLE", outdir="./", filename=None, overwrite=False,
    keys_as_table_name=True, delete_existing=False, **kwargs
):
    """Write a set of samples to an sql database

    Parameters
    ----------
    args: tuple, dict, MultiAnalysisSamplesDict
        the posterior samples you wish to save to file. Either a tuple
        of parameters and a 2d list of samples with columns corresponding to
        a given parameter, dict of parameters and samples, or a
        MultiAnalysisSamplesDict object with parameters and samples for
        multiple analyses
    table_name: str, optional
        name of the table to store the samples. If a MultiAnalysisSamplesDict
        if provided, this is ignored and the table_names are the labels stored
    outdir: str, optional
        directory to write the dat file
    filename: str, optional
        The name of the file that you wish to write
    overwrite: Bool, optional
        If True, an existing file of the same name will be overwritten
    keys_as_table_name: Bool, optional
        if True, ignore table_name and use the keys of the
        MultiAnalysisSamplesDict as the table name. Default True
    """
    default_filename = "pesummary_{}.db"
    filename = check_filename(
        default_filename=default_filename, outdir=outdir, label=table_name,
        filename=filename, overwrite=overwrite, delete_existing=delete_existing
    )

    if isinstance(args[0], MultiAnalysisSamplesDict):
        if isinstance(table_name, str):
            logger.info(
                "Ignoring the table name: {} and using the labels in the "
                "MultiAnalysisSamplesDict".format(table_name)
            )
            table_name = list(args[0].keys())
        elif isinstance(table_name, dict):
            if keys_as_table_name:
                logger.info(
                    "Ignoring table_name and using the labels in the "
                    "MultiAnalysisSamplesDict. To override this, set "
                    "`keys_as_table_name=False`"
                )
                table_name = list(args[0].keys())
            elif not all(key in table_name.keys() for key in args.keys()):
                raise ValueError("Please provide a table_name for all analyses")
            else:
                table_name = [
                    key for key in table_name.keys() if key in args.keys()
                ]
        else:
            raise ValueError(
                "Please provide table name as a dictionary which maps "
                "the analysis label to the table name"
            )
        table_name = list(args[0].keys())
        columns = [list(args[0][label].keys()) for label in table_name]
        rows = [
            np.array([args[0][label][param] for param in columns[num]]).T for
            num, label in enumerate(table_name)
        ]
    elif isinstance(args[0], dict):
        columns = list(args[0].keys())
        rows = np.array([args[0][param] for param in columns]).T
    else:
        columns, rows = args

    table_names = np.atleast_1d(table_name)
    columns = np.atleast_2d(columns)
    if np.array(rows).ndim == 1:
        rows = [[rows]]
    elif np.array(rows).ndim == 2:
        rows = [rows]

    if len(table_names) != len(columns):
        table_names = [
            "{}_{}".format(table_names[0], idx) for idx in range(len(columns))
        ]

    db = sqlite3.connect(filename)
    d = db.cursor()
    for num, table_name in enumerate(table_names):
        command = "CREATE TABLE {} (ROW INT, {});".format(
            table_name, ", ".join(["%s DOUBLE" % (col) for col in columns[num]])
        )
        for idx, row in enumerate(rows[num]):
            command += "INSERT INTO {} (ROW, {}) VALUES ({}, {});".format(
                table_name, ", ".join(columns[num]), idx,
                ", ".join([str(r) for r in row])
            )
        d.executescript(command)
