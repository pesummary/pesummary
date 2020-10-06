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


def paths_to_key(key, dictionary, current_path=None):
    """Return the path to a key stored in a nested dictionary

    Parameters
    ----------`
    key: str
        the key that you would like to find
    dictionary: dict
        the nested dictionary that has the key stored somewhere within it
    current_path: str, optional
        the current level in the dictionary
    """
    if current_path is None:
        current_path = []

    for k, v in dictionary.items():
        if k == key:
            yield current_path + [key]
        else:
            if isinstance(v, dict):
                path = current_path + [k]
                for z in paths_to_key(key, v, path):
                    yield z


def convert_list_to_item(dictionary):
    """Convert all nested lists of a single value to an item

    Parameters
    ----------
    dictionary: dict
        nested dictionary with nested lists
    """
    from pesummary.utils.samples_dict import Array

    for key, value in dictionary.items():
        if isinstance(value, dict):
            convert_list_to_item(value)
        else:
            if isinstance(value, (list, np.ndarray, Array)):
                if len(value) == 1 and isinstance(value[0], bytes):
                    dictionary.update({key: value[0].decode("utf-8")})
                elif len(value) == 1:
                    dictionary.update({key: value[0]})
    return dictionary


def load_recursively(key, dictionary):
    """Return an entry in a nested dictionary for a key of format 'a/b/c/d'

    Parameters
    ----------
    key: str
        key of format 'a/b/c/d'
    dictionary: dict
        the dictionary that has the key stored
    """
    if "/" in key:
        key = key.split("/")
    if isinstance(key, (str, float)):
        key = [key]
    if key[-1] in dictionary.keys():
        try:
            converted_dictionary = convert_list_to_item(
                dictionary[key[-1]]
            )
            yield converted_dictionary
        except AttributeError:
            yield dictionary[key[-1]]
    else:
        old, new = key[0], key[1:]
        for z in load_recursively(new, dictionary[old]):
            yield z


def edit_dictionary(dictionary, path, value):
    """Replace an entry in a nested dictionary

    Parameters
    ----------
    dictionary: dict
        the nested dictionary that you would like to edit
    path: list
        the path to the key that you would like to edit
    value:
        the replacement
    """
    from functools import reduce
    from operator import getitem

    edit = dictionary.copy()
    reduce(getitem, path[:-1], edit)[path[-1]] = value
    return edit


class Dict(dict):
    """Base nested dictionary class.

    Parameters
    ----------
    value_class: func, optional
        Class you wish to use for the nested dictionary
    value_columns: list, optional
        Names for each column in value_class to be stored as properties
    **kwargs: dict
        All other kwargs are turned into properties of the class. Key
        is the name of the property
    """
    def __init__(
        self, *args, value_class=np.ndarray, value_columns=None, **kwargs
    ):
        super(Dict, self).__init__()
        if isinstance(args[0], dict):
            data = args[0]
        else:
            data = {
                key: value for key, value in zip(*args)
            }
        for key, value in data.items():
            self[key] = value_class(value)
        if value_columns is not None:
            for key in self.keys():
                if len(value_columns) == self[key].shape[1]:
                    for num, col in enumerate(value_columns):
                        setattr(self[key], col, np.array(self[key].T[num]))
        for key, item in kwargs.items():
            setattr(self, key, item)
