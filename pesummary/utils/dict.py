# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
import copy

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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


def convert_value_to_string(dictionary):
    """Convert all nested lists of a single value to an item

    Parameters
    ----------
    dictionary: dict
        nested dictionary with nested lists
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            convert_value_to_string(value)
        else:
            dictionary.update({key: str(value)})
    return dictionary


def convert_list_to_item(dictionary):
    """Convert all nested lists of a single value to an item

    Parameters
    ----------
    dictionary: dict
        nested dictionary with nested lists
    """
    from pesummary.utils.array import Array

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
        self, *args, value_class=np.array, value_columns=None, _init=True,
        make_dict_kwargs={}, logger_warn="warn", latex_labels={},
        extra_kwargs={}, **kwargs
    ):
        from .parameters import Parameters
        super(Dict, self).__init__()
        if not _init:
            return
        self.logger_warn = logger_warn
        self.all_latex_labels = latex_labels
        if isinstance(args[0], dict):
            if args[0].__class__.__name__ == "SamplesDict":
                self.parameters = list(args[0].keys(remove_debug=False))
                _iterator = args[0].items(remove_debug=False)
            else:
                self.parameters = list(args[0].keys())
                _iterator = args[0].items()
            _samples = [args[0][param] for param in self.parameters]
            try:
                self.samples = np.array(_samples)
            except ValueError:
                self.samples = _samples
        else:
            self.parameters, self.samples = args
            _iterator = zip(self.parameters, self.samples)
        try:
            self.make_dictionary(**make_dict_kwargs)
        except (TypeError, IndexError):
            for key, item in _iterator:
                try:
                    self[key] = value_class(item)
                except Exception:
                    self[key] = value_class(*item)

        if value_columns is not None:
            for key in self.keys():
                if len(value_columns) == self[key].shape[1]:
                    for num, col in enumerate(value_columns):
                        setattr(self[key], col, np.array(self[key].T[num]))
        for key, item in kwargs.items():
            setattr(self, key, item)
        self._update_latex_labels()
        self.extra_kwargs = extra_kwargs
        self.parameters = Parameters(self.parameters)

    def __getitem__(self, key):
        """Return an object representing the specialization of Dict
        by type arguments found in key.
        """
        if isinstance(key, list):
            allowed = [_key for _key in key if _key in self.keys()]
            remove = [_key for _key in self.keys() if _key not in allowed]
            if len(allowed):
                if len(allowed) != len(key):
                    import warnings
                    warnings.warn(
                        "Only returning a dict with keys: {} as not all keys "
                        "are in the {} class".format(
                            ", ".join(allowed), self.__class__.__name__
                        )
                    )
                _self = copy.deepcopy(self)
                for _key in remove:
                    _self.pop(_key)
                return _self
            raise KeyError(
                "The keys: {} are not available in {}. The list of "
                "available keys are: {}".format(
                    ", ".join(key), self.__class__.__name__,
                    ", ".join(self.keys())
                )
            )
        elif isinstance(key, str):
            if key not in self.keys():
                raise KeyError(
                    "{} not in {}. The list of available keys are {}".format(
                        key, self.__class__.__name__, ", ".join(self.keys())
                    )
                )
        return super(Dict, self).__getitem__(key)

    @property
    def latex_labels(self):
        return self._latex_labels

    @property
    def plotting_map(self):
        return {}

    @property
    def available_plots(self):
        return list(self.plotting_map.keys())

    def _update_latex_labels(self):
        """Update the stored latex labels
        """
        self._latex_labels = {
            param: self.all_latex_labels[param] if param in
            self.all_latex_labels.keys() else param for param in self.parameters
        }

    def plot(self, *args, type="", **kwargs):
        """Generate a plot for data stored in Dict

        Parameters
        ----------
        *args: tuple
            all arguments are passed to the plotting function
        type: str
            name of the plot you wish to make
        **kwargs: dict
            all additional kwargs are passed to the plotting function
        """
        if type not in self.plotting_map.keys():
            raise NotImplementedError(
                "The {} method is not currently implemented. The allowed "
                "plotting methods are {}".format(
                    type, ", ".join(self.available_plots)
                )
            )
        return self.plotting_map[type](*args, **kwargs)

    def make_dictionary(self, *args, **kwargs):
        """Add the parameters and samples to the class
        """
        raise TypeError
