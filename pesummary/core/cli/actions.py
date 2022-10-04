# Licensed under an MIT style license -- see LICENSE.md

import re
import copy
import os
import ast
import argparse
import configparser
import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class CheckFilesExistAction(argparse.Action):
    """Class to extend the argparse.Action to identify if files exist
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.check_input(values)

    def check_input(self, value):
        """Check that all files provided exist

        Parameters
        ----------
        value: str, list, dict
            data structure that you wish to check
        """
        if isinstance(value, list):
            for ff in value:
                _ = self.check_input(ff)
        elif isinstance(value, str):
            _ = self._is_file(value)
        elif isinstance(value, dict):
            for _value in value.values():
                _ = self.check_input(_value)
        else:
            _ = self._is_file(value)

    def _is_file(self, ff):
        """Return True if the file exists else raise a FileNotFoundError
        exception

        Parameters
        ----------
        ff: str
            path to file you wish to check
        """
        cond = any(_str in ff for _str in ["*", "@", "https://"])
        cond2 = isinstance(ff, str) and ff.lower() == "none"
        if not os.path.isfile(ff) and not cond and not cond2:
            raise FileNotFoundError(
                "The file '{}' provided for '{}' does not exist".format(
                    ff, self.dest
                )
            )
        return True


class BaseDeprecatedAction(object):
    """Class to handle deprecated argparse options
    """
    class _BaseDeprecatedAction(object):
        def __call__(self, *args, **kwargs):
            import warnings
            msg = (
                "The option '{}' is out-of-date and may not be supported in "
                "future releases.".format(self.option_strings[0])
            )
            if _new_option is not None:
                msg += " Please use '{}'".format(_new_option)
            warnings.warn(msg)
            return super().__call__(*args, **kwargs)

    def __new__(cls, *args, new_option=None, **kwargs):
        global _new_option
        _new_option = new_option


class DeprecatedStoreAction(BaseDeprecatedAction):
    """Class to handle deprecated argparse._StoreAction options
    """
    class _DeprecatedStoreAction(
        BaseDeprecatedAction._BaseDeprecatedAction, argparse._StoreAction
    ):
        pass

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, *args, **kwargs)
        return cls._DeprecatedStoreAction


class DeprecatedStoreTrueAction(BaseDeprecatedAction):
    """Class to handle deprecated argparse._StoreTrueAction options
    """
    class _DeprecatedStoreTrueAction(
        BaseDeprecatedAction._BaseDeprecatedAction, argparse._StoreTrueAction
    ):
        pass

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, *args, **kwargs)
        return cls._DeprecatedStoreTrueAction


class DeprecatedStoreFalseAction(BaseDeprecatedAction):
    """Class to handle deprecated argparse._StoreFalseAction options
    """
    class _DeprecatedStoreFalseAction(
        BaseDeprecatedAction._BaseDeprecatedAction, argparse._StoreFalseAction
    ):
        pass

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, *args, **kwargs)
        return cls._DeprecatedStoreFalseAction


class ConfigAction(argparse.Action):
    """Class to extend the argparse.Action to handle dictionaries as input
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        items = {}
        config = configparser.ConfigParser()
        config.optionxform = str
        try:
            config.read(values)
            sections = config.sections()
            for section in sections:
                for key, value in config.items(section):
                    if value.lower() == "true":
                        items[key] = True
                    elif value.lower() == "false":
                        items[key] = False
                    elif value.lower() == "none":
                        items[key] = None
                    else:
                        try:
                            _type = getattr(
                                parser, "_option_string_actions"
                            )["--{}".format(key)].type
                        except Exception:
                            _type = None
                        if ":" in value or "{" in value:
                            try:
                                items[key] = self.dict_from_str(value, dtype=_type)
                            except Exception:
                                items[key] = value
                        elif "," in value or "[" in value:
                            items[key] = self.list_from_str(value, _type)
                        else:
                            if _type is not None:
                                items[key] = _type(value)
                            else:
                                items[key] = value
        except Exception:
            pass
        for i in vars(namespace).keys():
            if i in items.keys():
                setattr(namespace, i, items[i])

    @staticmethod
    def dict_from_str(string, delimiter=":", dtype=None):
        """Reformat the string into a dictionary

        Parameters
        ----------
        string: str
            string that you would like reformatted into a dictionary
        """
        string = string.replace("'", "")
        string = string.replace('"', '')
        string = string.replace("=", ":")
        string = string.replace(delimiter, ":")
        if "dict(" in string:
            string = string.replace("dict(", "{")
            string = string.replace(")", "}")
        string = string.replace(" ", "")
        string = re.sub(r'([A-Za-z/\.0-9][^\[\],:"}]*)', r'"\g<1>"', string)
        string = string.replace('""', '"')
        try:
            mydict = ast.literal_eval(string)
        except ValueError as e:
            pass
        for key in mydict:
            if isinstance(mydict[key], str) and mydict[key].lower() == "true":
                mydict[key] = True
            elif isinstance(mydict[key], str) and mydict[key].lower() == "false":
                mydict[key] = False
            else:
                try:
                    mydict[key] = int(mydict[key])
                except ValueError:
                    try:
                        mydict[key] = float(mydict[key])
                    except ValueError:
                        mydict[key] = mydict[key]
            if dtype is not None:
                mydict[key] = dtype(mydict[key])
        return mydict

    @staticmethod
    def list_from_str(string, dtype=None):
        """Reformat the string into a list

        Parameters
        ----------
        string: str
            string that you would like reformatted into a list
        """
        list = []
        string = string.replace("'", "")
        if "[" in string:
            string = string.replace("[", "")
        if "]" in string:
            string = string.replace("]", "")
        if ", " in string:
            list = string.split(", ")
        elif "," in string:
            list = string.split(",")
        else:
            list = [string]
        if dtype is not None:
            list = [dtype(_) for _ in list]
        return list


class DictionaryAction(argparse.Action):
    """Class to extend the argparse.Action to handle dictionaries as input
    """
    def __call__(self, parser, namespace, values, option_string=None):
        bool = [True if ':' in value else False for value in values]
        if all(i is True for i in bool):
            setattr(namespace, self.dest, {})
        elif all(i is False for i in bool):
            setattr(namespace, self.dest, [])
        else:
            raise ValueError("Did not understand input")

        items = getattr(namespace, self.dest)
        items = copy.copy(items)
        for value in values:
            value = value.split(':')
            if len(value) > 2:
                value = [":".join(value[:-1]), value[-1]]
            if len(value) == 2:
                if value[0] in items.keys():
                    if not isinstance(items[value[0]], list):
                        items[value[0]] = [items[value[0]]]
                    items[value[0]].append(value[1])
                else:
                    items[value[0]] = value[1]
            elif len(value) == 1:
                items.append(value[0])
            else:
                raise ValueError("Did not understand input")
        setattr(namespace, self.dest, items)


class DelimiterSplitAction(argparse.Action):
    """Class to extend the argparse.Action to handle inputs which need to be split with
    with a provided delimiter
    """
    def __call__(self, parser, namespace, values, option_string=None):
        import sys

        args = np.array(sys.argv[1:])
        cond1 = "--delimiter" in args
        cond2 = False
        if cond1:
            cond2 = (
                float(np.argwhere(args == "--delimiter"))
                > float(np.argwhere(args == self.option_strings[0]))
            )
        if cond1 and cond2:
            raise ValueError(
                "Please provide the '--delimiter' command line argument "
                "before the '{}' argument".format(self.option_strings[0])
            )
        delimiter = namespace.delimiter
        items = {}
        for value in values:
            value = value.split(delimiter)
            if len(value) > 2:
                raise ValueError(
                    "'{}' appears multiple times. Please choose a different "
                    "delimiter".format(delimiter)
                )
            if value[0] in items.keys() and not isinstance(items[value[0]], list):
                items[value[0]] = [items[value[0]]]
            if value[0] in items.keys():
                items[value[0]].append(value[1])
            elif len(value) == 1 and len(values) == 1:
                items = [value[0]]
            else:
                items[value[0]] = value[1]
        setattr(namespace, self.dest, items)
