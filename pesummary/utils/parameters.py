# Copyright (C) 2020 Charlie Hoy <charlie.hoy@ligo.org>
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

from .list import List
from pesummary.gw.file.standard_names import descriptive_names


class Parameter(str):
    """Class to handle a single parameter

    Parameters
    ----------
    parameter: str
        name of the parameter
    description: str, optional
        text describing what parameter represents. Default, look to see if
        pesummary has a descriptive name for parameter else 'Unknown parameter
        description

    Attributes
    ----------
    description: str
        return text describing what parameter represents
    """
    def __new__(self, parameter, **kwargs):
        return super(Parameter, self).__new__(self, parameter)

    def __init__(self, parameter, description=None):
        self._parameter = parameter
        self._description = "Unknown parameter description"
        if description is None:
            if parameter in descriptive_names.keys():
                self._description = descriptive_names[parameter]
        else:
            self._description = description

    @property
    def description(self):
        return self._description

    def __repr__(self):
        return repr(self._parameter)


class Parameters(List):
    """Class to store the list of parameters

    Parameters
    ----------
    args: tuple
        all arguments are passed to the list class
    **kwargs: dict
        all kwargs are passed to the list class

    Attributes
    ----------
    added: list
        list of parameters that have been appended to the original list
    """
    def __init__(self, *args, cls=Parameter, **kwargs):
        if len(args) == 1:
            super(Parameters, self).__init__(*args, cls=Parameter, **kwargs)
        else:
            _args = list(args)
            _args[2] = Parameter
            super(Parameters, self).__init__(*_args, **kwargs)
