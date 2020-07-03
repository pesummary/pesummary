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
