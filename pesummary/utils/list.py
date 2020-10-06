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


class List(list):
    """Base list class to extend the core `list` class

    Parameters
    ----------
    *args: tuple
        all arguments are passed to the list class
    **kwargs: dict
        all kwargs are passed to the list class

    Attributes
    ----------
    added: list
        list of values appended to the original list
    """
    __slots__ = ["original", "added", "removed"]

    def __init__(self, *args, **kwargs):
        self.original = list(*args, **kwargs)
        super(List, self).__init__(*args, **kwargs)
        self.added = []
        self.removed = []

    @property
    def ndim(self):
        return np.array(self).ndim

    def __add__(self, *args, **kwargs):
        self.added.extend(*args)
        obj = List(super(List, self).__add__(*args, **kwargs))
        for attr in self.__slots__:
            setattr(obj, attr, getattr(self, attr))
        return obj

    def __iadd__(self, *args, **kwargs):
        self.added.extend(*args)
        obj = List(super(List, self).__iadd__(*args, **kwargs))
        for attr in self.__slots__:
            setattr(obj, attr, getattr(self, attr))
        return obj

    def append(self, *args, **kwargs):
        self.added.append(*args)
        return super(List, self).append(*args, **kwargs)

    def extend(self, *args, **kwargs):
        self.added.extend(*args)
        return super(List, self).extend(*args, **kwargs)

    def insert(self, index, obj, **kwargs):
        self.added.append(obj)
        return super(List, self).insert(index, obj, **kwargs)

    def remove(self, element, **kwargs):
        obj = super(List, self).remove(element, **kwargs)
        self.removed.append(element)
        if element in self.added:
            self.added.remove(element)
        return obj

    def pop(self, index, **kwargs):
        self.removed.append(self[index])
        obj = super(List, self).pop(index)
        return obj
