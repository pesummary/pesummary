# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
    __slots__ = ["original", "cls", "added", "removed"]

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self.original = list(*args)
            self.cls = kwargs.get("cls", None)
            self.added = kwargs.get("added", [])
            self.removed = kwargs.get("removed", [])
            super(List, self).__init__(*args)
        else:
            _, self.original, self.cls, self.added, self.removed = args
            super(List, self).__init__(_)

    @property
    def ndim(self):
        return np.array(self).ndim

    def __reduce__(self):
        _slots = [getattr(self, i) for i in self.__slots__]
        slots = [list(self)] + _slots
        return (self.__class__, tuple(slots))

    def __setstate__(self, state):
        _state = state[1]
        self.original = _state["original"]
        self.cls = _state["original"]
        self.added = _state["added"]
        self.removed = _state["removed"]

    def __getitem__(self, *args, **kwargs):
        output = super(List, self).__getitem__(*args, **kwargs)
        if self.cls is None:
            return output
        if isinstance(output, list):
            return [self.cls(value) for value in output]
        else:
            return self.cls(output)

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
