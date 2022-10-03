# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.samples_dict import SamplesDict

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Injection(SamplesDict):
    """Class to handle injection information

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each parameter
    """
    def __init__(self, *args, mapping={}, conversion_kwargs={}, **kwargs):
        super(Injection, self).__init__(*args, **kwargs)
        self.update(self.standardize_parameter_names(mapping=mapping))

    @classmethod
    def read(cls, path, mapping={}, num=0, read_kwargs={}, **kwargs):
        """Read an injection file and initalize the Injection class

        Parameters
        ----------
        path: str
            Path to the injection file you wish to read
        num: int, optional
            The row you wish to load. Default is 0
        """
        from pesummary.io import read
        original = read(path, **read_kwargs).samples_dict
        if num is not None:
            import numpy as np
            for key, item in original.items():
                item = np.atleast_1d(item)
                if num <= len(item):
                    original[key] = [item[num]]
                else:
                    raise ValueError(
                        "Unable to extract row {} because the file only has {} "
                        "rows".format(num, len(item))
                    )
        return cls(original, mapping=mapping, **kwargs)

    @property
    def samples_dict(self):
        return self
