# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.default import Default
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
    def __init__(self, *args, conversion_kwargs={}, **kwargs):
        super(Injection, self).__init__(*args, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        """Read an injection file and initalize the Injection class

        Parameters
        ----------
        path: str
            Path to the injection file you wish to read
        """
        data = Default.load_file(path)
        return cls(data.samples_dict, **kwargs)

    @property
    def samples_dict(self):
        return self
