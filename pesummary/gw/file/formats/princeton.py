# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.file.formats.numpy import load

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
# column names taken from https://github.com/jroulet/O2_samples
_O1O2_column_names = [
    'chirp_mass', 'symmetric_mass_ratio', 'spin_1z', 'spin_2z', 'ra', 'dec',
    'psi', 'iota', 'phase', 'geocent_time', 'luminosity_distance'
]
_column_names = {
    "O1O2": _O1O2_column_names
}


def read_princeton(path, type="O1O2", column_names=None, **kwargs):
    """Grab the parameters and samples in a princeton file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    type: str, optional
        the type of file being loaded. This affects how the columns are
        assigned. Default O1O2
    column_names: list, optional
        list of column names corresponding to the loaded numpy array
    """
    if column_names is None and type not in _column_names:
        raise ValueError(
            "Please specify the type of file you are loading. This is used to "
            "make sure that the columns are correctly assigned. The list of "
            "available types are {}".format(", ".join(_column_names.keys()))
        )
    elif column_names is None:
        column_names = _column_names[type]
    return _read_princeton(path, column_names=column_names, **kwargs)


def _read_princeton(path, column_names, **kwargs):
    """Grab the samples from a princeton file

    Parameters
    ----------
    path: str
        path to the result file you wish to read in
    column_names: list
        list of column names corresponding to the loaded numpy array
    """
    samples = load(path)
    if len(column_names) != len(samples[0]):
        raise ValueError(
            "The number of column names does not match the number of columns "
            "in the loaded numpy array. Did you pick the incorrect 'type'?"
        )
    return column_names, samples
