# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def magnitude_from_vector(vector):
    """Return the magnitude of a vector

    Parameters
    ----------
    vector: list, np.ndarray
        The vector you wish to return the magnitude for.
    """
    vector = np.atleast_2d(vector)
    return np.linalg.norm(vector, axis=1)
