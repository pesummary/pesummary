# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _2DArray(input_array, likelihood=None, prior=None, weights=None):
    """Helper function for initialising multiple Array objects.

    Parameters
    ----------
    input_array: np.ndarray, list
        input list/array
    likelihood: list, optional
        log likelihood samples to use when calculating maximum likelihood
    prior: list, optional
        log prior samples to use when calculating maximum posterior
    weights: list, optional
        weights to use for the samples

    Returns
    -------
    array: list
        list of Array objects of size len(input_array)
    """
    obj = np.atleast_2d(input_array)
    if obj.ndim > 2:
        raise ValueError("Input must be one- or two-dimensional")
    if not obj.shape[-1]:
        standard_deviation, minimum, maximum = [None], [None], [None]
    else:
        standard_deviation = np.std(obj, axis=1)
        minimum = np.min(obj, axis=1)
        maximum = np.max(obj, axis=1)
    try:
        maxL = Array._maxL(obj.T, likelihood=likelihood)
    except Exception:
        maxL = None
    try:
        maxP = Array._maxP(obj.T, log_likelihood=likelihood, log_prior=prior)
    except Exception:
        maxP = None
    return [
        Array(
            _array, minimum=minimum[num], maximum=maximum[num],
            standard_deviation=standard_deviation[num],
            maxL=maxL[num] if maxL is not None else None,
            maxP=maxP[num] if maxP is not None else None, weights=weights
        ) for num, _array in enumerate(obj)
    ]


class Array(np.ndarray):
    """Class to add extra functions and methods to np.ndarray

    Parameters
    ----------
    input_aray: list/array
        input list/array

    Attributes
    ----------
    median: float
        median of the input array
    mean: float
        mean of the input array
    key_data: dict
        dictionary containing the key data associated with the array
    """
    __slots__ = [
        "standard_deviation", "minimum", "maximum", "maxL", "maxP", "weights"
    ]

    def __new__(
        cls, input_array, standard_deviation=None, minimum=None, maximum=None,
        maxL=None, maxP=None, likelihood=None, prior=None,
        weights=None
    ):
        obj = np.asarray(input_array).view(cls)
        obj.weights = weights
        mapping = {
            "standard_deviation": [standard_deviation, np.std, {}],
            "minimum": [minimum, np.min, {}],
            "maximum": [maximum, np.max, {}],
            "maxL": [maxL, cls._maxL, {"likelihood": likelihood}],
            "maxP": [
                maxP, cls._maxP,
                {"log_likelihood": likelihood, "log_prior": prior}
            ],
        }
        for attr, item in mapping.items():
            if item[0] is None:
                try:
                    setattr(obj, attr, item[1](obj, **item[2]))
                except Exception:
                    setattr(obj, attr, None)
            else:
                setattr(obj, attr, item[0])
        return obj

    def __reduce__(self):
        pickled_state = super(Array, self).__reduce__()
        new_state = pickled_state[2] + tuple(
            [getattr(self, i) for i in self.__slots__]
        )
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.standard_deviation = state[-6]
        self.minimum = state[-5]
        self.maximum = state[-4]
        self.maxL = state[-3]
        self.maxP = state[-2]
        self.weights = state[-1]
        super(Array, self).__setstate__(state[0:-6])

    def average(self, type="mean"):
        """Return the average of the array

        Parameters
        ----------
        type: str
            the method to average the array
        """
        if type == "mean":
            return self._mean(self, weights=self.weights)
        elif type == "median":
            return self._median(self, weights=self.weights)
        else:
            return None

    @staticmethod
    def _mean(array, weights=None):
        """Compute the mean from a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        """
        if weights is None:
            return np.mean(array)
        weights = np.array(weights).flatten() / float(sum(weights))
        return float(np.dot(np.array(array), weights))

    @staticmethod
    def _median(array, weights=None):
        """Compute the median from a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        """
        if weights is None:
            return np.median(array)
        return Array.percentile(array, weights=weights, percentile=50)

    @staticmethod
    def _maxL(array, likelihood=None):
        """Return the maximum likelihood value of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        likelihood: np.ndarray, optional
            likelihoods associated with each sample
        """
        if likelihood is not None:
            likelihood = list(likelihood)
            ind = likelihood.index(np.max(likelihood))
            return array[ind]
        return None

    @staticmethod
    def _maxP(array, log_likelihood=None, log_prior=None):
        """Return the maximum posterior value of the array

        Parameters
        ----------
        array: np.ndarray
            input array
        log_likelihood: np.ndarray, optional
            log likelihoods associated with each sample
        log_prior: np.ndarray, optional
            log prior associated with each sample
        """
        if any(param is None for param in [log_likelihood, log_prior]):
            return None
        likelihood = np.array(log_likelihood)
        prior = np.array(log_prior)
        posterior = likelihood + prior
        ind = np.argmax(posterior)
        return array[ind]

    def to_dtype(self, _dtype):
        return _dtype(self)

    @property
    def key_data(self):
        return self._key_data(self)

    @staticmethod
    def _key_data(
        array, header=[
            "mean", "median", "std", "maxL", "maxP", "5th percentile",
            "95th percentile"
        ]
    ):
        """Return a dictionary containing the key data associated with the
        array

        Parameters
        ----------
        array: np.ndarray
            input array
        header: list
            list of properties you wish to return
        """
        def safe_dtype_change(array, _dtype):
            if array is not None:
                if isinstance(array, Array):
                    return array.to_dtype(_dtype)
                else:
                    return _dtype(array)
            return array

        mydict = {}
        for key in header:
            if not hasattr(np.ndarray, key):
                try:
                    _value = safe_dtype_change(getattr(array, key), float)
                except AttributeError:
                    if key == "5th percentile":
                        _value = safe_dtype_change(
                            array.confidence_interval(percentile=5), float
                        )
                    elif key == "95th percentile":
                        _value = safe_dtype_change(
                            array.confidence_interval(percentile=95), float
                        )
                    else:
                        _value = safe_dtype_change(
                            array.average(type=key), float
                        )
            else:
                if key == "std":
                    _value = safe_dtype_change(array.standard_deviation, float)
                else:
                    _value = safe_dtype_change(array.average(type=key), float)
            mydict[key] = _value
        return mydict

    def confidence_interval(self, percentile=None):
        """Return the confidence interval of the array

        Parameters
        ----------
        percentile: int/list, optional
            Percentile or sequence of percentiles to compute, which must be
            between 0 and 100 inclusive
        """
        from pesummary.utils.credible_interval import two_sided_credible_interval
        if percentile is None:
            percentile = [5, 95]
        return two_sided_credible_interval(self, percentile, weights=self.weights)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.standard_deviation = getattr(obj, 'standard_deviation', None)
        self.minimum = getattr(obj, 'minimum', None)
        self.maximum = getattr(obj, 'maximum', None)
        self.maxL = getattr(obj, 'maxL', None)
        self.maxP = getattr(obj, 'maxP', None)
        self.weights = getattr(obj, 'weights', None)

    def to_numpy(self):
        """Convert Array object to a numpy.ndarray

        Returns
        -------
        data: np.ndarray/tuple
            return stored data as a np.ndarray. If weights are stored in the
            Array object, return a tuple containing the stored data as a
            np.ndarray and the weights as a np.ndarray
        """
        _array = np.asarray(self)
        if self.weights is None:
            return _array
        return _array, np.asarray(self.weights)
