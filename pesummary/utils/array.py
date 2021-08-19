# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        "standard_deviation", "minimum", "maximum", "maxL", "maxP", "weights",
        "key_data"
    ]

    def __new__(cls, input_array, likelihood=None, prior=None, weights=None):
        obj = np.asarray(input_array).view(cls)
        try:
            obj.standard_deviation = np.std(obj)
            obj.minimum = np.min(obj)
            obj.maximum = np.max(obj)
            obj.maxL = cls._maxL(obj, likelihood)
            obj.maxP = cls._maxP(obj, log_likelihood=likelihood, log_prior=prior)
            obj.weights = weights
            obj.key_data = cls._key_data(obj)
        except Exception:
            obj.standard_deviation = None
            obj.minimum, obj.maximum, obj.maxL = None, None, None
            obj.maxP, obj.key_data = None, {}
        return obj

    def __reduce__(self):
        pickled_state = super(Array, self).__reduce__()
        new_state = pickled_state[2] + tuple(
            [getattr(self, i) for i in self.__slots__]
        )
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.standard_deviation = state[-7]
        self.minimum = state[-6]
        self.maximum = state[-5]
        self.maxL = state[-4]
        self.maxP = state[-3]
        self.weights = state[-2]
        self.key_data = state[-1]
        super(Array, self).__setstate__(state[0:-7])

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

    @staticmethod
    def percentile(array, weights=None, percentile=None):
        """Compute the Nth percentile of a set of weighted samples

        Parameters
        ----------
        array: np.ndarray
            input array
        weights: np.ndarray, optional
            list of weights associated with each sample
        percentile: float, list
            list of percentiles to compute
        """
        if weights is None:
            return np.percentile(array, percentile)

        array, weights = np.array(array), np.array(weights)
        _type = percentile
        if not isinstance(percentile, (list, np.ndarray)):
            percentile = np.array([float(percentile)])
        percentile = np.array([float(i) for i in percentile])
        ind_sorted = np.argsort(array)
        sorted_data = array[ind_sorted]
        sorted_weights = weights[ind_sorted]
        Sn = 100 * sorted_weights.cumsum() / sorted_weights.sum()
        data = np.zeros_like(percentile)
        for num, p in enumerate(percentile):
            inds = np.argwhere(Sn >= p)[0]
            data[num] = np.interp(percentile, Sn[inds], sorted_data[inds])[0]

        if isinstance(_type, (int, float, np.int, np.float64, np.float32)):
            return float(data[0])
        return data

    def confidence_interval(self, percentile=None):
        """Return the confidence interval of the array

        Parameters
        ----------
        percentile: int/list, optional
            Percentile or sequence of percentiles to compute, which must be
            between 0 and 100 inclusive
        """
        if percentile is not None:
            if isinstance(percentile, int):
                return self.percentile(self, self.weights, percentile)
            return np.array(
                [self.percentile(self, self.weights, i) for i in percentile]
            )
        return np.array(
            [self.percentile(self, self.weights, i) for i in [5, 95]]
        )

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.standard_deviation = getattr(obj, 'standard_deviation', None)
        self.minimum = getattr(obj, 'minimum', None)
        self.maximum = getattr(obj, 'maximum', None)
        self.maxL = getattr(obj, 'maxL', None)
        self.maxP = getattr(obj, 'maxP', None)
        self.weights = getattr(obj, 'weights', None)
        self.key_data = getattr(obj, 'key_data', None)
