# Licensed under an MIT style license -- see LICENSE.md

from scipy.stats._distn_infrastructure import rv_continuous, rv_sample
from scipy.interpolate import interp1d, interp2d
import numpy as np
from pesummary.utils.array import Array

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class InterpolatedPDF(rv_continuous):
    """Subclass of the scipy.stats._distn_infrastructure.rv_continous class.
    This class handles interpolated PDFs

    Attributes
    ----------
    interpolant: func
        the interpolant to use when evaluate probabilities

    Methods
    -------
    pdf:
        Evaluate the interpolant for a given input and return the PDF at that
        input
    """
    def __new__(cls, *args, **kwargs):
        return super(InterpolatedPDF, cls).__new__(cls)

    def __init__(self, *args, interpolant=None, dimensionality=1, **kwargs):
        self.interpolant = interpolant
        self.dimensionality = dimensionality
        if self.dimensionality > 2:
            import warnings
            warnings.warn(
                "The .rvs() method will currently only work for dimensionalities "
                "< 3"
            )
        super(InterpolatedPDF, self).__init__(*args, **kwargs)

    def _pdf(self, x):
        return self.interpolant(x)

    def rvs(self, size=None, N=10**6):
        if size is None:
            return
        if self.dimensionality > 1:
            raise ValueError("method only valid for one dimensional data")
        _xrange = np.linspace(
            self.interpolant.x[0], self.interpolant.x[-1], N
        )
        args = _xrange
        # if self.dimensionality > 1:
        #    _yrange = np.linspace(
        #        self.interpolant.y[0], self.interpolant.y[-1], N
        #    )
        #    args = [args, _yrange]

        probs = self.interpolant(args).flatten()
        inds = np.random.choice(
            np.arange(N**self.dimensionality), p=probs / np.sum(probs),
            size=size
        )
        # if self.dimensionality > 1:
        #    row = inds // 10
        #    column = inds % 10
        #    return np.array([_xrange[row], _yrange[columns]]).T
        return args[inds]


class DiscretePDF(rv_sample):
    """Subclass of the scipy.stats._distn_infrastructure.rv_sample class. This
    class handles discrete probabilities.

    Parameters
    ----------
    args: list, optional
        2d list of length 2. First element integers, second element probabilities
        corresponding to integers.
    values: list, optional
        2d list of length 2. First element integers, second element probabilities
        corresponding to integers.
    **kwargs: dict, optional
        all additional kwargs passed to the
        scipy.stats._distn_infrastructure.rv_sample class

    Attributes
    ----------
    x: np.ndarray
        array of integers with corresponding probabilities
    probs: np.ndarray
        array of probabilities corresponding to the array of integers

    Methods
    -------
    interpolate:
        interpolate the discrete probabilities and return a continuous
        InterpolatedPDF object
    percentile:
        calculate a percentile from the discrete PDF
    write:
        write the discrete PDF to file using the pesummary.io.write module

    Examples
    --------
    >>> from pesummary.utils.pdf import DiscretePDF
    >>> numbers = [1, 2, 3, 4]
    >>> distribution = [0.1, 0.2, 0.3, 0.4]
    >>> probs = DiscretePDF(numbers, distribution)
    >>> print(probs.x)
    [1, 2, 3, 4]
    >>> probs = DiscretePDF(values=[numbers, distribution])
    >>> print(probs.x)
    [1, 2, 3, 4]
    """
    def __new__(cls, *args, **kwargs):
        return super(DiscretePDF, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        if len(args):
            try:
                self.x, self.probs = args
            except ValueError:
                self.x, self.probs = list(args)[0]
            kwargs["values"] = [self.x, self.probs]
            args = ()
        else:
            self._values = kwargs.get("values", None)
            self.x = self._values[0] if self._values is not None else None
            self.probs = self._values[1] if self._values is not None else None
        super(DiscretePDF, self).__init__(*args, **kwargs)

    def _pmf(self, x):
        _x = np.atleast_1d(x)
        if any(value not in self.x for value in _x):
            raise ValueError(
                "Unable to compute PMF for some of the provided values as "
                "provided probabilities are discrete. Either choose a value "
                "from {} or interpolate the data with "
                "`.interpolate().pdf({}).`".format(np.array(self.x), np.array(x))
            )
        return super(DiscretePDF, self)._pmf(x)

    def rvs(self, *args, **kwargs):
        return Array(super(DiscretePDF, self).rvs(*args, **kwargs))

    def interpolate(self, interpolant=interp1d, include_bounds=True):
        """Interpolate the discrete pdf and return an InterpolatedPDF object

        Parameters
        ----------
        interpolant: func
            function to use to interpolate the discrete pdf
        include_bounds: Bool, optional
            if True, pass the upper and lower bounds to InterpolatedPDF
        """
        kwargs = {}
        if include_bounds:
            kwargs.update({"a": self.x[0], "b": self.x[-1]})
        return InterpolatedPDF(
            interpolant=interp1d(self.x, self.probs), **kwargs
        )

    def percentile(self, p):
        """Calculate a percentile from the discrete PDF

        Parameters
        ----------
        p: float, list
            percentile/list of percentiles to calculate
        """
        from pesummary.utils.array import Array
        return Array.percentile(self.x, weights=self.probs, percentile=p)

    def write(self, *args, **kwargs):
        """Write the discrete PDF to file using the pesummary.io.write module

        Parameters
        ----------
        *args: tuple
            all args passed to pesummary.io.write function
        **kwargs: dict, optional
            all kwargs passed to the pesummary.io.write function
        """
        from pesummary.io import write
        if "dataset_name" not in kwargs.keys():
            kwargs["dataset_name"] = "discrete_pdf"
        write(
            ["x", "PDF"], np.array([self.x, self.probs]).T, *args, **kwargs
        )


class DiscretePDF2D(object):
    """Class to handle 2D discrete probabilities.

    Parameters
    ----------
    args: list, optional
        2d list of length 3. First element integers for the x axis, second
        element inters for the y axis and third element, the 2d joint
        probability density.

    Attributes
    ----------
    x: np.ndarray
        array of integers for the x axis
    y: np.ndarray
        array of integers for the y axis
    probs: np.ndarray
        2D array of probabilities for the x y join probability density

    Methods
    -------
    marginalize:
        marginalize the 2D joint probability distribution to obtain the
        discrete probability density for x and y. Returns the probabilities as
        as a DiscretePDF2Dplus1D object
    level_from_confidence:
        return the level to use for plt.contour for a given confidence.
    minimum_encompassing_contour_level:
        return the minimum contour level that encompasses a specific point

    Examples
    --------
    >>> from pesummary.utils.pdf import DiscretePDF2D
    >>> x = [1., 2., 3.]
    >>> y = [0.1, 0.2, 0.3]
    >>> probs = [
    ...     [1./9, 1./9, 1./9],
    ...     [1./9, 1./9, 1./9],
    ...     [1./9, 1./9, 1./9]
    ... ]
    >>> pdf = DiscretePDF2D(x, y, probs)
    >>> pdf.x
    [1.0, 2.0, 3.0]
    >>> pdf.y
    [0.1, 0.2, 0.3]
    >>> pdf.probs
    array([[0.11111111, 0.11111111, 0.11111111],
           [0.11111111, 0.11111111, 0.11111111],
           [0.11111111, 0.11111111, 0.11111111]])
    """
    def __init__(self, x, y, probability, **kwargs):
        self.x = x
        self.y = y
        self.dx = np.mean(np.diff(x))
        self.dy = np.mean(np.diff(y))
        self.probs = np.array(probability)
        if not self.probs.ndim == 2:
            raise ValueError("Please provide a 2d array of probabilities")
        if not np.isclose(np.sum(self.probs), 1.):
            raise ValueError("Probabilities do not sum to 1")

    def marginalize(self):
        """Marginalize the 2d probability distribution and return a
        DiscretePDF2Dplus1D object containing the probability distribution
        for x, y and xy
        """
        probs_x = np.sum(self.probs, axis=0) * self.dy
        probs_x /= np.sum(probs_x)
        probs_y = np.sum(self.probs, axis=1) * self.dx
        probs_y /= np.sum(probs_y)
        return DiscretePDF2Dplus1D(
            self.x, self.y, [probs_x, probs_y, self.probs]
        )

    def interpolate(self, interpolant=interp2d):
        """Interpolate the discrete pdf and return an InterpolatedPDF object

        Parameters
        ----------
        interpolant: func
            function to use to interpolate the discrete pdf
        include_bounds: Bool, optional
            if True, pass the upper and lower bounds to InterpolatedPDF
        """
        return InterpolatedPDF(
            interpolant=interp2d(self.x, self.y, self.probs),
            dimensionality=2
        )

    def sort(self, descending=True):
        """Flatten and sort the stored probabilities

        Parameters
        ----------
        descending: Bool, optional
            if True, sort the probabilities in descending order
        """
        _sorted = np.sort(self.probs.flatten())
        if descending:
            return _sorted[::-1]
        return _sorted

    def cdf(self, normalize=True):
        """Return the cumulative distribution function

        Parameters
        ----------
        normalize: Bool, optional
            if True, normalize the cumulative distribution function. Default
            True
        """
        cumsum = np.cumsum(self.sort())
        if normalize:
            cumsum /= np.sum(self.probs)
        return cumsum

    def level_from_confidence(self, confidence):
        """Return the level to use for plt.contour for a given confidence.
        Confidence must be less than 1.

        Parameters
        ----------
        confidence: float/list
            float/list of confidences
        """
        confidence = np.atleast_1d(confidence)
        if any(_c > 1 for _c in confidence):
            raise ValueError("confidence must be less than 1")
        _sorted = self.sort()
        idx = interp1d(
            self.cdf(), np.arange(len(_sorted)), bounds_error=False,
            fill_value=len(_sorted)
        )(confidence)
        level = interp1d(
            np.arange(len(_sorted)), _sorted, bounds_error=False, fill_value=0.
        )(idx)
        try:
            return sorted(level)
        except TypeError:
            return level

    def minimum_encompassing_contour_level(self, x, y, interpolate=False):
        """Return the minimum encompassing contour level that encompasses a
        specific point

        Parameters
        ----------
        point: tuple
            the point you wish to find the minimum encompassing contour for
        """
        _sorted = self.sort()
        if interpolate:
            _interp = self.interpolate()
            _idx = _interp.interpolant(x, y)
        else:
            _x = min(
                range(len(self.x)), key=lambda i: abs(self.x[i] - x)
            )
            _y = min(
                range(len(self.y)), key=lambda i: abs(self.y[i] - y)
            )
            _idx = [self.probs[_x, _y]]
        idx = interp1d(
            _sorted[::-1], np.arange(len(_sorted))[::-1], bounds_error=False,
            fill_value=len(_sorted)
        )(_idx)
        level = interp1d(
            np.arange(len(_sorted)), self.cdf(), bounds_error=False, fill_value=1.
        )(idx)
        return level

    def write(self, *args, include_1d=False, **kwargs):
        """Write the discrete PDF to file using the pesummary.io.write module

        Parameters
        ----------
        *args: tuple
            all args passed to pesummary.io.write function
        include_1d: Bool, optional
            if True, save the 1D marginalized as well as the 2D PDF to file
        **kwargs: dict, optional
            all kwargs passed to the pesummary.io.write function
        """
        from pesummary.io import write
        if not include_1d:
            if "dataset_name" not in kwargs.keys():
                kwargs["dataset_name"] = "discrete_pdf"
            X, Y = np.meshgrid(self.x, self.y)
            write(
                ["x", "y", "PDF"],
                np.array([X.ravel(), Y.ravel(), self.probs.flatten()]).T, *args,
                **kwargs
            )
        else:
            _pdf = self.marginalize()
            _pdf.write(*args, **kwargs)


class DiscretePDF2Dplus1D(object):
    """Class to handle 2D discrete probabilities alongside 1D discrete
    probability distributions.

    Parameters
    ----------
    args: list, optional
        3d list of length 3. First element integers for the x axis, second
        element inters for the y axis and third element, a list containing
        the probability distribution for x, y and the 2d join probability
        distribution xy.

    Attributes
    ----------
    x: np.ndarray
        array of integers for the x axis
    y: np.ndarray
        array of integers for the y axis
    probs: np.ndarray
        3D array of probabilities for the x axis, y axis and the xy joint
        probability density
    probs_x: DiscretePDF
        the probability density function for the x axis stored as a
        DiscretePDF object
    probs_y: DiscretePDF
        the probability density function for the y axis stored as a
        DiscretePDF object
    probs_xy: DiscretePDF2D
        the joint probability density function for the x and y axes stored as
        DiscretePDF2D object

    Methods
    -------
    write:
        write the discrete PDF to file using the pesummary.io.write module
    """
    def __init__(self, x, y, probabilities, **kwargs):
        self.x = x
        self.y = y
        self.probs = [np.array(_p) for _p in probabilities]
        if len(self.probs) != 3:
            raise ValueError(
                "Please provide a tuple of length 3. Probabilities for x "
                "y and xy"
            )
        if not any(_p.ndim == 2 for _p in self.probs):
            raise ValueError("Please provide the probabilities for xy")
        if not len([_p for _p in self.probs if _p.ndim == 1]) == 2:
            raise ValueError(
                "2 probabilities array must be 1 dimensional and 1 2 dimensional"
            )
        _x = 0.
        for num, _p in enumerate(self.probs):
            if _p.ndim == 1 and _x == 0:
                self.probs_x = DiscretePDF(self.x, _p)
                self.probs[num] = self.probs_x
                _x = 1.
            elif _p.ndim == 1:
                self.probs_y = DiscretePDF(self.y, _p)
                self.probs[num] = self.probs_y
            else:
                self.probs_xy = DiscretePDF2D(self.x, self.y, _p)
                self.probs[num] = self.probs_xy

    def write(self, *args, **kwargs):
        """Write the 1D and 2D discrete PDF to file using the pesummary.io.write
        module

        Parameters
        ----------
        *args: tuple
            all args passed to pesummary.io.write function
        **kwargs: dict, optional
            all kwargs passed to the pesummary.io.write function
        """
        if "filename" in kwargs.keys() and not isinstance(kwargs["filename"], dict):
            raise ValueError(
                "Please provide filenames as a dictionary with keys '1d' and "
                "'2d'"
            )
        elif "filename" in kwargs.keys():
            if not all(k in ["1d", "2d"] for k in kwargs["filename"].keys()):
                raise ValueError("Filename must be keyed by '1d' and/or '2d'")
        else:
            _format = "dat" if "file_format" not in kwargs.keys() else kwargs[
                "file_format"
            ]
            _default = "pesummary_{}_pdf.%s" % (_format)
            kwargs["filename"] = {
                "1d": _default.format("1d"), "2d": _default.format("2d")
            }
        _filenames = kwargs.pop("filename")
        self.probs_x.write(*args, filename=_filenames["1d"], **kwargs)
        self.probs_xy.write(*args, filename=_filenames["2d"], **kwargs)
