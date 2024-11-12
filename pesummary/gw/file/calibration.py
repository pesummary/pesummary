# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from scipy.interpolate import interp1d
from pesummary import conf
from pesummary.utils.utils import check_file_exists_and_rename, logger
from pesummary.utils.dict import Dict

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def _apply_calibration_correction(array=None, amplitude=None, phase=None, type="data"):
    """Apply the calibration correction based on the calibration type. If
    type='data', the amplitude is inverted and phase multiplied by -1. If
    type='template', the amplitude and phase is left unchanged.

    Parameters
    ----------
    array: np.ndarray, optional
        an opened calibration envelope file (with e.g. np.genfromtxt). Columns must
        be "Frequency", "Median Mag", "Phase (Rad)", "-1 Sigma Mag", "-1 Sigma Phase",
        "+1 Sigma Mag", "+1 Sigma Phase"
    amplitude: np.ndarray, optional
        array of only calibration amplitudes
    phase: np.ndarray, optional
        array of only calibration phases
    type: str, optional
        type of calibration data. Must be either 'data' or 'template'
    """
    if type not in ["data", "template"]:
        raise ValueError(f"Unknown calibration type: {type}")
    if all(_ is None for _ in [array, amplitude, phase]):
        raise ValueError(
            "Please provide either an opened calibration file, or amplitude and phase "
            "corrections"
        )
    elif array is None and any(_ is None for _ in [amplitude, phase]):
        raise ValueError(
            "Please provide data for both the amplitude and phase corrections"
        )
    elif array is not None and any(_ is not None for _ in [amplitude, phase]):
        logger.warning(
            "An opened calibration file and amplitude/phase corrections are provided. "
            "Using opened calibration file"
        )
    if array is not None:
        amplitude = np.array([array.T[1], array.T[3], array.T[5]])
        phase = np.array([array.T[2], array.T[4], array.T[6]])
    if type == "data":
        if amplitude is not None:
            amplitude = 1. / amplitude
        if phase is not None:
            phase *= -1
    return amplitude, phase


def _spline_angle_xform(delta_psi):
    """Returns the angle in degrees corresponding to the spline
    calibration parameters delta_psi. Code taken from lalinference.bayespputils

    Parameters
    ----------
    delta_psi: array
        calibration phase uncertainty
    """
    rotation = (2.0 + 1.0j * delta_psi) / (2.0 - 1.0j * delta_psi)
    return 180.0 / np.pi * np.arctan2(np.imag(rotation), np.real(rotation))


def _interpolate_spline_model(
    frequencies, data, interpolated_frequencies, nfreqs=100, xform=None,
    level=0.9, pbar=None
):
    """Interpolate calibration posterior estimates for a spline model in log
    space. Code based upon same function in lalinference.bayespputils

    Parameters
    ----------
    frequencies: array
        The spline control points
    data: ndarray
        Array of posterior samples at each spline control point
    interpolated_frequencies: array
        Array of frequencies you wish to evaluate the interpolant for
    nfreqs: int, optional
        Number of points to evaluate the interpolates spline. Default 100
    xform: func, optional
        Function to transform the spline
    """
    interpolated_data = np.zeros((np.asarray(data).shape[0], nfreqs))
    for num, samp in enumerate(data):
        interp = interp1d(
            frequencies, samp, kind="cubic", fill_value=0., bounds_error=False
        )(interpolated_frequencies)
        if xform is not None:
            interp = xform(interp)
        interpolated_data[num] = interp
        if pbar is not None:
            pbar.update(1)

    mean = np.mean(interpolated_data, axis=0)
    lower = np.quantile(interpolated_data, (1 - level) / 2., axis=0)
    upper = np.quantile(interpolated_data, (1 + level) / 2., axis=0)
    return mean, lower, upper


def interpolate_calibration_posterior_from_samples(
    log_frequencies, amplitudes, phases, nfreqs=100, level=0.9, **kwargs
):
    """Interpolate calibration posterior estimates for a spline model in log
    space and return the amplitude and phase uncertainties. Code based upon same
    function in lalinference.bayespputils

    Parameters
    ----------
    log_frequencies: array
        The spline control points.
    amplitudes: ndarray
        Array of amplitude posterior samples at each of the spline control
        points
    phases: ndarray
        Array of phase posterior samples at each of the spline control points
    nfreqs: int, optional
        Number of points to evaluate the interpolates spline. Default 100
    **kwargs: dict
        All kwargs passed to _interpolate_spline_model
    """
    frequencies = np.exp(log_frequencies)
    interpolated_frequencies = np.logspace(
        np.min(log_frequencies), np.max(log_frequencies), nfreqs, base=np.e
    )
    amp_mean, amp_lower, amp_upper = (
        1 + np.array(
            _interpolate_spline_model(
                log_frequencies, np.column_stack(amplitudes),
                np.log(interpolated_frequencies), nfreqs=nfreqs, xform=None,
                level=level, **kwargs
            )
        )
    )
    phase_mean, phase_lower, phase_upper = np.array(
        _interpolate_spline_model(
            log_frequencies, np.column_stack(phases),
            np.log(interpolated_frequencies), nfreqs=nfreqs,
            xform=_spline_angle_xform, level=level, **kwargs
        )
    ) * (np.pi / 180)
    return np.column_stack(
        [
            interpolated_frequencies, amp_mean, phase_mean, amp_lower,
            phase_lower, amp_upper, phase_upper
        ]
    )


class CalibrationDict(Dict):
    """Class to handle a dictionary of calibration data

    Parameters
    ----------
    detectors: list
        list of detectors
    data: nd list
        list of calibration samples for each detector. Each of the columns
        should represent Frequency, Median Mag, Phase (Rad), -1 Sigma Mag,
        -1 Sigma Phase, +1 Sigma Mag, +1 Sigma Phase

    Attributes
    ----------
    detectors: list
        list of detectors stored in the dictionary

    Methods
    -------
    plot:
        Generate a plot based on the calibration samples stored
    """
    def __init__(self, *args):
        _columns = [
            "frequencies", "magnitude", "phase", "magnitude_lower",
            "phase_lower", "magnitude_upper", "phase_upper"
        ]
        super(CalibrationDict, self).__init__(
            *args, value_class=Calibration, value_columns=_columns
        )

    @property
    def detectors(self):
        return list(self.keys())


class Calibration(np.ndarray):
    """Class to handle Calibration data
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape[1] != 7:
            raise ValueError(
                "Invalid input data. See the docs for instructions"
            )
        return obj

    @classmethod
    def read(cls, path_to_file, IFO=None, type="data", **kwargs):
        """Read in a file and initialize the Calibration class

        Parameters
        ----------
        path_to_file: str
            the path to the file you wish to load
        IFO: str, optional
            name of the IFO which relates to the input file
        type: str, optional
            the calibration definition. This can either be 'data' or 'template'
            and determines whether the calibration is applied to the data
            or the template. Default: 'data'
        **kwargs: dict
            all kwargs are passed to the np.genfromtxt method
        """
        try:
            f = np.genfromtxt(path_to_file, **kwargs)
            amplitudes, phases = _apply_calibration_correction(array=f, type=type)
            data = np.array([
                f.T[0], amplitudes[0], phases[0], amplitudes[1], phases[1],
                amplitudes[2], phases[2]
            ]).T
            return cls(data)
        except Exception:
            raise

    @classmethod
    def from_spline_posterior_samples(
        cls, log_frequencies, amplitudes, phases, **kwargs
    ):
        """Interpolate calibration posterior estimates for a spline model in log
        space and initialize the Calibration class

        Parameters
        ----------
        """
        samples = interpolate_calibration_posterior_from_samples(
            log_frequencies, amplitudes, phases, level=0.68, nfreqs=300,
            **kwargs
        )
        return cls(samples)

    def save_to_file(self, file_name, comments="#", delimiter=conf.delimiter):
        """Save the calibration data to file

        Parameters
        ----------
        file_name: str
            name of the file name that you wish to use
        comments: str, optional
            String that will be prepended to the header and footer strings, to
            mark them as comments. Default is '#'.
        delimiter: str, optional
            String or character separating columns.
        """
        check_file_exists_and_rename(file_name)
        header = [
            "Frequency", "Median Mag", "Phase (Rad)", "-1 Sigma Mag",
            "-1 Sigma Phase", "+1 Sigma Mag", "+1 Sigma Phase"
        ]
        np.savetxt(
            file_name, self, delimiter=delimiter, comments=comments,
            header=delimiter.join(header)
        )

    def __array_finalize__(self, obj):
        if obj is None:
            return
