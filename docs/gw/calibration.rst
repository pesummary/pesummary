=================
Calibration class
=================

`pesummary` handles calibration data through a custom `Calibration` class. This
`Calibration` class is inherited from the `numpy.ndarray` class. Multiple
calibration envelopes are stored in the `CalibrationDict` class.

Initializing the Calibration class
----------------------------------

The `Calibration` class is initialized with an 7d array. Each of the columns
should represent the Frequency, Median Magnitude, Phase (Rad),
-1 Sigma Magnitude, -1 Sigma Phase, +1 Sigma Magnitude, +1 Sigma Phase,

.. code-block:: python

    >>> from pesummary.gw.file.calibration import Calibration
    >>> import numpy as np
    >>> frequencies = [0, 0.125, 0.25]
    >>> magnitude = [0.5, 0.5, 0.5]
    >>> phase = [0.1, 0.1, 0.1]
    >>> magnitude_lower = magnitude_upper = [0, 0, 0]
    >>> phase_lower = phase_upper = [0, 0, 0]
    >>> calibration_data = np.vstack([magnitude, phase, magnitude_lower, phase_lower, magnitude_upper, phase_upper]).T
    >>> calibration = Calibration(calibration_data)

Using the Calibration object
----------------------------

The `Calibration` object allows for you to save the stored calibration data with
ease. This can be done with the following:

.. code-block:: python

    >>> calibration.save_to_file("new_calibration.dat", delimiter="\t")


Initializing the CalibrationDict class
------------------------------

The `CaibrationDict` class is initialized with a dictionary of 7d array's
containing the Frequency, Median Magnitude, Phase (Rad),
-1 Sigma Magnitude, -1 Sigma Phase, +1 Sigma Magnitude, +1 Sigma Phase
for each detector you are interested in,

.. code-block:: python

    >>> from pesummary.gw.file.calibration import CalibrationDict
    >>> calibration_data = {
    ...     "H1": [[0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
    ...            [0.125, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
    ...            [0.25, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]],
    ...     "V1": [[0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
    ...            [0.125, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
    ...            [0.25, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]]
    ... }
    >>> cal_dict = CalibrationDict(calibration_data)

The data for each detector is stored as a `Calibration` object:

.. code-block:: python

    >>> type(cal_dict["H1"])
    <class 'pesummary.gw.file.calibration.Calibration'>

Using the CalibrationDict object
--------------------------------

The `CalibrationDict` object has extra helper functions and properties to make
it easier for you to extract the information stored within. For example, you can
list the detectors stored with,

.. code-block:: python

    >>> cal_dict.detectors
    ['H1', 'V1']

The frequency and magnitude data for a specific IFO can also be extracted with,

.. code-block:: python

    >>> cal_dict["H1"].frequencies
    array([0., 0.125, 0.25])
    >>> cal_dict["H1"].magnitude
    array([0.5, 0.5, 0.5])

`pesummary.gw.file.calibration.Calibration`
-------------------------------------------

.. autoclass:: pesummary.gw.file.calibration.Calibration
    :members:

`pesummary.gw.file.calibration.CalibrationDict`
-----------------------------------------------

.. autoclass:: pesummary.gw.file.calibration.CalibrationDict
    :members:
