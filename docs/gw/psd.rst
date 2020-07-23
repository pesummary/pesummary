=========
PSD class
=========

`pesummary` handles PSD data through a custom `PSD` class. This `PSD` class
is inherited from the `numpy.ndarray` class. Multiple PSDs are stored in the
`PSDDict` class.

Initializing the PSD class
--------------------------

The `PSD` class is initialized with an 2d array containing the frequency and
strain data,

.. code-block:: python

    >>> from pesummary.gw.file.psd import PSD
    >>> import numpy as np
    >>> frequencies = [0, 0.125, 0.25]
    >>> strains = [0.25, 0.25, 0.25]
    >>> psd_data = np.vstack([frequencies, strains]).T
    >>> psd = PSD(psd_data)

Alternatively, a file containing the psd can be read in with,

.. code-block:: python

    >>> filename = "./IFO0_psd.dat"
    >>> psd = PSD.read(filename)

Using the PSD object
--------------------

The `PSD` object allows for you to save the stored PSD data with ease. This can
be done with the following:

.. code-block:: python

    >>> psd.save_to_file("new_psd.dat", delimiter="\t")


Initializing the PSDDict class
------------------------------

The `PSDDict` class is initialized with a dictionary of 2d array's containing
the frequency and strain data for each detector you are interested in,

.. code-block:: python

    >>> from pesummary.gw.file.psd import PSDDict
    >>> psd_data = {
    ...     "H1": [[0.00000e+00, 2.50000e-01],
    ...            [1.25000e-01, 2.50000e-01],
    ...            [2.50000e-01, 2.50000e-01]],
    ...     "V1": [[0.00000e+00, 2.50000e-01],
    ...            [1.25000e-01, 2.50000e-01],
    ...            [2.50000e-01, 2.50000e-01]]
    ... }
    >>> psd_dict = PSDDict(psd_data)

The data for each detector is stored as a `PSD` object:

.. code-block:: python

    >>> type(psd_dict["H1"])
    <class 'pesummary.gw.file.psd.PSD'>

If you have multiple psds stored in multiple files, you can simply run the
following,

.. code-block:: python

    >>> psd_H1 = PSD.read("IFO0_psd.dat")
    >>> psd_V1 = PSD.read("IFO2_psd.dat")
    >>> psd_data = {"H1": psd_H1, "V1": psd_V1}
    >>> psd_dict = PSDDict(psd_data)

Using the PSDDict object
------------------------

The `PSDDict` object has extra helper functions and properties to make it easier
for you to extract the information stored within. For example, you can list the
detectors stored with,

.. code-block:: python

    >>> psd_dict.detectors
    ['H1', 'V1']

Or you can simply plot all stored PSDs on a single axis,

.. code-block:: python

    >>> psd_dict.plot()

The frequency and strain data for a specific IFO can also be extracted with,

.. code-block:: python

    >>> psd_dict["H1"].frequencies
    array([0.   , 0.125, 0.25 ])
    >>> psd_dict["H1"].strains
    array([0.25, 0.25, 0.25])

`pesummary.gw.file.psd.PSD`
---------------------------

.. autoclass:: pesummary.gw.file.psd.PSD
    :members:

`pesummary.gw.file.psd.PSDDict`
-------------------------------

.. autoclass:: pesummary.gw.file.psd.PSDDict
    :members:
