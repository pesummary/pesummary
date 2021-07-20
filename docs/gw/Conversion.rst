=====================
The Conversion module
=====================

For the case of GW specific result files, certain parameters can be derived
given a certain set of samples. For instance, if you have samples for the mass
of the bigger and smaller black holes, then the chirp mass, mass ratio,
total mass and symmetric mass ratio can all be derived.

PESummary boasts a user friendly conversion module that handles all this for
you. The user simply passes the parameters and samples either as a dictionary
or a list and the conversion class will calculate all possible derived
quantities.

The `convert` function
----------------------

The `convert` function can be initalized as follows,

.. code-block:: python

    >>> from pesummary.gw.conversions import convert
    >>> posterior = {"mass_1": [10, 2, 40], "mass_2": [5, 1, 20]}
    >>> data = convert(posterior)
    >>> print(data.keys())
    dict_keys(['mass_1', 'mass_2', 'a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio'])
    >>> print(data.parameters.added)
    ['a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'reference_frequency', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio']

Alternatively,

.. code-block:: python

    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [[10, 5], [2, 1], [40, 20]]
    >>> data = convert(parameters, samples)
    >>> print(data.keys())
    dict_keys(['mass_1', 'mass_2', 'a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio'])

The `convert` function takes a range of keyword arguments, some of which are
required for certain conversions, and others allow the user to run often
time-consuming conversions. A full list of the kwargs can be seen below,

.. autofunction:: pesummary.gw.conversions.convert

Extra notes on specific conversions
-----------------------------------

Below are seperate pages which goes into extra detail about specific parameter
conversions:

.. toctree::
    :maxdepth: 1

    remnant_fits 

Core conversion functions
-------------------------

Of course, the core conversion functions can be also be used,

Cosmology
+++++++++

Below are conversion functions that are specific to a given cosmology,

.. automodule:: pesummary.gw.conversions.cosmology
    :members:

Evolve
++++++

Below are functions that evolve BBH spins up to a specified value,

.. automodule:: pesummary.gw.conversions.evolve
    :members:

Mass
++++

Below are conversion functions specific to the binaries component masses,

.. automodule:: pesummary.gw.conversions.mass
    :members:

NRUtils
+++++++

Below are a series of helper functions that calculate the remnant properties
of the binary,

.. automodule:: pesummary.gw.conversions.nrutils
    :members:

Remnant
+++++++

Below are conversion functions that calculate the remnant properties of the
binary,

.. automodule:: pesummary.gw.conversions.remnant
    :members:

SNR
+++

Below are functions to calculate conversions specific to the signal-to-noise
ratio (SNR),

.. automodule:: pesummary.gw.conversions.snr
    :members:

Spins
+++++

Below are functions that deal with the binaries spins,

.. automodule:: pesummary.gw.conversions.spins
    :members:

Tidal
+++++

Below are functions that are specific to objects that experience tidal forces,

.. automodule:: pesummary.gw.conversions.tidal
    :members:

Time
++++

Below are functions that deal with the event time,

.. automodule:: pesummary.gw.conversions.time
    :members:

TGR
+++

Below are functions that generate parameters required for testing General
Relativity

.. automodule:: pesummary.gw.conversions.tgr
    :members:
