====================
The Conversion class
====================

For the case of GW specific result files, certain parameters can be derived
given a certain set of samples. For instance, if you have samples for the mass
of the bigger and smaller black holes, then the chirp mass, mass ratio,
total mass and symmetric mass ratio can all be derived.

PESummary boasts a user friendly conversion module that handles all this for
you. The user simply passes the parameters and samples either as a dictionary
or a list and the conversion class will calculate all possible derived
quantities.

Initializing the Conversion class
---------------------------------

The `Conversion` class can be initalized as follows,

.. code-block:: python

    >>> from pesummary.gw.file.conversions import _Conversion
    >>> posterior = {"mass_1": [10, 2, 40], "mass_2": [5, 1, 20]}
    >>> data = _Conversion(posterior)
    >>> print(data.keys())
    dict_keys(['mass_1', 'mass_2', 'a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio'])
    >>> print(data.parameters.added)
    ['a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'reference_frequency', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio']
Alternatively,

.. code-block:: python

    >>> from pesummary.gw.file.conversions import _Conversion
    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [[10, 5], [2, 1], [40, 20]]
    >>> data = _Conversion(parameters, samples)
    >>> print(data.keys())
    dict_keys(['mass_1', 'mass_2', 'a_1', 'a_2', 'mass_ratio', 'inverted_mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio'])

Keyword Arguments
-----------------

Extra keyword arguments can be passed to the `Conversion` class to aid in
parameter conversion. These include,

.. autoclass:: pesummary.gw.file.conversions._Conversion

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

.. automodule:: pesummary.gw.file.conversions
    :members:
