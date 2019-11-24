=======================
Calculating Conversions
=======================

For the case of GW specific result files, certain parameters can be derived
given a certain set of samples. For instance, if you have samples for the mass
of the bigger and smaller black holes, then the chirp mass, mass ratio,
total mass and symmetric mass ratio can all be derived.

PESummary boasts a user friendly conversion module that handles all this for
you. It is run with the following,

.. code-block:: python

    >>> from pesummary.gw.file.conversions import _Conversion
    >>> parameters = ["mass_1", "mass_2"]
    >>> samples = [[10, 5], [2, 1], [40, 20]]
    >>> extra_kwargs = {"sampler": {"f_ref": 20}}
    >>> data = _Conversion(parameters, samples, extra_kwargs)
    >>> print(data.keys())
    dict_keys(['mass_1', 'mass_2', 'a_1', 'a_2', 'mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio'])
    >>> print(data)
    idx     mass_1         mass_2         a_1            a_2            mass_ratio     total_mass     chirp_mass     symmetric_mass_ratio
    0       10.000000      5.000000       0.000000       0.000000       0.500000       15.000000      6.083643       0.222222
    1       2.000000       1.000000       0.000000       0.000000       0.500000       3.000000       1.216729       0.222222
    2       40.000000      20.000000      0.000000       0.000000       0.500000       60.000000      24.334574      0.222222

`pesummary.gw.file.conversions`
-------------------------------

.. automodule:: pesummary.gw.file.conversions
    :members:

.. autoclass:: pesummary.gw.file.conversions._Conversion
    :members:
