=====================
MCMCSamplesDict class
=====================

`pesummary` handles multiple MCMC chains through a custom `MCMCSamplesDict`
class. This `MCMCSamplesDict` class is inhertied from the core `dict` class and
stores multiple `SamplesDict` objects, one for each chain. See
`SanplesDict class <SamplesDict.html>`_ for more information. This
`MCMCSamplesDict` class has many useful properties, for example working out the
Gelman-Rubin statistic.

Initializing the MCMCSamplesDict class
--------------------------------------

The `MCMCSamplesDict` class is initialized with an array of parameters and an
array containing the samples for each parameter for each chain,

.. code-block:: python

    >>> from pesummary.utils.samples_dict import MCMCSamplesDict
    >>> parameters = ["a", "b"]
    >>> samples = [
    ...     [
    ...         [1, 1.2, 1.7, 1.1, 1.4, 0.8, 1.6],
    ...         [10.2, 11.3, 11.6, 9.5, 8.6, 10.8, 10.9]
    ...     ], [
    ...         [0.8, 0.5, 1.7, 1.4, 1.2, 1.7, 0.9],
    ...         [10, 10.5, 10.4, 9.6, 8.6, 11.6, 16.2]
    ...     ]
    ... ]
    >>> dataset = MCMCSamplesDict(parameter, samples)
    >>> print(dataset.keys())
    dict_keys(['chain_0', 'chain_1'])
    >>> print(dataset["chain_0"])
    idx     a              b
    0       1.000000       10.200000
    1       1.200000       11.300000
    2       1.700000       11.600000
    3       1.100000       9.500000
    4       1.400000       8.600000
    5       0.800000       10.800000
    6       1.600000       10.900000


Using the MCMCSamplesDict properties
------------------------------------

Below we show some of the useful properties of the `MCMCSamplesDict` class. For
full details see the doc string,

.. code-block:: python

    >>> dataset.nchains
    2
    >>> transpose = dataset.T
    >>> transpose["a"]
    idx     chain_0        chain_1
    0       1.000000       0.800000
    1       1.200000       0.500000
    2       1.700000       1.700000
    3       1.100000       1.400000
    4       1.400000       1.200000
    5       0.800000       1.700000
    6       1.600000       0.900000
    >>> dataset.average
    idx     a              b
    0       0.900000       10.100000
    1       0.850000       10.900000
    2       1.700000       11.000000
    3       1.250000       9.550000
    4       1.300000       8.600000
    5       1.250000       11.200000
    6       1.250000       13.550000


Using the MCMCSamplesDict function
------------------------------------

Below we show some of the useful function of the `MCMCSamplesDict` class. For
full details see the doc string,

.. code-block:: python

    >>> dataset.gelman_rubin("a")
    1.02018
