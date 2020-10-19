===========
Array class
===========

`pesummary` handles a set of marginalized posterior samples through a custom
`Array` class. This `Array` class is inherited from the `numpy.ndarray` class
and includes extra properties to make it easier to return key information.

Initializing the Array class
----------------------------

The `Array` class is initalized with the following:

.. code-block:: python

    >>> from pesummary.utils.array import Array
    >>> samples = [1,2,3,4,5,6]
    >>> array = Array(samples)

Using the Array properties
--------------------------

Below we show some of the useful properties of the `Array` class. For full
details see the doc string,

.. code-block:: python

    >>> array.minimum
    Array(1)
    >>> array.maximum
    Array(6)
    >>> array.average(type="mean")
    Array(3.5)
    >>> array.average(type="median")
    Array(3.5)
    >>> array.key_data
    {'mean': 3.5, 'median': 3.5, 'std': 1.707825127659933, 'maxL': None, 'maxP': None, '5th percentile': 1.25, '95th percentile': 5.75}

Using the Array functions
-------------------------

Below we show some of the useful functions of the `Array` class,

.. code-block:: python

    >>> array.confidence_interval(percentile=[5, 95])
    array([1.25, 5.75])
    >>> array.confidence_interval(percentile=[45, 55])
    array([3.25, 3.75])
    >>>
