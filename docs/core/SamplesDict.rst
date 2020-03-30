=================
SamplesDict class
=================

`pesummary` handles the posterior samples through a custom `SamplesDict` class.
This `SamplesDict` class is inherited from the core `dict` class and includes
extra properties for returning things like the maximum likelihood samples,
median for each marginalized posterior, number of samples stored etc. The
marginalized posterior is stored in a `pesummary.utils.utils.Array` object. For
details about this class see `Array class <Array.html>`_

Initializing the SamplesDict class
----------------------------------

The SamplesDict class is initialized with an array of parameters and an array
of samples for each parameter,

.. code-block:: python

    >>> from pesummary.utils.utils import SamplesDict
    >>> parameters = ["a", "b"]
    >>> samples = [[1,2,3,4,5,6], [6,5,4,3,2,1]]
    >>> samplesdict = SamplesDict(parameters, samples)
    >>> print(samplesdict)
    idx     a              b
    0       1.000000       6.000000
    1       2.000000       5.000000
    2       3.000000       4.000000
    3       4.000000       3.000000
    4       5.000000       2.000000
    5       6.000000       1.000000


Using the SamplesDict properties
--------------------------------

Below we show some of the useful properties of the `SamplesDict` class. For
full details see the doc string,

.. code-block:: python

    >>> samplesdict.minimum
    {'a': Array([1]), 'b': Array([1])}
    >>> samplesdict.maximum
    {'a': Array([6]), 'b': Array([6])}
    >>> samplesdict.median
    {'a': Array([3.5]), 'b': Array([3.5])}
    >>> samplesdict.mean
    {'a': Array([3.5]), 'b': Array([3.5])}
    >>> samplesdict.number_of_samples
    6

Using the SamplesDict functions
-------------------------------

The `SamplesDict` class also has functions to try and make it easier to
manipulate the stored posterior samples. These functions include conversions
to other data structures, downsampling and even removing a set of posterior
samples,

.. code-block:: python

    >>> numpy_structured_array = samplesdict.to_structured_array()
    >>> numpy_structured_array
    rec.array([(1., 6.), (2., 5.), (3., 4.), (4., 3.), (5., 2.), (6., 1.)],
          dtype=[('a', '<f8'), ('b', '<f8')])
    >>> pandas_dataframe = samplesdict.to_pandas()
    >>> pandas_dataframe
    
    >>> downsample = samplesdict.downsample(2)
    >>> downsample
    {'a': Array([1, 6]), 'b': Array([6, 1])}
    >>> discard = samplesdict.discard_samples(3)
    >>> discard
    {'a': Array([4, 5, 6]), 'b': Array([3, 2, 1])}