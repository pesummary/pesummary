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

    >>> from pesummary.utils.samples_dict import SamplesDict
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
    >>> samplesdict = SamplesDict({"a": [1,2,3,4,5,6], "b": [6,5,4,3,2,1]})
    >>> print(samplesdict)
    idx     a              b
    0       1.000000       6.000000
    1       2.000000       5.000000
    2       3.000000       4.000000
    3       4.000000       3.000000
    4       5.000000       2.000000
    5       6.000000       1.000000

Alternatively, it may be initialized with the path to a result file containing
posterior samples,

.. code-block:: python

    >>> samplesdict = SamplesDict.from_file("path_to_file.hdf5")

This `classmethod` simply calls the `read function <read.html>`_ and
initializes the class with the parameters and samples that are already stored.

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
    >>> sliced = samplesdict[:2]
    >>> sliced
    {'a': Array([1, 2]), 'b': Array([6, 5.])}
    >>> pandas_dataframe = samplesdict.to_pandas()
    >>> pandas_dataframe
    
    >>> downsample = samplesdict.downsample(2)
    >>> downsample
    {'a': Array([1, 6]), 'b': Array([6, 1])}
    >>> discard = samplesdict.discard_samples(3)
    >>> discard
    {'a': Array([4, 5, 6]), 'b': Array([3, 2, 1])}

The `SamplesDict` class also provides the ability to plot the posterior samples
directly. This can be achieved through the `plot()` method. For example, if we
want to plot the samples as a KDE, we can run,

.. code-block:: python

    >>> parameter = "a"
    >>> fig = samplesdict.plot(parameter, type="hist", kde=True)
    >>> fig.show()

To see the full list of available plots, you can run:

.. code-block:: python

    >>> samplesdict.available_plots

To see the list of args and kwargs for each plot type, you can run:

.. code-block:: python

    >>> help(samplesdict.plot)
