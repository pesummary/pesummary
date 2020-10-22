==============================
MultiAnalysisSamplesDict class
==============================

`pesummary` is able to store multiple sets of posterior samples in a custom
`MultiAnalysisSamplesDict` class. This `MultiAnalysisSamplesDict` class is
inherited from the core `dict` class and includes extra properties for returning
comparison statistics between samples. The posterior samples for each analysis
is stored in a `pesummary.utils.samples_dict.SamplesDict` object. For details
about this class see `SamplesDict class <SamplesDict.html>`_

Initializing the MultiAnalysisSamplesDict class
-----------------------------------------------

The `MultiAnalysisSamplesDict` is initialized with either an array of
parameters and samples for each analysis, or a dictionary containing the label
as key, and posterior samples stored as the items,

.. code-block:: python

    >>> from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
    >>> parameters = ["a", "b", "c"]
    >>> samples = [
    ...     [[1,2,3,4,5,6], [6,5,4,3,2,1], [1,3,5,2,4,6]],
    ...     [[1,2,3,4,5,6], [6,5,4,3,2,1], [1,3,5,2,4,6]]
    ... ]
    >>> samplesdict = MultiAnalysisSamplesDict(
    ...     parameters, samples, labels=["analysis_1", "analysis_2"]
    ... )
    >>> print(samplesdict.keys())
    dict_keys(['analysis_1', 'analysis_2'])
    >>> print(samplesdict["analysis_1"])
    idx     a              b              c
    0       1.000000       6.000000       1.000000
    1       2.000000       5.000000       3.000000
    2       3.000000       4.000000       5.000000
    3       4.000000       3.000000       2.000000
    4       5.000000       2.000000       4.000000
    5       6.000000       1.000000       6.000000
    >>> data = {
    ...     "analysis_1": {
    ...         "a": [1,2,3,4,5,6,7,8],
    ...         "b": [8,7,6,5,4,3,2,1],
    ...     },
    ...     "analysis_2": {
    ...         "b": [10,20,30,40,50,60,70,80],
    ...         "c": [1,4,8,2,3,5,6,7]
    ...     }
    ... }
    >>> samplesdict = MultiAnalysisSamplesDict(data)
    >>> print(samplesdict.keys())
    dict_keys(['analysis_1', 'analysis_2'])
    >>> print(samplesdict["analysis_2"])
    idx     b              c
    0       10.000000      1.000000
    1       20.000000      4.000000
    2       30.000000      8.000000
    3       40.000000      2.000000
    .       .              .
    .       .              .
    6       70.000000      6.000000
    7       80.000000      7.000000

Alternatively, it may be initialized with a dictionary containing the path to
multiple result files containing posterior samples and a label for each analysis,

.. code-block:: python

    >>> samplesdict = MultiAnalysisSamplesDict.from_files(
    ...     {
    ...         "analysis_1": "path_to_analysis_1.hdf5",
    ...         "analysis_2": "path_to_analysis_2.dat"
    ...     }
    ... )

This `classmethod` simply calls the `read function <read.html>`_ for each file
and initializes the class with the parameters and samples that are already
stored.

Finally, you may also initialize this class with `SamplesDict <SamplesDict.html>`_
instances,

.. code-block:: python

    >>> from pesummary.io import read
    >>> f = read("path_to_analysis_1.hdf5")
    >>> g = read("path_to_analysis_2.dat")
    >>> samplesdict = MultiAnalysisSamplesDict(
    ...     {"analysis_1": f.samples_dict, "analysis_2": g.samples_dict}
    ... )

After initializing the class, you may select a subset of columns by running:

.. code-block:: python

    >>> samplesdict[["analysis_1", "analysis_2"]]

Using the MultiAnalysisSamplesDict properties
---------------------------------------------

Below we show some of the useful properties of the `MultiAnalysisSamplesDict`
class. For full details see the doc string,

.. code-block:: python

    >>> samplesdict.total_number_of_samples
    16
    >>> samplesdict.minimum_number_of_samples
    8

Using the MultiAnalysisSamplesDict functions
--------------------------------------------

The `MultiAnalysisSamplesDict` class also has functions to try and make it
easier to manipulate and compare the stored analyses,

.. code-block:: python

    >>> samplesdict.js_divergence("b")
    0.55198
    >>> samplesdict.ks_statistic("b")
    0.00016

The `MultiAnalysisSamplesDict` class also provides the ability to plot the
posterior samples directly. This can be achieved through the `plot()` method.
For example, if we want to make a comparison plot, comparing `analysis_1`
and `analysis_2` samples and plot them as a KDE, we can run,

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> parameter = "a"
    >>> fig = samplesdict.plot(parameter, type="hist", labels=["analysis_1", "analysis_2"], kde=True)
    >>> plt.show()

or if we wanted to make a corner plot which compares a subset of the
`analysis_1` and `analysis_2` samples, we can run,

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> parameters = ["a", "b"]
    >>> fig = samplesdict.plot(type="corner", labels=["analysis_1", "analysis_2"], parameters=parameters)
    >>> plt.show()

To see the full list of available plots, you can run:

.. code-block:: python

    >>> samplesdict.available_plots
