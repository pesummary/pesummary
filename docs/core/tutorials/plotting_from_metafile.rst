=========================
Plotting from a meta file
=========================

Here we will go through step by step how to generate plots directly from the
`pesummary` meta file. This tutorial will utilize the plotting methods in the
`pesummary.utils.samples_dict.SamplesDict <../SamplesDict.html>`_ class and the
`pesummary.utils.samples_dict.MultiAnalysisSamplesDict <../MultiAnalysisSamplesDict.html>`_
class. Firstly, let us make a meta file.

.. code-block:: python

    >>> from pesummary.io import write
    >>> import numpy as np
    >>> parameters = ["a", "b", "c", "d"]
    >>> samples = np.array([np.random.normal(10, np.random.random(), 1000) for _ in range(len(parameters))]).T
    >>> write(parameters, samples, outdir="./", label="one", filename="example.h5", file_format="pesummary")

Now, lets make several plots showing the data. The list of available plots can
be displayed by running:

.. code-block:: python

    >>> from pesummary.io import read
    >>> f = read("example.h5")
    >>> samples = f.samples_dict
    >>> type(samples)
    <class 'pesummary.utils.samples_dict.MultiAnalysisSamplesDict'>
    >>> samples.available_plots

Now, lets make a histogram showing the posterior distribution for `a`:

.. code-block:: python

    >>> fig = samples.plot("a", type="hist", kde=True)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisHistogram.png

Alternatively, if you prefer to see this displayed as a violin plot,

.. code-block:: python

    >>> fig = samples.plot("a", type="violin", palette="colorblind", latex_labels={"a": "a"})
    >>> fig.show()

.. image:: ./examples/violin.png

To see how the prior can also be added to this plot see `Violin plots <../../gw/violin.html>`_

Alternatively, a corner plot can be generated for a subset of parameters:

.. code-block:: python

    >>> fig = samples.plot(type="corner", parameters=["a", "b", "c"])
    >>> fig.show()

.. image:: ./examples/MultiAnalysisCorner.png

A triangle plot showing the posterior distributions for `a` and `b` can also be
generated with:

.. code-block:: python

    >>> fig, _, _, _ = samples.plot(["a", "b"], type="triangle", smooth=4, fill_alpha=0.2)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisTriangle.png

Or the reverse triangle plot for `a` and `b` can be generated with:

.. code-block:: python

    >>> fig, _, _, _ = samples.plot(["a", "b"], type="reverse_triangle", smooth=4, fill_alpha=0.2)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisReverseTriangle.png

All of these plots are generated with the `MultiAnalysisSamplesDict` class. Of
course, the `SamplesDict` class can also be used for plotting:

.. code-block:: python

    >>> one = samples["one"]
    >>> fig = one.plot("a", type="hist", kde=True)
    >>> fig.show()

.. image:: ./examples/Histogram.png

Which shows additional information.
