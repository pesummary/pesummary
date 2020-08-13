=========================
Plotting from a meta file
=========================

Here we will go through step by step how to generate GW specific plots directly
from the `pesummary` meta file. This tutorial builds upon the
`core plotting from a meta file tutorial <../../core/tutorials/plotting_from_metafile.html>`_, so please take a look at the core functionality there first.

Firstly, let us download a GW specific result file,

.. code-block:: bash

    $ curl https://dcc.ligo.org/public/0163/P190412/009/posterior_samples.h5 -o GW190412_posterior_samples.h5

Now, lets make several plots showing the data. The list of available plots can
be displayed by running:

.. code-block:: python

    >>> from pesummary.io import read
    >>> f = read("GW190412_posterior_samples.h5")
    >>> samples = f.samples_dict
    >>> type(samples)
    <class 'pesummary.utils.samples_dict.MultiAnalysisSamplesDict'>
    >>> samples.available_plots

Now, lets make a histogram showing the posterior distribution for `mass_1` for a
subset of the analyses stored in the metafile,

.. code-block:: python

    >>> labels_of_interest = ["IMRPhenomPv3HM", "SEOBNRv4PHM", "combined"]
    >>> fig = samples.plot("mass_1", type="hist", kde=True, labels=labels_of_interest)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisHistogram.png

Alternatively, if you prefer to see this displayed as a violin plot,

.. code-block:: python

    >>> fig = samples.plot("mass_1", type="violin", palette="colorblind", labels=labels_of_interest)
    >>> fig.show()

.. image:: ./examples/violin.png

To see how the prior can also be added to this plot see `Violin plots <../violin.html>`_

Alternatively, a comparison corner plot can be generated for a subset of
parameters,

.. code-block:: python

    >>> fig = samples.plot(type="corner", parameters=["mass_1", "mass_2", "a_1"], labels=labels_of_interest)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisCorner.png

A triangle plot showing the posterior distributions for `mass_1` and `mass_2`
can also be generated with:

.. code-block:: python

    >>> fig, _, _, _ = samples.plot(["mass_1", "mass_2"], type="triangle", smooth=4, fill_alpha=0.2, labels=labels_of_interest)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisTriangle.png

Or the reverse triangle plot for `mass_1` and `mass_2` can be generated with:

.. code-block:: python

    >>> fig, _, _, _ = samples.plot(["mass_1", "mass_2"], type="reverse_triangle", smooth=4, fill_alpha=0.2, labels=labels_of_interest)
    >>> fig.show()

.. image:: ./examples/MultiAnalysisReverseTriangle.png

All of these plots are generated with the `MultiAnalysisSamplesDict` class. Of
course, the `SamplesDict` class can also be used for plotting:

.. code-block:: python

    >>> one = samples["combined"]
    >>> fig = one.plot("mass_1", type="hist", kde=True)
    >>> fig.show()

.. image:: ./examples/Histogram.png

Which shows additional information.

As this is a gravitational wave result file, we may also generate plots which
show the spin distribution of each component. This can be generated with,

.. code-block:: python

    >>> fig = one.plot(type="spin_disk", annotate=True, show_label=True, cmap="Blues")
    >>> fig.show()

.. image:: ./examples/spin_disk.png

We may also generate plots showing the PSD used in the analysis,

.. code-block:: python

    >>> psd = f.psd["combined"]
    >>> fig = psd.plot(fmin=20)
    >>> fig.show()

.. image:: ./examples/PSD.png
		    
