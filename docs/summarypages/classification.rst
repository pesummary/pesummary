=======================
The Classification page
=======================

PESummary is able to produce source classification probabilies using
`PEPredicates`_ and `pastro`_ packages.

.. _PEPredicates: https://will-farr.docs.ligo.org/pepredicates/
.. _pastro: https://lscsoft.docs.ligo.org/p-astro/

Both probabilities based on the raw samples and probabilities based on samples
that have been reweighted to a population based prior are displayed. An example
is shown below:

.. image:: ./examples/probabilities.png

Bar plot
--------

By default, PESummary produces a bar plot showing the probabilities for each
classification based on the raw samples and reweighted samples. An example
of the bar plot is shown below:

.. image:: ./examples/classification_bar.png

Scatter plot
------------

By default, PESummary produces a scatter plot over the primary source and
secondary source mass parameter space for both the raw samples and reweighted
samples. Each sample is colored by its classification. If unknown, it is
colored black. An example scatter plot is shown below:

.. image:: ./examples/classification_scatter.png
