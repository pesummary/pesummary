===============================
The Comparison Interactive page
===============================

PESummary is able to produce interactive plots using the
`plotly <https://plot.ly/python>`_ Python package.

Alongside the `interactive plots <../IMRPhenomPv3HM/interactive.html>`_ for a
single analysis, we produce interactive plots to compare multiple result files.
These can be seen on the
`comparison interactive page <https://pesummary.github.io/GW190412/html/Comparison_Interactive_Ridgeline.html>`_.

Ridgeline plot
--------------

If more than one result file is passed, PESummary will produce an interactive
ridgeline plot comparing the posterior distributions for every parameter that is
common to all result files by default. An example ridgeline plot is shown
below:

.. raw:: html
   :file: ./examples/interactive_ridgeline.html

.. automodule:: pesummary.core.plots.interactive
    :members: ridgeline
