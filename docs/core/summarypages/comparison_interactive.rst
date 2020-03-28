===============================
The Comparison Interactive page
===============================

PESummary is able to produce interactive plots using the `plotly`_ Python
package.

.. _plotly: https://plot.ly/python/

By default, these plots will be generated (however, this can be turned off by
passing the `--disable_interactive`)

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
