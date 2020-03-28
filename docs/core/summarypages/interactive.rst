====================
The Interactive page
====================

PESummary is able to produce interactive plots using the `plotly`_ Python
package.

.. _plotly: https://plot.ly/python/

By default, these plots will be generated (however, this can be turned off by
passing the `--disable_interactive`)

Corner plot
-----------

If the `pesummary.core` module is used, a single corner plot is produced
containing the samples for all parameters in the result file. If the
`pesummary.gw` module is used, two corner plots are produced, one for the
extrinsic parameters, and one for the source parameters.

This interactive corner plot has many useful features. The first is custom
selection. Here you are able to select a small region of the a/b parameter
space and see how these points (now colored orange) vary across the full
parameter space. This image can then be saved and sent to collaborators etc.
An example extrinsic corner plot is shown below:

.. raw:: html
   :file: ./examples/interactive_corner_source.html

.. automodule:: pesummary.core.plots.interactive
    :members: corner

