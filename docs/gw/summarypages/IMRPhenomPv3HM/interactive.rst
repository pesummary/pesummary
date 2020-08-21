====================
The Interactive page
====================

PESummary is able to produce interactive plots using the `plotly`_ Python
package.

.. _plotly: https://plot.ly/python/

These interactive plots are displayed on the
`interactive page <https://pesummary.github.io/GW190412/html/IMRPhenomPv3HM_IMRPhenomPv3HM_Interactive_Corner.html>`_. By default, these plots will be generated
(however, this can be turned off by passing the `--disable_interactive`)

Corner plot
-----------
This interactive corner plot has many useful features. The first is custom
selection. Here you are able to select a small region of the a/b parameter
space and see how these points (now colored orange) vary across the full
parameter space. This image can then be saved and sent to collaborators etc.
An example extrinsic corner plot is shown below:

.. raw:: html
   :file: ./examples/interactive_corner_source.html

.. automodule:: pesummary.core.plots.interactive
    :members: corner
