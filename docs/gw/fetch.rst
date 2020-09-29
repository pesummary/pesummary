====================
Fetching public data
====================

Through `pesummary`'s `pesummary.gw.fetch` module, we can download publicly
available posterior samples. For example, we show how to download the
posterior samples based on GW190412,

.. literalinclude:: ../../../examples/gw/fetch_open_data.py
   :language: python
   :linenos:

This simply downloads the samples from the LIGO/Virgo Document Control Center
and then opens the file with the `pesummary.io.read <read.html>`_ module. For
details about how to plot the data stored in this file, see the
`Plotting from a meta file <./tutorials/plotting_from_metafile.html>`_ tutorial.

.. autofunction:: pesummary.gw.fetch.fetch_open_data
