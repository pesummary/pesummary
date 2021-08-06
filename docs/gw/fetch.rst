=============
Fetching data
=============

Public data
+++++++++++

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

.. autofunction:: pesummary.gw.fetch.fetch_open_samples

We may also download publicly available strain data with,

.. literalinclude:: ../../../examples/gw/fetch_open_strain.py
    :language: python
    :linenos:

.. autofunction:: pesummary.gw.fetch.fetch_open_strain

Authenticated data
++++++++++++++++++

You may also want to download LIGO/Virgo authenticated posterior samples. This
can be done with the following,

.. code-block:: python

    >>> from pesummary.gw.fetch import fetch
    >>> data = fetch(URL)
    Enter username for login.ligo.org: albert.einstein
    Enter password for 'albert.einstein' on login.ligo.org:

.. autofunction:: pesummary.gw.fetch.fetch
