====================
Allowed file formats
====================

Alongside the list of file formats that can be read in with the
`core <../core/index.html>`_ module, the `gw` specific module provides
additional functionality for reading in gravitational specific file formats.

GWTC1
-----

As part of the `GWTC-1 <>`_
LIGO-Virgo publication, the posterior samples for the first 11 gravitational
wave candidates was released. For these files, we use the
`pesummary.gw.file.formats.GWTC1.open_GWTC1` function to extract the posterior
samples. If :code:`path_to_samples` is not specified, the `Overall_posterior`
group is used.

.. autofunction:: pesummary.gw.file.formats.GWTC1.open_GWTC1

LALInference
------------

For the :code:`lalinference` file format, we use the
`pesummary.gw.file.formats.lalinference.open_lalinference` function which calls
the `pesummary.core.file.formats.hdf5.open_hdf5` function to extract the
posterior samples. As part of the `open_lalinference` function, We also extract
attributes and file versions from this file.

.. autofunction:: pesummary.gw.file.formats.lalinference.open_lalinference

bilby
-----

A file produced by the gravitational wave module in the
`bilby <https://lscsoft.docs.ligo.org/bilby/>`_ parameter
estimation code, is read in through the
`pesummary.gw.file.formats.bilby.read_bilby` function. This function calls
the `pesummary.core.file.formats.bilby.read_bilby` function (as explained
`here <../core/file_formats.html>`_) to extract the posterior samples.
Posterior samples are then extracted through the `.posterior` property.

.. autofunction:: pesummary.gw.file.formats.bilby.read_bilby
