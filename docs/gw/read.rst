=================
The read function
=================

`pesummary` provides functionality to read in nearly all result file formats.
This is done with the `pesummary.io.read.read <../io/read.html>`_ function. Below
we show how the `gw` read function is different from the `core` implementation.
For details about the `core` implementation, see the
`core read docs <../core/read.html>`_.

Reading a result file
---------------------

Below, we show how to read in a result file,

.. code-block:: python

    >>> from pesummary.io import read
    >>> data = read("example.dat", package="gw")
    >>> json_load = read("example.json", package="gw")
    >>> hdf5_load = read("example.hdf5", package="gw")
    >>> txt_load = read("example.txt", package="gw")

`pesummary` is able to read in `json`, `dat`, `txt` and `hdf5` file formats.
When the `gw` package is specified, the samples are read in and the parameter
names are converted to a standard naming convention. The definition of these
standard names can be seen `here <parameters.html>`_. This is to allow
the samples from different codes (where different naming conventions are used)
to be compared.

For details about how to read in a Testing General Relativity specific result
file, see the `pesummary TGR file <./tgr_file.html>`_ documentation. 

Converting samples
------------------

`pesummary` provides an extensive conversion suite. For details about this
conversion suite see the `Conversion docs <Conversion.html>`_. We may convert
our existing samples using the `generate_all_posterior_samples` method,

.. code-block:: python

    >>> data.generate_all_posterior_samples()

and we can see which parameters have been derived using the
`converted_parameters` attribute,

.. code-block:: python

    >>> data.converted_parameters
