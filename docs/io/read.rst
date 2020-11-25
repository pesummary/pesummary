===========================
The universal read function
===========================

Builtin functions
-----------------

`pesummay` offers a `read` function which allows for configuration files,
injection files, prior files and nearly result files formats to be read in.
For a full list of result files that can be read in with the
`core <../core/index.html>`_ module see `here <../core/file_formats.html>`_.
For a full list of result files that can be read in with the
`gw <../gw/index.html>`_ module see `here <../gw/file_formats.html>`_.
Below we show a few examples. of how the `read` function works.

First we import the universal read function,

.. code-block:: python

     >>> from pesummary.io import read

A :code:`dat` file containing a set of posterior samples can then be read in
with,

.. code-block:: python

    >>> data = read("example.dat")


Of course, if you would like, you may specify the package that you wish to
use when reading the file. This is done with the :code:`package` kwarg,

.. code-block:: python

    >>> data = read("example.dat", package="core")

For details about the returned posterior samples object, see
`the core package tutorial <../core/read.html>`_ or
`the gw package tutorial <../gw/read.html>`_.

As with the above, a config file can be read in with,

.. code-block:: python

    >>> config = read("config.ini")

This will then return a dictionary containing the section headers and various
settings. If your config file does not have a `ini` extension, this can still
be read in with,

.. code-block:: python

    >>> config = read("config.txt", file_format="ini")

A skymap may also be read in with the following,

.. code-block:: python

     >>> skymap = read("skymap.fits")

Of course, if your skymap does not have the `fits` extension, this skymap
can still be read in with,

.. code-block:: python

    >>> skymap = read("skymap.gz", skymap=True)


Custom functions
----------------

Of course, you might have a file in a format which pesummary is unable to read
in with the inbuilt functions. As a result of the modularity of pesummary, we
may define a class which is capable of reading in this custom file format and
pass it as a kwarg to the universal read function. Below we show a couple of
examples,

.. literalinclude:: ../../../examples/core/single_analysis_custom_read.py
.. literalinclude:: ../../../examples/core/multiple_analysis_custom_read.py 
