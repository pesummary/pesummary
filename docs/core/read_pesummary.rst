===============================
Reading in a pesummary metafile
===============================

Although the PESummary metafile can be read in with the core `JSON` or
`h5py` packages, the recommended way of reading in a PESummary metafile is
with the same `pesummary.core.file.read.read` function. For details see

functions that was used in the 
`The read function <read.html>`_ tutorial.

Identifying the label for a specific run
----------------------------------------

Each run stored in the PESummary metafile will have a unique label associated
with it. The list of labels (and consequently the list of runs stored in the
metafile) can be found by running the following:

.. code-block:: python

    >>> from pesummary.gw.file.read import read
    >>> data = read("posterior_samples.json")
    >>> print(data.labels)
    ['EXP1', 'EXP2', EXP3']

Loading the posterior samples for a specific run
------------------------------------------------

If, for example, you wanted to extract the posterior samples for the 'EXP1'
analysis, this can be done with the following:

.. code-block:: python

    >>> print(data.samples_dict['EXP1'])

This will return a `pesummary.utils.utils.SamplesDict` object. For details
about the `SamplesDict` class see `SamplesDict class <SamplesDict.html>`_.
Of course, you may choose to use `astropy` or `pandas` to read in your posterior
samples. For details about this see below.

Loading the configuration file for a specific run
-------------------------------------------------

If passed from the command line, the configuration file is also stored in the
PESummary metafile. You can extract it for a specific run by running:

.. code-block:: python

    >>> config = data.config["EXP1"]
    >>> for i in config.keys():
    ...     print("[{}]".format(i))
    ...     for key, item in config[i].items():
    ...         print("{}={}".format(key, item))
    ...     print("\n")

and you can save it to a file by running:

.. code-block:: python

    >>> data.write_config_to_file("EXP1", outdir="./") 

Converting file format
----------------------

You are able to convert the posterior samples stored in the PESummary metafile
into either a `.dat`, a `bilby .json` or `lalinference .hdf5` or a
`lalinference .dat` file. This can be done by using the following:

.. code-block:: python

    >>> data.to_dat(label="EXP1", outdir="./")
    >>> bilby_objects = data.to_bilby()

Extract posterior samples using `astropy` or `pandas`
-----------------------------------------------------

Of course, you may want to read in the posterior samples into an `astropy`
Table or a `pandas` dataframe. The PESummary metafile stores the data such that
both can be done with ease. To load the posterior samples into an `astropy`
Table, you may use the following:

.. code-block:: python

    >>> from astropy.table import Table
    >>> data = Table.read("posterior_samples.h5", path="EXP1/posterior_samples")

To load in the posterior samples into a pandas dataframe, you may use the
following:

.. code-block:: python

    >>> import pandas as pd
    >>> data = pd.read_hdf("posterior_samples.h5", key="EXP1/posterior_samples")


Loading all data for a specific run without PESummary
-----------------------------------------------------

Of course, you do not have to use PESummary to load in the data from a
PESummary metafile. Below we show how to extract the data using the core
`JSON` and `h5py` python libraries:

.. literalinclude:: ../../../../examples/extract_information_without_pesummary.py
    :language: python
    :linenos:

`pesummary.core.file.formats.pesummary`
---------------------------------------

.. autoclass:: pesummary.core.file.formats.pesummary.PESummary
    :members:
