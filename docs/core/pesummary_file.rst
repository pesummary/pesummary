=========================================
Extract information from a PESummary file
=========================================

Identifying the label for a specific run
----------------------------------------

Each run stored in the PESummary metafile will have a unique label associated
with it. The list of labels (and consequently the list of runs stored in the
metafile) can be found by running the following:

.. code-block:: python

    >>> print(data.labels)
    ['EXP1', 'EXP2', 'EXP3']

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

Loading the samples for a specific run
--------------------------------------

Your result file may store either mcmc chains or posterior samples. The
resulting data structures are a bit different with different useful features.
To identify if yours is storing mcmc chains or posterior samples, you may run
the following:

.. code-block:: python

    >>> print(data.mcmc_chains)

If `True`, then please look at `Loading the mcmc samples for a specific run`
below. If `False`, then please look at `Loading the posterior samples for a
specific run` below.

Loading the posterior samples for a specific run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All posterior samples are stored in a
`pesummary.utils.samples_dict.MultiAnalysisSamplesDict`
object. For details about the `MultiAnalysisSamplesDict` class, see
`MultiAnalysisSamplesDict class <MultiAnalysisSamplesDict.html>`_.
If, for example, you wanted to extract the posterior samples for the 'EXP1'
analysis, this can be done with the following:

.. code-block:: python

    >>> print(data.samples_dict['EXP1'])

This will return a `pesummary.utils.samples_dict.SamplesDict` object. For details
about the `SamplesDict` class see `SamplesDict class <SamplesDict.html>`_.
Of course, you may choose to use `astropy` or `pandas` to read in your posterior
samples. For details about this see below.

Loading the mcmc chains for a specific run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wanted to load the mcmc chains stored in the result file, this can be
done with the following:

.. code-block:: python

    >>> print(data.samples_dict)

This will return a `pesummary.utils.samples_dict.MCMCSamplesDict` object. For
details about the `MCMCSamplesDict` class see
`MCMCSamplesDict class <MCMCSamplesDict.html>`_. To load a specific chain, you
may run something like the following:

.. code-block:: python

    >>> print(data.samples_dict["chain_1"])

The data for each chain is stored as a
`pesummary.utils.samples_dict.SamplesDict` object. For details about
the `SamplesDict` class see `SamplesDict class <SamplesDict.html>`_. To see how
many chains are stored in the result file, you may run:

.. code-block:: python

    >>> print(data.samples_dict.nchains)

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

Converting file format
----------------------

You are able to convert the posterior samples stored in the PESummary metafile
into multiple file formats, including a `.dat` and a `bilby .json`. This can be
done by using the following:

.. code-block:: python

    >>> data.write(package="core", file_format="dat", outdir="./", filename="example.dat")
    >>> bilby_objects = data.to_bilby()

Loading all data for a specific run without PESummary
-----------------------------------------------------

Of course, you do not have to use PESummary to load in the data from a
PESummary metafile. The core `JSON` and `h5py` python libraries can be
used. The PESummary file has the following data structure:

.. literalinclude:: ../../../examples/core/pesummary_data_structure.txt
