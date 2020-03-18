============================
Reading a PESummary metafile
============================

Although the PESummary metafile can be read in with the core `JSON` or
`h5py` packages, the recommended way of reading in a PESummary metafile is
with the same `pesummary.core.file.read.read` or `pesummary.gw.file.read.read`
functions that was used in the 
`Reading result files <reading_a_result_file.html>`_ tutorial.

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

This will return a `pesummary.utils.utils.SamplesDict` object. To see some of
the functions and properties of this object see the
`Reading result files <reading_a_result_file.html>`_ tutorial. Of course, you
may choose to use `astropy` or `pandas` to read in your posterior samples. For
details about this see below.

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

Loading the calibration envelope for a specific run
---------------------------------------------------

If passed from the command line, the calibration envelope that was used during
the analysis is also stored in the PESummary metafile. You can extract it for
a specific run by running:

.. code-block:: python

    >>> calibration_data = data.priors["calibration"]["EXP1"]
    >>> IFOs = list(calibration_data.keys())
    >>> calibration_envelope = {
    ...     i: np.array(
    ...         [tuple(j) for j in calibration_data[i]], dtype=[
    ...             ("Frequency", "f"),
    ...             ("Median Mag", "f"),
    ...             ("Phase (Rad)", "f"),
    ...             ("-1 Sigma Mag", "f"),
    ...             ("-1 Sigma Phase", "f"),
    ...             ("+1 Sigma Mag", "f"),
    ...             ("+1 Sigma Phase", "f")
    ...         ]
    ...     ) for i in IFOs
    ... }


Loading the psd for a specific run
----------------------------------

If passed from the command line, the psds that were used during the analysis
is also stored in the PESummary metafile. You can extract it for a specific run
by running:

.. code-block:: python

    >>> psd = data.psd["EXP1"]
    >>> IFOs = list(psd.keys())
    >>> psd_data = {
    ...     i: np.array(
    ...         [tuple(j) for j in psd[i]], dtype=[
    ...             ("Frequency", "f"),
    ...             ("Strain", "f")
    ...         ]
    ...     ) for i in IFOs
    ... }


Converting file format
----------------------

You are able to convert the posterior samples stored in the PESummary metafile
into either a `.dat`, a `bilby .json` or `lalinference .hdf5` or a
`lalinference .dat` file. This can be done by using the following:

.. code-block:: python

    >>> data.to_dat(label="EXP1", outdir="./")
    >>> bilby_objects = data.to_bilby()
    >>> data.to_lalinference(outdir="./")
    >>> data.to_lalinference(outdir="./", dat=True)


Example script to extract all information from a PESummary metafile
-------------------------------------------------------------------

An example python script showing how to extract information from the metafile
is shown below:

.. literalinclude:: ../../../examples/extract_information.py
    :language: python
    :linenos:

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

.. literalinclude:: ../../../examples/extract_information_without_pesummary.py
    :language: python
    :linenos:

`pesummary.core.file.formats.pesummary`
---------------------------------------

.. autoclass:: pesummary.core.file.formats.pesummary.PESummary
    :members:

`pesummary.gw.file.formats.pesummary`
-------------------------------------

.. autoclass:: pesummary.gw.file.formats.pesummary.PESummary
    :members:
