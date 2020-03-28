===============================
Reading in a pesummary metafile
===============================

Extra information is stored in the `pesummary` metafile generated with the
`gw` module. To access the extra information, the `pesummary.gw.file.read.read`
function must be used. For details see `The read function <read.html>`_. 
`The read function <read.html>`_ tutorial. The
`pesummary.gw.file.formats.pesummary.PESummary` class is inherited from the
`pesummary.core.file.formats.pesummary.PESummary` class and therefore adds
extra functionality on top of the core package. For details about the core
functionality see
`The core pesummary read function <../core/read_pesummary.html>`_.

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

.. literalinclude:: ../../../../examples/extract_information.py
    :language: python
    :linenos:

Loading all data for a specific run without PESummary
-----------------------------------------------------

Of course, you do not have to use PESummary to load in the data from a
PESummary metafile. Below we show how to extract the data using the core
`JSON` and `h5py` python libraries:

.. literalinclude:: ../../../../examples/extract_information_without_pesummary.py
    :language: python
    :linenos:

`pesummary.gw.file.formats.pesummary`
-------------------------------------

.. autoclass:: pesummary.gw.file.formats.pesummary.PESummary
    :members:
