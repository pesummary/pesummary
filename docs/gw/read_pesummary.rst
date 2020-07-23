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

Extracting skymap statistics
----------------------------

If a `ligo.skymap` skymap was successfully produced when the metafile was created,
the skymap statistics (sky localisation at 90% confidence, localization volume
at 90% confidence etc.) are stored in the metadata. These can be extracted by
running:

.. code-block:: python

    >>> ind = data.labels.index("EXP1")
    >>> kwargs = data.extra_kwargs[ind]["other"]
    >>> print(kwargs["area90"])
    '1234.0'

Loading the calibration envelope for a specific run
---------------------------------------------------

If passed from the command line, the calibration envelope that was used during
the analysis is also stored in the PESummary metafile. You can extract it for
a specific run by running:

.. code-block:: python

    >>> calibration_data = data.priors["calibration"]["EXP1"]
    >>> IFOs = calibration_data.detectors
    >>> IFO = IFOs[0]
    >>> frequency = calibration_data[IFO].frequencies
    >>> median_mag = calibration_data[IFO].magnitude
    >>> mag_lower = calibration_data[IFO].magnitude_lower


For more details see the `Calibration class <calibration.html>`_ tutorial.

Loading the psd for a specific run
----------------------------------

If passed from the command line, the psds that were used during the analysis
are also stored in the PESummary metafile. You can extract it for a specific
analysis by running:

.. code-block:: python

    >>> psd = data.psd["EXP1"]
    >>> print(type(psd))
    <class 'pesummary.gw.file.psd.PSDDict'>
    >>> IFOs = psd.detectors
    >>> IFO = IFOs[0]
    >>> frequency = psd[IFO].frequencies
    >>> strains = psd[IFO].strains


For more details see the `PSD class <psd.html>`_ tutorial.


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

Loading all data for a specific run without PESummary
-----------------------------------------------------

Of course, you do not have to use PESummary to load in the data from a
PESummary metafile. The core `JSON` and `h5py` python libraries can be
used. The PESummary file has the following data structure:

.. literalinclude:: ../../../examples/gw/pesummary_data_structure.txt

Below we show how to extract the data using the core `JSON` and `h5py` python
libraries:

.. literalinclude:: ../../../examples/extract_information_without_pesummary.py
    :language: python
    :linenos:

`pesummary.gw.file.formats.pesummary`
-------------------------------------

.. autoclass:: pesummary.gw.file.formats.pesummary.PESummary
    :members:
