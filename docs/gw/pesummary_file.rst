=========================================
Extract information from a pesummary file
=========================================

Alongside the data that is stored in a core `pesummary` metafile
(see the `core docs <../core/pesummary_file.html>`_), the gw
`pesummary` metafile stores additional information, for example the PSD and
calibration data.

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

Extracting strain data
----------------------

If passed from the command line, the gravitational wave strain data used during
the analysis  is also stored in the PESummary metafile. You can extract it by
running:

.. code-block:: python

    >>> strain_data = data.gwdata
    >>> IFOs = strain_data.detectors
    >>> strain = strain_data[IFOs[0]]

Here :code:`strain_data` is a :code:`pesummary.gw.file.strain.StrainDataDict`
object and :code:`strain` is a :code:`pesummary.gw.file.strain.StrainData`
object. For details about these objects see the
`<Strain Data in PESummary <./strain.html>`_ tutorial.

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

As well as converting to same file formats as described in the
`core docs <../core/pesummary_file.html>`_, the `gw` package also allows
for you to convert to a `lalinference .hdf5` or a `lalinference .dat` file.
This can be done by using the following:

.. code-block:: python

    >>> data.to_lalinference(outdir="./")
    >>> data.to_lalinference(outdir="./", dat=True)

Loading all data for a specific run without PESummary
-----------------------------------------------------

Of course, you do not have to use PESummary to load in the data from a
PESummary metafile. The core `JSON` and `h5py` python libraries can be
used. The PESummary file has the following data structure:

.. literalinclude:: ../../../examples/gw/pesummary_data_structure.txt
