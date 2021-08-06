Strain Data in PESummary
------------------------

The `pesummary.gw.file.strain.StrainData` class
+++++++++++++++++++++++++++++++++++++++++++++++

:code:`pesummary` can read in frame files through the
`universal read function <../io/read.html>`_. :code:`pesummary` stores strain
data in the :code:`pesummary.gw.file.strain.StrainData` class. This object is
inherited from the
`GWPy TimeSeries object <https://gwpy.github.io/docs/stable/timeseries/index.html#the-timeseries>`_ and therefore all `GWPy TimeSeries` methods can be used with this class.
Strain data can be read in directly with the :code:`StrainData` class through the
`read` class method. For example,

.. code-block:: python

    >>> from pesummary.gw.file.strain import StrainData
    >>> f = StrainData.read("frame_file.gwf", channel="channel")

The :code:`StrainData` object extends :code:`gwpy`'s compatible frame file
formats. For instance the :code:`StrainData` object can read in
`Bilby <https://lscsoft.docs.ligo.org/bilby/>`_ :code:`pickle` files which
contain the gravitational wave strain. For example,

.. code-block:: python

    >>> f = StrainData.read("bilby_strain_data.pickle")

The :code:`StrainData` class also offers the :code:`fetch_open_frame` method
which allows the user to fetch frame files from
`GWOSC <https://www.gw-openscience.org/about/>`_ for a given event,

.. code-block:: python

    >>> from pesummary.gw.file.strain import StrainData
    >>> f = StrainData.fetch_open_frame(
    ...    "GW190412", IFO="L1", sampling_rate=4096., duration=32,
    ...    channel="L1:GWOSC-4KHZ_R1_STRAIN"
    ... )

.. autoclass:: pesummary.gw.file.strain.StrainData
    :members:

The `pesummary.gw.file.strain.StrainDataDict` class
+++++++++++++++++++++++++++++++++++++++++++++++++++

If you wish to load numerous frame files from different detectors,
:code:`pesummary` offers the
:code:`pesummary.gw.file.strain.StrainDataDict` class to read in these files.
As with the :code:`StrainData` class, :code:`StrainDataDict` offers a `read`
class method to load a dictionary of frame files.

.. code-block:: python

    >>> from pesummary.gw.file.strain import StrainDataDict
    >>> data = {
    ...     "H1": "./H-H1_LOSC_4_V2-1126257414-4096.gwf",
    ...     "L1": "./L-L1_LOSC_4_V2-1126257414-4096.gwf"
    ... }
    >>> channels = {"H1": "H1:LOSC-STRAIN", "L1": "L1:LOSC-STRAIN"}
    >>> strain = StrainDataDict.read(data, channels=channels)

The output is a dictionary keyed by the IFO with each value being a
:code:`StrainData` object. 

.. autoclass:: pesummary.gw.file.strain.StrainDataDict
    :members:
