==============
summarydetchar
==============

The `summarydetchar` executable allows the user to generate plots specific
to gravitational wave strain data by interacting with the
`pesummary.gw.plots.detchar` module of PESummary and `GWpy`_. This includes
omegascans and spectrograms.

.. _GWpy: https://gwpy.github.io

To see help for this executable please run:

.. code-block:: console

    $ summarydetchar --help

.. program-output:: summarydetchar --help

`--gwdata`
----------

The gravitational wave strain data can be passed to `summarydetchar` via the
`--gwdata` option. Currently only cache files or bilby pickle files can be
used. If using cache files, you must pass this to `summarydetchar` with the
following syntax

.. code-block:: console

    $ summarydetchar --gwdata DETECTOR:CHANNEL:CACHEFILE

where `DETECTOR` is the name of the detector that the `CACHEFILE` is associated
with and `CHANNEL` is the channel used to collect the strain data. If a single
`bilby` pickle file is used, simply use the following syntax,

.. code-block:: console

    $ summarydetchar --gwdata PICKLEFILE

Generating an Omegascan
-----------------------

An example command line for generating an omegascan given some GW strain data
is shown below:

.. code-block:: console

    $ summarydetchar --gwdata H1:GDS-CALIB_STRAIN:H-H1_HOFT_C00_CACHE.lcf \
                     --window 2 \
                     --gps 1000000 \
                     --vmin 0 \
                     --vmax 25 \
                     --plot omegascan
                     --webdir ./

Generating a Spectrogram
------------------------

An example command line for generating a spectrogram given some GW strain data
is shown below:

.. code-block:: console

    $ summarydetchar --gwdata H1:GDS-CALIB_STRAIN:H-H1_HOFT_C00_CACHE.lcf \
                     --plot spectrogram \
                     --webdir ./


`pesummary.gw.plots.detchar`
----------------------------

.. automodule:: pesummary.gw.plots.detchar
    :members:
