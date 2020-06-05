==============================
Interacting with `ligo.skymap`
==============================

If `ligo.skymap` is installed in your environment and a skymap can be
successfully produced, `pesummary` will store the skymap data in the
`pesummary` metafile when the `summarypages` executable is run. A `fits` file
is also produced and can be downloaded directly from the webpages. For details
about how to install the `ligo.skymap` package, see the
`ligo.skymap` `documentation`_. Below we will go through how this data can be
extracted and a skymap produced. We will also explain how a fits file can be
produced and added to an existing `pesummary` metafile is a skymap is not
already present.

.. _documentation: https://lscsoft.docs.ligo.org/ligo.skymap/quickstart/install.html

Generating a fits file with the `ligo.skymap` executables
---------------------------------------------------------

Here we will go through step by step how to generate a skymap with the
`ligo.skymap` package when you have a PESummary metafile. If, you wish to produce
the skymap without running the `summarypages` executable, this can be done by
following the instructions below.

If your metafile has a single analysis stored, a skymap can be generated with
ease:

.. code-block:: bash

    $ ligo-skymap-from-samples --enable-multiresolution \
                               --samples posterior_samples.h5 \
                               --outdir ./

If your metafile has more than one analysis stored, you will need to select
which analysis you wish to use. This may be done with the `--path` command
line executable,

.. code-block:: bash

    $ ligo-skymap-from-samples --enable-multiresolution \
                               --samples posterior_samples.h5 \
                               --outdir ./ \
                               --path analysis_one/posterior_samples

where `--path` should be `{label}/posterior_samples`.

Appending a fits file to an existing metafile
---------------------------------------------

A `ligo.skymap` fits file may be added to an existing `pesummary` metafile
with the `summarymodify <../../cli/summarymodify.html>`_ executable. An example
command line is below which demonstrates how to add the contents of the fits
file located at `./skymap.fits` to the `posterior_samples.h5` file for the
analysis with label `analysis_one`:

.. code-block:: bash

    $ summarymodify --webdir ./ \
                    --store_skymap analysis_one:./skymap.fits \
                    --samples posterior_samples.h5


Generating a skymap
-------------------

A skymap can be produced easily when the `pesummary` file has been read in
using the `pesummary` package. For details about how to read in a `pesummary`
file with the `pesummary` package, see
`Reading in a pesummary metafile <../read_pesummary.html>`_,

.. literalinclude:: ../../../../examples/gw/ligo_skymap.py
    :language: python
    :linenos:
