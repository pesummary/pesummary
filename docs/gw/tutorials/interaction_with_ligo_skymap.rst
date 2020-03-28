==============================
Interacting with `ligo.skymap`
==============================

Here we will go through step by step how to generate a skymap with the
`ligo.skymap` package when you have a PESummary metafile. For details about how
to install the `ligo.skymap` package, see the `ligo.skymap` `documentation`_.

.. _documentation: https://lscsoft.docs.ligo.org/ligo.skymap/quickstart/install.html

A `ligo.skymap` skymap is produced automatically with the `summarypages`
executable if the `ligo.skymap` package is installed. The `fits` file can then
be downloaded directly from the webpages. If, however, you wish to produce the
skymap without running the `summarypages` executable, this can be done by
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
