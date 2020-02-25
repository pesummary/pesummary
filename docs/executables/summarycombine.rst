==============
summarycombine
==============

The `summarycombine` executable combines multiple result files from multiple
analyses into a single PESummary metafile. As with the `summarypages` executable
you are able to pass configuration files, injections files, psds and calibration
envelopes and all this information will also be stored in the metafile.

You are also able to pass 2 or more PESummary metafiles, and the data stored in each
will be combined into a single PESummary metafile.

To see help for this executable please run:

.. code-block:: console

    $ summarycombine --help

.. program-output:: summarycombine --help
