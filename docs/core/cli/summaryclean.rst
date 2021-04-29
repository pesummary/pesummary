============
summaryclean
============

Sometimes, non-physical samples are produced from your sampler and stored in
your result file. In order to fix this, the `summaryclean` executable removes
these non-physical points from your result file, derives other possible
posterior samples (using the `conversion module <../../gw/Conversion.html>`_),
and saves the cleaned data as either a `dat`, `hdf5` or `json` file ready to
use with PESummary.

.. note::
    Most PESummary executables run `summaryclean` by default

To see help for this executable please run:

.. code-block:: console

    $ summaryclean --help

.. program-output:: summaryclean --help

Therefore a biproduct of this executable is to enable the user to switch between
file formats easily from the command line.
