===========
summarypipe
===========

The `summarypipe` executable is designed to compose a `summarypages` command line
given the run directory which was used by your sampler. It therefore searches
the supplied run directory for a) a result file, b) a configuration file, c)
any PSDs that were used, d) any calibration envelopes that were used etc... and prints
a `summarypages` executable to :code:`std.out` ready for you to run.

To see help for this executable please run:

.. code-block:: console

    $ summarypipe --help

.. program-output:: summarypipe --help
