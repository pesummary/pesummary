===========
summarypipe
===========

The `summarypipe` executable is designed to generate a `summarypages` command
line with all available options already filled out for you. You simply provide
the run directory of your parameter estimation job, and the `summarypipe`
executable will locate the posterior samples file, configuration file used,
psds, calibration envelopes etc. The command will then be printed to std.out
for your convenience.

To see help for this executable please run:

.. code-block:: console

    $ summarypipe --help

.. program-output:: summarypipe --help
