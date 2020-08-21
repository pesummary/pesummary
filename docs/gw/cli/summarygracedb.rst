==============
summarygracedb
==============

The `summarygracedb` exeuctable is designed to allow the user to easily access
data from gracedb for a specific event. You simply provide the GraceDB ID,
and the `summarygracedb` executable will return either the full data available
or the data requested with the `--info` command line argument. The data is
printed to stdout by default, however, it can easily be saved to a json file
with the `--output` command line argument.

To see help for this executable please run:

.. code-block:: console

    $ summarygracedb --help

.. program-output:: summarygracedb --help
