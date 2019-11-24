=============
summaryreview
=============

The `summaryreview` executable allows the user to check that the outputs from
the `pesummary.gw` module agrees with the `cbcBayesPostProc`_ executable
provided from the `LALSuite`_ package. Once the `summaryreview` executable has
run, a single html page is generated containing the plots from both
PESummary and `cbcBayesPostProc`_.

.. _cbcBayesPostProc: https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/python/cbcBayesPostProc.py
.. _LALSuite: https://git.ligo.org/lscsoft/lalsuite


To see help for this executable please run:

.. code-block:: console

    $ summaryreview --help

.. program-output:: summaryreview --help
