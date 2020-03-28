============
summarypages
============

The `summarypages` executable is the main interface to PESummary, it generates
all plots (`publication <summarypages.html>`_, `detchar <summarydetchar.html>`_,
`classification <summaryclassification.html>`_), and all webpages to display
the plots and a single metafile containing all the data about the analysis. It
is the go to executable for your analysis.

To see help for this executable please run:

.. code-block:: console

    $ summarypages --help

.. program-output:: summarypages --help

For details about each page produced, visit `the summary pages <../index.html>`_
section.

Multiple result files
---------------------

`summarypages` allows for you to pass a single or multiple result files
generated from any parameter estimation code. If a multiplt result files are
passed, then all single plots/pages as well as comparison plots/pages are
generated.

Add to existing
---------------

PESummary also allows for you to add to an existing webpage that has already
been generated using the `summarypages` executable. This is done by using the
`--existing_webdir` flag as opposed to the `--webdir` flag.

Labels
------

Each result file requires a label that is unique. This label is used throughout
the PESummary workflow and is used to index the plots and html pages to ensure
that the plots are associated with the correct result file. If no label is
chosen, then `summarypages` will assign a label for the given result file. This
label is chosen to be `{time}_{filename}` where `time` is the time at which
you submitted the job and filename is the name of the result file.

Passing a PESummary configuration file
--------------------------------------

You are also able to pass `PESummary` a configuration file storing all of the
command line arguments. For instance, you can generate a summarypage by running,

.. code-block:: console

    $ summarypages pesummary.ini

Where the configuration file has the following structure:

.. literalinclude:: ../../../examples/pesummary.ini
   :language: ini
   :linenos:

You are also able to override all commands in the configuration file by also
including them in the command line. For instance, if you run,

.. code-block:: console

    $ summarypages pesummary.ini --webdir ./different_webpage

The webpages will be saved in the directory `./different_webpage`.
