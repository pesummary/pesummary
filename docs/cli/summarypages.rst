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

Multiple MCMC chains
--------------------

`summarypages` allows for you to pass multiple mcmc chains for a single
analysis. This may be done by passing the `--mcmc_samples` command line
argument. An example command line is shown below:

.. code-block:: console

    $ summarypages --webdir ./mcmc --samples chain1.hdf5 chain2.dat chain2.json \
                   --labels one --mcmc_chains

When this is run, a single plot for each parameter is generated which shows the
posterior distribution for each chain. The Gelman-Rubin statistic is then
printed at the top of each plot to measure convergence. If you have already
generated a summarypage comparing 10 seperate chains, you may then select
a subset of chains to compare by rerunning with the `--compare` option:

.. code-block:: console

    $ summarypages --webdir ./select \
                   --samples ./mcmc/samples/posterior_samples.h5 \
                   --compare chain_0 chain_1

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
