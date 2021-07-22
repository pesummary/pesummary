Welcome to PESummary's core package
===================================

PESummary's core package provides all of the necessary code for analysing,
displaying and comparing data files from general inference problems.

Unified input/output
++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    write
    read

Manipulating posterior samples
++++++++++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    Array
    SamplesDict
    MCMCSamplesDict
    MultiAnalysisSamplesDict

Manipulating PDFs
+++++++++++++++++

.. toctree::
    :maxdepth: 1

    pdf
    ProbabilityDict
    ProbabilityDict2D

The pesummary metafile
++++++++++++++++++++++

.. toctree::
    :maxdepth: 2

    pesummary_file

Plotting
++++++++

.. toctree::
    :maxdepth: 1

    seaborn
    bounded_kdes
    tutorials/plotting_from_metafile

Executables
+++++++++++

PESummary boasts several executables to make it even easier to use, see below
for details:

.. toctree::
    :maxdepth: 1

    cli/summaryclean
    cli/summarycombine
    cli/summarycombine_posteriors
    cli/summarycompare
    cli/summaryextract
    cli/summarymodify
    cli/summarypages
    cli/summarypageslw
    cli/summarypublication
    cli/summarysplit
    cli/summaryversion
