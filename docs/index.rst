.. PESummary documtentation master file, created by
   sphinx-quickstart on Sat Jan 12 14:02:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: PESummary | Home

Welcome to PESummary's documentation!
=====================================

PESummary: The code agnostic Parameter Estimation Summary page builder
----------------------------------------------------------------------

.. warning::

      You are viewing documentation for a development build of PESummary.
      This version may include unstable code, or breaking changes relative
      the most recent stable release.
      To view the documentation for the latest stable release of PESummary,
      please `click here <../stable_docs/index.html>`_.

PESummary is a collaboration-driven Python package providing tools for
generating summary pages for all sample generating codes. 

PESummary provides a user-friendly, intuitive interface to the common
LIGO/Virgo samples and allows for the user to reproduce all outputs from the
LIGO/Virgo Scientific collaboration.

First Steps
+++++++++++

.. toctree::
    :maxdepth: 1

    what_is_pesummary
    installation
    citing_pesummary
    core/making_a_result_file

Configuration
+++++++++++++

.. toctree::
    :maxdepth: 1

    conf/configuration

Reading a result file
+++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    read

Manipulating posterior samples
++++++++++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    core/Array
    core/SamplesDict
    core/MCMCSamplesDict

Customisation
+++++++++++++

One of the main advantages for using PESummary as your default post-processing
script is the degree of customisation that is possible. Below we will highlight
a couple of the main customisation features of PESummary (some have already
been mentioned and discussed in 'The summary pages' section above).

.. toctree::
    :maxdepth: 1

    core/custom_plots

Base Summary pages
++++++++++++++++++

Below we show details about the `summarypages` that are built for the `core`
module:

.. toctree::
    :maxdepth: 1

    core/summarypages/corner
    core/summarypages/config
    core/summarypages/1d_histogram
    core/summarypages/interactive
    core/summarypages/comparison
    core/summarypages/comparison_interactive
    core/summarypages/notes
    core/summarypages/downloads
    core/summarypages/examples

Executables
+++++++++++

PESummary boasts several executables to make it even easier to use, see below
for details:

.. toctree::
    :maxdepth: 1

    cli/summaryclassification
    cli/summaryclean
    cli/summarycombine
    cli/summarydetchar
    cli/summarymodify
    cli/summarygracedb
    cli/summarypages
    cli/summarypageslw
    cli/summarypipe
    cli/summarypublication
    cli/summaryrecreate
    cli/summaryreview
    cli/summaryversion

Tutorials
+++++++++

.. toctree::
    :maxdepth: 1

    tutorials

Specific to Gravitational Waves
-------------------------------

Parameter definitions
+++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    gw/parameters

Converting parameters
+++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    gw/Conversion

GW specific summarypages
++++++++++++++++++++++++

Below we give details about the extra summary pages that are built when we use
the `gw` module:

.. toctree::
    :maxdepth: 1

    gw/summarypages/classification
    gw/summarypages/publication

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
