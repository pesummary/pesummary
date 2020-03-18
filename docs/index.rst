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
-----------

.. toctree::
    :maxdepth: 1

    what_is_pesummary
    installation
    citing_pesummary

Working with data
-----------------

Making a result file
++++++++++++++++++++

.. toctree::
    :maxdepth: 2

    data/making_a_result_file

Reading a result file
+++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    data/reading_a_result_file
    data/reading_the_metafile
    data/conversion
    data/latex_table
    data/parameters

The summary pages
+++++++++++++++++

.. toctree::
    :maxdepth: 1

    summarypages/corner
    summarypages/config
    summarypages/1d_histogram
    summarypages/interactive
    summarypages/classification
    summarypages/comparison
    summarypages/comparison_interactive
    summarypages/publication
    summarypages/notes
    summarypages/downloads
    summarypages/examples

Customisation
+++++++++++++

One of the main advantages for using PESummary as your default post-processing
script is the degree of customisation that is possible. Below we will highlight
a couple of the main customisation features of PESummary (some have already
been mentioned and discussed in 'The summary pages' section above).

.. toctree::
    :maxdepth: 1

    custom/plots

Executables
+++++++++++

PESummary boasts several executables to make it even easier to use, see below
for details:

.. toctree::
    :maxdepth: 1

    executables/summaryclassification
    executables/summaryclean
    executables/summarycombine
    executables/summarydetchar
    executables/summarymodify
    executables/summarypages
    executables/summarypageslw
    executables/summarypipe
    executables/summarypublication
    executables/summaryreview
    executables/summaryversion

Tutorials
---------

.. toctree::
    :maxdepth: 1

    tutorials/GWTC1_plots
    tutorials/public_pages
    tutorials/make_your_own_page_from_metafile
    tutorials/population_scatter_plot_GWTC-1
    tutorials/latex
    tutorials/interaction_with_ligo_skymap

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
