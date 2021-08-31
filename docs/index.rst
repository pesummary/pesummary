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

PESummary provides 2 packages: `core` and `gw`. The `core` package provides all
of the necessary code for analysing, displaying and comparing data files from
general inference problems. The `gw` specific package contains GW functionality,
including converting posterior distributions, deriving event classifications and
GW specific plots.

First Steps
+++++++++++

.. toctree::
    :maxdepth: 1

    what_is_pesummary
    installation
    pesummary paper <https://arxiv.org/pdf/2006.06639.pdf>
    citing_pesummary


Unified input/output
++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    io/write
    io/read

Core Package
++++++++++++

.. toctree::
    :maxdepth: 2

    core/index

GW Package
++++++++++

.. toctree::
    :maxdepth: 2

    gw/index


Configuration
+++++++++++++

.. toctree::
    :maxdepth: 1

    conf/configuration


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
