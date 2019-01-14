============
Installation
============

Install PESummary
-----------------

:code:`PESummary` is developed and tested for Python 2.7 and/or Python 3.5+. We recommend that this is installed inside a virtual environment. To set up the virtual environment for Python 2.7 or Python 3.5+, we recommend that you use `pyenv`_.

.. _pyenv: https://github.com/pyenv/pyenv

In the following we assume that you have a working python installation.

Install PESummary using pip
---------------------------

If you choose to install via :code:`pip`, then simply run:

.. code-block:: console

   $ pip install pesummary

Once you have run these steps, you have :code:`pesummary` installed. To check, run :code:`summarypages.py --help`.


Install PESummary from source
-----------------------------

If you choose to install PESummary from source, then please clone the repository and install the software:

.. code-block:: console

   $ git clone git@git.ligo.org:charlie.hoy/pesummary.git
   $ cd pesummary/
   $ python setup.py install

Once you have run these steps, you have :code:`pesummary` installed. To check, run :code:`summarypages.py --help`.
