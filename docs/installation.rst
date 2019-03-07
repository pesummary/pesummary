============
Installation
============

Install PESummary
-----------------

:code:`PESummary` is developed and tested for python 3.5+. This code will work for python 2.7, however this is not suggested. We recommend that this code is installed inside a virtual environment using :code:`virtualenv`. This environment can be installed with python 3.5+ using `pyenv`_.

.. _pyenv: https://github.com/pyenv/pyenv

For detailed instructions on how to set up your virtual environment, please refer to `setting up a virtual environment           
<virtual_environment.html>`_. 

Installing PESummary using pip
------------------------------

If you choose to install :code:`PESummary` using :code:`pip`, then simply run:

.. code-block:: console

   $ source ~/virtualenvs/pesummary_py3.6/bin/activate
   $ pip install pesummary


Installing PESummary using conda
--------------------------------

If you choose to install :code:`PESummary` using :code:`conda`, then simply run:

.. code-block:: console

    $ source ~/virtualenvs/pesummary_pyenv3.6/bin/activate
    $ conda install -c conda-forge pesummary

Installing PESummary from source
--------------------------------

If you would like to install :code:`PESummary` from source, then please make sure that you set up your virtual environment correctly using either the instructions highlighted abov or using your own techniques, you have a working version of `pip` and you have `git` correctly installed.

First clone the repository, then install all requirements, then install the software,

.. code-block:: console

   $ source ~/virtualenvs/pesummary_pyenv3.6/bin/activate
   $ git clone git@git.ligo.org:charlie.hoy/pesummary.git
   $ cd pesummary/
   $ pip install -r requirements.txt
   $ python setup.py install

Installing optional requirements
################################

The :code:`requirements.txt` file contains all the necessary packages for running :code:`PESummary`. In addition we provide additional packages for running the tests and creating these docs. To install these :code:`optional_requirements`, please install using,

.. code:: console

   $ source ~/virtualenvs/pesummary_pyenv3.6/bin/activate
   $ pip install -r optional_requirements.txt

Identifying the version number of your installation
---------------------------------------------------

We recommend that you always keep up to date with new releases. If you would like to know what version of :code:`PESummary` you are running,

.. code:: console

   $ python -c "import pesummary; print(pesummary.__version__)"
   0.1.3

This shows that we are running version 0.1.3.