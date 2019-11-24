================================
Setting up a virtual environment
================================

Creating your first virtual environment
---------------------------------------

`Virtualenv`_ is a package that is used to generate virtual environments.
To install this, use `pip`

.. _Virtualenv: https://packaging.python.org/key_projects/#virtualenv

.. code-block:: console

   $ pip install virtualenv

This package allows you to create an isolated python environment. The main
advantage of this is that you do not have to worry with breaking the package
previously installed on your machine. All packages that are installed within
this environment are isolated from not only the main environment of your
machine, but also your other virtual environments. It is always recommened to
install different python packages into different virtual environments.

To create a virtual environment, run :code:`virtualenv`:

.. code-block:: console

   $ mkdir -p ~/virtualenvs
   $ virtualenv ~/virtualenvs/environment

Here, the second argument is simply the location that you would like to create
the virtual environment. Here we have chosen to create the virtual environment
inside the `virtualenvs` directory in a sub directory called `environment`. To
activate this environment, run,

.. code-block:: console

   $ source ~/virtualenvs/environment/bin/activate
   (environment) $

You are now inside your virtual environment (as noted by the virtual environment
name before the :code:`$`) and you are free to install whatever packages that
you would like without affecting your main installation. It is always advised
to upgrade your :code:`pip` and :code:`setuptools` before installing any
packages,

.. code-block:: console

   (environment) $ pip install --upgrade pip setuptools

To deactivate from a virtual environment, run,

.. code-block:: console

   (environment) $ deactivate
   $

You will then leave your virtual environment and you will not be able to import
any python packages that were installed within your environment. 

Creating a virtual environment with a specified python version
--------------------------------------------------------------

In order to create a virtual environment with a specified python installation,
we recommend that you use `pyenv`_

.. _pyenv: https://github.com/pyenv/pyenv

This package allows the user to install any released version of python and use
that installation in your virtual environment. Please follow the
instructructions located `here <https://github.com/pyenv/pyenv>`_ for details on
how to do this. Once you have chosen the version of python that you would like
to install (for clarity lets say we wanted to install python 3.6), you can
create a virtual environment with that python installation by using the command,

.. code-block:: console

   $ virtualenv -p ~/.pyenv/versions/3.6.0/bin/python3 ~/virtualenvs/environment

A virtual environment with a python 3.6 installation will then be created.
