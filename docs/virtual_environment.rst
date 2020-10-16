================================
Setting up a virtual environment
================================

pyenv
-----

In order to build a virtual environment which has a unique python version, we
first need to download a specific python version. The easiest way of doing this
is by using `pyenv <https://github.com/pyenv/pyenv>`_.

After following the `installation instructions <https://github.com/pyenv/pyenv/blob/master/README.md#basic-github-checkout>`_,
we may download a specific python version with the following,

.. code-block:: bash

    $ pyenv install 3.7.0

This will download the `python==3.7.0` binary file to the following directory:
:code:`~/.pyenv/versions/3.7.0/bin/python3`.

Creating a virtualenv
---------------------

`virtualenv <https://pypi.org/project/virtualenv/>`_ is a tool used to create
isolated python environments by creating a folder which contains all of the
necessary executables to install multiple python packages.

A virtual environment called `myenv` may be created in the directory
:code:`~/virtualenvs` with the following,

.. code-block:: bash

    $ virtualenv -p ~/.pyenv/versions/3.7.0/bin/python3 ~/virtualenvs/myenv

Where the flag :code:`-p` allows for you to specify the python interpreter
of your choice -- here we use :code:`python==3.7.0` installed above.

We may then activate this virtual environment with,

.. code-block:: bash

    $ source ~/virtualenvs/myenv/bin/activate

Packages can then be installed using the :code:`pip` command,

.. code-block:: bash

    (myenv) $ pip install numpy

Once you are done working in the virtual environment, you can deactivate it.

.. code-block:: bash

    $ deactivate

Installing a jupyter kernel
---------------------------

It is often convenient to use jupyter notebooks for a specific virtual
environment. By default, only the base environment will have a useable kernel.
We may install a kernal for a specific virtual environment with the following,

.. code-block:: bash

    $ source ~/virtualenvs/myenv/bin/activate
    (myenv) $ pip install ipykernel
    (myenv) $ ipython kernel install --user --name=myenv

You may then launch your jupyter notebook (or refresh if already open) and a
kernel called :code:`myenv` will be available for you to use. This will then
have all of the packages installed in your virtual environment available for
use.


