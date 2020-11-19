==================
What is PESummary?
==================

The PESummary Python library provides tools for creating professional looking
summary pages and a single complete meta file containing all information
about the analysis allowing for complete reproducability for all sample
generating codes. 

This package is designed for users who want an easy solution to visualize the
contents of their data files and to distribute the contents to collaborators or
the general public. As a result, this package is meant to be as easy-to-use as
possible with self explanatory function and class names and extensive
documentation to show how easy it is to use this Summary page builder.

The code is hosted at https://git.ligo.org/lscsoft/pesummary .

The basic idea
--------------

The basic idea of PESummary is to simplify reading in result files of
differing formats, allowing the users to study and plot data quickly and
effectively and to share their results with others. PESummary provides simple
methods for reading `hdf5`, `dat`, `txt` and `json` result files:

.. code-block:: python

    >>> from pesummary.io import read
    >>> hdf5_object = read("posterior_samples.hdf5")
    >>> json_object = read("posterior_samples.json")

plotting the contents:

.. code-block:: python

    >>> samples = json_object.samples_dict
    >>> samples.plot("a", type="hist")

and for producing html pages to visualise the plots from a browser; either from
the command line,

.. code-block:: console

    $ summarypages --webdir /home/albert.einstein/example \
                   --samples ./posterior_samples.dat \
                   --labels example

or via the python interface,

.. code-block:: python

    >>> from pesummary.core.plots.main import _PlotGeneration
    >>> from pesummary.core.webpage.main import _WebpageGeneration
    >>> plotting_object = _PlotGeneration(
    ...    webdir="/home/albert.einstein/example", labels=["example"],
    ...    samples=json_object.samples_dict
    ... )
    >>> webpage_object = _WebpageGeneration(
    ...    webdir="/home/albert.einstein/example", labels=["example"],
    ...    samples=json_object.samples_dict, user="albert.einstein",
    ...    colors=["b"]
    ... )
