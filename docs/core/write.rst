==================
The write function
==================

`pesummary` provides functionality to save samples in nearly all result file
formats. This is done with the `pesummary.io.write.write` function. Below we
show how the write function works.

Creating a result file
----------------------

In order to first use PESummary, you need a result file which is compatible.
The current allowed formats are `hdf5/h5`, `JSON`, `dat` and `txt` but if you
would prefer to have a file format that is not one of these, raise an `issue`_
and we will add this feature!

.. _issue: https://git.ligo.org/lscsoft/pesummary/issues

First, let us generate some random samples:

.. code-block:: python

    >>> samples = {
    ...     "a": np.random.uniform(10, 1, 100),
    ...     "b": np.random.uniform(5, 2, 100),
    ... }

The `pesummary.io.write.write` function requires a list of parameters and a
2d list of samples, where the columns correspond to the samples for a given
parameter. Therefore,

.. code-block:: python

     >>> parameters = list(samples.keys())
     >>> samples_array = np.array([samples[param] for param in parameters]).T

We may now save the samples with the following,

.. code-block:: python

    >>> from pesummary.io import write
    >>> write(parameters, samples_array, package="core", file_format="dat", filename="example.dat")

The contents of which is simply:

.. code-block:: python

    $ head -n 5 example.dat
    a	b
    7.456794389738383266e+00	2.875315515644248254e+00
    9.666238149526694912e+00	4.035072370273886655e+00
    2.297477206764988011e+00	4.091785657314235713e+00
    3.419056390335218687e+00	3.289249670307629714e+00

Alternative file formats may be chosen by changing the `file_format` keyword
argument.
