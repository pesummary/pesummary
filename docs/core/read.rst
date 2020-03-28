=================
The read function
=================

`pesummary` provides functionality to read in nearly all result file formats.
This is done with the `pesummary.core.file.read.read` function. Below we show
how the read function works and some of the functions and properties that it
includes.

For details about the extra properties there are when reading in a `pesummary`
result file, see `Reading in a pesummary metafile <read_pesummary.html>`_

Reading a result file
---------------------

Below is how you might read in a result file,

.. code-block:: python

    >>> from pesummary.core.file.read import read
    >>> data = read("example.dat")
    >>> json_load = read("example.json")
    >>> hdf5_load = read("example.hdf5")
    >>> txt_load = read("example.txt")

`pesummary` is able to read in `json`, `dat`, `txt` and `hdf5` file formats.

Manipulating the result file
----------------------------

Once read in, `pesummary` offers properties and functions to inspect and
manipulate the result file. Firstly, the stored samples can be inspected through
the `.samples_dict` property:

.. code-block:: python

    >>> samples_dict = data.samples_dict
    >>> type(samples_dict)
    <class 'pesummary.utils.utils.SamplesDict'>

For details about this `SamplesDict` class see
`SamplesDict class <SamplesDict.html>`_

Latex
-----

`pesummary` provides the ability to interact with `latex` and produce a
`latex table` displaying the results or `latex macros` to aid in writing
publications. You are able to specify which parameters you want included in
the latex table as well as detailed descrptions of what each parameter is.
Below is an example for how to make a latex table,

.. code-block:: python

    >>> description_mapping = {
    ...     "a": "The a parameter",
    ...     "b": "The b parameter"
    ... }
    >>> data.to_latex_table(parameter_dict=description_mapping)
    \begin{table}[hptb]
    \begin{ruledtabular}
    \begin{tabular}{l c }
    The a parameter & $35.04^{+8.00}_{-5.00}$\\
    The b parameter & $76.01^{+7.56}_{-0.45}$\\
    \end{tabular}
    \end{ruledtabular}
    \caption{}
    \end{table}
    >>> data.to_latex_table(save_to_file="table.tex", parameter_dict=description_mapping)

And latex macros may be generated as follows,

.. code-block:: python

    >>> macros_map = {"a": "A"}
    >>> data.generate_latex_macros(parameter_dict=macros_map)
    \def\A{$35.04^{+8.00}_{-5.00}$}
    \def\Amedian{$35.04$}
    \def\Aupper{$43.04$}
    \def\Alower{$30.04$}
    >>> data.generate_latex_macros(save_to_file="macros.tex", parameter_dict=macros_map)

Changing file format
--------------------

We may convert the result file to a different file format by using the one of
the inbuild functions. For example, we may convert to a `.dat` file, by using
the `.to_dat` method,

.. code-block:: python

    >>> data.to_dat(outdir="./", label="my_example")
