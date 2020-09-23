=================
The read function
=================

`pesummary` provides functionality to read in nearly all result file formats.
This is done with the `pesummary.io.read.read <../io/read.html>`_ function. Below
we show how the read function works and some of the functions and properties
that it includes.

Reading a result file
---------------------

Below, we show how to read in a result file,

.. code-block:: python

    >>> from pesummary.io import read
    >>> data = read("example.dat", package="core")
    >>> json_load = read("example.json", package="core")
    >>> hdf5_load = read("example.hdf5", package="core")
    >>> txt_load = read("example.txt", package="core")

`pesummary` is able to read in `json`, `dat`, `txt` and `hdf5` file formats.

Extracting data
---------------

Once read in, `pesummary` offers properties and functions to extract the data
that is stored in the input file. As `pesummary` is able to read in both a file
that contains the samples for a single analysis and a `pesummary` metafile
which is able to store samples from multiple analyses, the way to extract the
samples is slightly different for the two situations.

For both cases, the stored samples can be inspected through the `.samples_dict`
property

Single analysis file
++++++++++++++++++++

When the input file only contains a single analysis, the samples are stored
All samples are stored in a subclass of the Python builtin dict:
`pesummary.utils.samples_dict.SamplesDict`,

.. code-block:: python

    >>> samples_dict = data.samples_dict
    >>> type(samples_dict)
    <class 'pesummary.utils.samples_dict.SamplesDict'>
    >>> print(samples_dict.keys())
    dict_keys(['a', 'b',...])

Each marginalized posterior distribution is stored as a subclass of
`numpy.ndarray`: `pesummary.utils.samples_dict.Array`. For more information
about the `SamplesDict` class, see the `SamplesDict docs <./SamplesDict.html>`_.
For more information about the `Array` class, see the
`Array docs <./Array.html>`_. This structure provides direct access to the
optimised array functions from `numpy` and the usability and familiarity of
dictionaries.

PESummary metafile
++++++++++++++++++

When the input file is a pesummary metafile, this time, the samples are stored
in a different subclass of the Python builtin dict:
`pesummary.utils.samples_dict.MultiAnalysisSamplesDict`,

.. code-block:: python

    >>> samples_dict = data.samples_dict
    >>> type(samples_dict)
    <class 'pesummary.utils.samples_dict.MultiAnalysisSamplesDict'>
    >>> print(samples_dict.keys())
    dict_keys(['analysis_1', 'analysis_2',...])
    >>> type(samples_dict['analysis_1'])
    <class 'pesummary.utils.samples_dict.SamplesDict'>

The samples for a given analysis and now assigned a label in order to
distinguish them from another analysis. The samples for a given analysis are
stored as a `pesummary.utils.samples_dict.SamplesDict` object. For more
information about the `MultiAnalysisSamplesDict` class, see the
`MultiAnalysisSamplesDict socs <./MultiAnalysisSamplesDict.html>`_.

For details about how to extract additional information from the `pesummary`
metafile, see the
`extract information from a pesummary file <pesummary_file.html>`_ docs.

Changing file format
--------------------

We may convert the result file to a different file format by using the
`.write()` method. For example, we may convert to a `.dat` file with,

.. code-block:: python

    >>> data.write(package="core", file_format="dat", outdir="./", filename="example.dat")

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
