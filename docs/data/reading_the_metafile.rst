============================
Reading a PESummary metafile
============================

Although the PESummary metafile can be read in with the core `JSON` or
`h5py` packages, the recommended way of reading in a PESummary metafile is
with the same `pesummary.core.file.read.read` or `pesummary.gw.file.read.read`
functions that was used in the 
`Reading result files <reading_a_result_file.html>`_ tutorial.


An example python script showing how to extract information from the metafile
is shown below:

.. literalinclude:: ../../examples/extract_information.py
    :language: python
    :linenos:

`pesummary.core.file.formats.pesummary`
---------------------------------------

.. autoclass:: pesummary.core.file.formats.pesummary.PESummary
    :members:

`pesummary.gw.file.formats.pesummary`
-------------------------------------

.. autoclass:: pesummary.gw.file.formats.pesummary.PESummary
    :members:
