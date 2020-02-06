===================
Incorporating latex
===================

Here we will go though step by step how to generate latex table to show the
contents of a given result file and how to generate latex macros for all of
the parameters that you are interested in.

Firstly, for a standard non-PESummary metafile, you can follow the the following
instructions

.. literalinclude:: ../../examples/latex.py
   :language: python
   :lines: 1-21
   :linenos:

If you want to generate a latex table from a PESummary metafile, then because
a single PESummary metafile can contain many runs, you need to specify which
analysis you want included in the table. This is done via the labels keyword
argument. You are able to select and many or as few as you wish. An example
if shown below:

.. literalinclude:: ../../examples/latex.py
   :language: python
   :lines: 27-30
   :linenos:

We are able to include more than one analysis in the latex table by simply
passing more than one label:

.. literalinclude:: ../../examples/latex.py
   :language: python
   :lines: 35-38
   :linenos:

If we wish to make latex macros, then we can again, simply follow the same
logic as above. For example, if we had a standard non-PESummary metafile

.. literalinclude:: ../../examples/latex.py
   :language: python
   :lines: 42-50
   :linenos:

If we wish to generate latex macros for a PESummary metafile, this can be
done by as follows,

.. literalinclude:: ../../examples/latex.py
   :language: python
   :lines: 55-58
   :linenos:
