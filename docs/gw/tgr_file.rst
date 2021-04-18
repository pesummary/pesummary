=============================================
Extract information from a pesummary TGR file
=============================================

When the `summarytgr <./cli/summarytgr.html>`_ executable is used, all
data is stored in the output :code:`h5` file (by default called
`tgr_samples.h5`). Here, we give details about how to extract the information
stored.

Loading the file
----------------

The `tgr_samples.h5` file can be loaded with the universal
`read <./read.html>`_ function. For example:

.. code-block:: python

    >>> from pesummary.io import read
    >>> f = read('tgr_samples.h5')
    >>> type(f)
    <class 'pesummary.gw.file.formats.pesummary.TGRPESummary'>

IMRCT data
----------

The purpose of the IMRCT test is to calculate a PDF for the final mass
and final spin deviation parameters, 
see `Ghosh et al 2018 <https://arxiv.org/abs/1704.06784>`_ for details. If only
one analysis is stored in the `tgr_samples.h5` file, the PDF can be extracted
with the following code snippet:

.. code-block:: python

    >>> deviation = f.imrct_deviation["final_mass_final_spin_deviations"]
    >>> type(deviation)
    <class 'pesummary.utils.pdf.DiscretePDF2D'>

If more than one analysis is stored, we need to specify which analysis we wish
to load:

.. code-block:: python

    >>> interested = "analysis_1"
    >>> deviation = f.imrct_deviation[interested]["final_mass_final_spin_deviations"]
    >>> type(deviation)
    <class 'pesummary.utils.pdf.DiscretePDF2D'>

The PDF is stored as a `ProbabilityDict2D class <../core/ProbabilityDict2D.html>`_.
This only stores the 2D PDF. However, we can generate the marginalized PDF for
both the final mass deviation and final spin deviation at once with the
following:

.. code-block:: python

    >>> full = deviation.marginalize()
    >>> type(full)
    <class 'pesummary.utils.pdf.DiscretePDF2Dplus1D'>

The output is the a
`DiscretePDF2Dplus1D class <../core/pdf.html>`_. We may
then use the input class methods to extract the PDF for the final mass deviation:

.. code-block:: python

    >>> final_mass_deviation = full.probs_x.x
    >>> final_mass_deviation_pdf = full.probs_x.probs

We may extract the final spin deviation PDF using similar code to above but
changing :code:`x` to :code:`y`. For example:

.. code-block:: python

    >>> final_spin_deviation = full.probs_y.x
    >>> final_spin_deviation_pdf = full.probs_y.probs

These PDFs are dependent on a set of posterior samples which were generated from
an inspiral only and postinspiral only analysis. These posterior samples are
also stored in the `tgr_samples.h5` file. These posterior samples are accessed
in the same way as other files. For example,

.. code-block:: python

    >>> pe_samples = f.samples_dict
    >>> type(pe_samples)
    <class 'pesummary.utils.samples_dict.MultiAnalysisSamplesDict'>
    >>> type(pe_samples["inspiral"])
    <class 'pesummary.utils.samples_dict.SamplesDict'>

This returns a
`MultiAnalysisSamplesDict class <../core/MultiAnalysisSamplesDict.html>`_ which
stores both the inspiral and postinspiral samples. The individual inspiral
and postinspiral analyses are stored as a
`SamplesDict class <../core/SamplesDict.html>`_.

All additional information is stored in the meta data. We can extract the meta
data with the following:

.. code-block:: python

    >>> kwargs = f.extra_kwargs

If only one analysis is stored, the kwargs are stored under the label
`primary`. If multiple analyses are stored, we use the labels which were provided
from the command line,

.. code-block:: python

    >>> interested = "primary"
    >>> print(kwargs[interested])
    {'GR Quantile (%)': X, 'N_bins': Y, 'Time (seconds)': Z, 'evolve_spins': array([b'False', b'False'], dtype='|S5'), 'inspiral approximant': 'approximant', 'inspiral maximum frequency (Hz)': 'frequency', 'postinspiral approximant': 'approximant', 'postinspiral minimum frequency (Hz)': 'frequency'}

Where we are using place holders (:code:`X`, :code:`Y`, :code:`Z`,
:code:`'approximant'`, ...) for the above code snippet.
