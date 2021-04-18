=============
Analytic PDFs
=============

In some circumstances, an analytic PDF is known and therefore we are not
required to store samples from a distribution. In :code:`pesummary`,
we handle 1D analytic PDFs through the
:code:`pesummary.utils.pdf.InterpolatedPDF` class (inherited from the
`rv_continous class <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html>`_
class in scipy) and :code:`pesummary.utils.pdf.DiscretePDF` class (inherited
from the
`rv_discrete class <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html>`_
class in scipy). Two dimensional analytic PDFs are handled by the
:code:`pesummary.utils.pdf.DiscretePDF2D` class. We also store two dimensional
analytic PDFs alongside their marginalized one dimensional analytic PDFs by the
:code:`pesummary.utils.pdf.DiscretePDF2Dplus1D` class.

.. autoclass:: pesummary.utils.pdf.InterpolatedPDF

.. autoclass:: pesummary.utils.pdf.DiscretePDF

.. autoclass:: pesummary.utils.pdf.DiscretePDF2D

.. autoclass:: pesummary.utils.pdf.DiscretePDF2Dplus1D
