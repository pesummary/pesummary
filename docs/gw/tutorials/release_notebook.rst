Making a jupyter notebook for release
=====================================

When samples are released to the public, it can be helpful to generate a
jupyter notebook showing how to extract and visualise the samples that are
stored. Through `pesummary`'s `pesummary.gw.notebook` module, we can create
this jupyter notebook. 

As an example, we generate a jupyter notebook showing the contents of the
`GW190814 public data release <https://dcc.ligo.org/public/0168/P2000183/008/GW190814_posterior_samples.h5>`_,

.. literalinclude:: ../../../../examples/gw/make_public_release.py
   :language: python
   :linenos: 

The notebook that was generated can be seen `here <posterior_samples.html>`_.
