==================
Prior conditioning
==================

In some circumstances, parameters can be strongly correlated with each other
(for instance `a` might be strongly correlated with `b` and `c`). For these
cases, it is useful to show the marginalized prior samples for `a` conditioned
on the posterior distribution for `b` and `c`. `pesummary` provides
functionality to do this within the `pesummary.utils` module.

As an example, we recreate Figure 5 of the 
`GW190412 discovery paper <https://arxiv.org/pdf/2004.08342.pdf>`_ which shows
the `chi_p` prior samples conditioned on the `chi_eff` posterior distribution.

.. literalinclude:: ../../../examples/gw/prior_conditioning.py
   :language: python
   :linenos:

.. image:: ./examples/prior_conditioning.png
