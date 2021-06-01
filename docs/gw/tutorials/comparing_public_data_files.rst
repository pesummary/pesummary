=================================================
Comparing posterior samples from different groups
=================================================

As a result of the public release of the gravitational wave strain data, groups
outside of LIGO--Virgo have also been able to analyse the gravitational wave
strain data and obtain posterior samples for a series of events. For example,
see the
`2-OGC: Open Gravitational-wave Catalog of binary mergers from analysis of public Advanced LIGO and Virgo data <https://arxiv.org/abs/1910.05331>`_
and
`New binary black hole mergers in the second observing run of Advanced LIGO and Advanced Virgo <https://arxiv.org/abs/1904.07214>`_
papers. Below, we show that :code:`pesummary` provides an easy method for
comparing the posterior samples obtained by these different groups.

.. literalinclude:: ../../../../examples/gw/compare_GW150914.py
   :language: python
   :linenos:

.. image:: ./examples/comparison_for_GW150914.png
