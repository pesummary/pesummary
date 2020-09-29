=======================================
Plotting gravitational wave strain data
=======================================

When LIGO/Virgo announces a gravitational wave, both the samples and
gravitational wave data are released to the public. Through `pesummary`'s
`pesummary.gw.file.strain` module, we can not only plot the gravitational
wave data, but we can also compare it to the maximum likelihood waveform
from the parameter estimation analysis. Below we show an example for GW150914,

.. literalinclude:: ../../../../examples/gw/plot_waveform_on_strain_data_GW150914.py
   :language: python
   :linenos:

.. image:: ./examples/GW150914.png

and GW190814,

.. literalinclude:: ../../../../examples/gw/plot_waveform_on_strain_data_GW190814.py
   :language: python
   :linenos:

.. image:: ./examples/GW190814.png
