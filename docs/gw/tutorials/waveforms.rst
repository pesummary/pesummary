====================
Generating waveforms
====================

Through `pesummary`'s `SamplesDict class <../SamplesDict.html>`_, we can easily
generate a waveform based on a specific set of posterior samples either in the
time domain or the frequency domain. This is through the `td_waveform` and
`fd_waveform` methods. Below we show an example using the publically available
GW190814 posterior samples. Considering the time domain,

.. literalinclude:: ../../../../examples/gw/making_a_waveform_in_time_domain.py
   :language: python
   :linenos:

.. image:: ./examples/waveform_td.png

and in the frequency domain,

.. literalinclude:: ../../../../examples/gw/making_a_waveform_in_frequency_domain.py
   :language: python
   :linenos:

.. image:: ./examples/waveform_fd.png
