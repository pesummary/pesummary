====================
Generating waveforms
====================

Through :code:`pesummary`'s `SamplesDict class <../SamplesDict.html>`_, we can easily
generate a waveform based on a specific set of posterior samples either in the
time domain or the frequency domain. This is through the :code:`td_waveform` and
:code:`fd_waveform` methods. Below we show an example using the publically available
GW190814 posterior samples.

First let us plot a waveform in the time domain,

.. literalinclude:: ../../../../examples/gw/making_a_waveform_in_time_domain.py
   :language: python
   :linenos:

.. image:: ./examples/waveform_td.png

In the above example only the maximum likelihood waveform is plotted. Sometimes
it is useful to know the uncertainty on this waveform. We can calculate and plot
the 1 sigma and 2 sigma symmetric confidence intervals of this waveform by
taking advantage of the :code:`level` kwarg,

.. literalinclude:: ../../../../examples/gw/making_a_waveform_in_time_domain_with_uncertainty.py
   :language: python
   :linenos:

.. image:: ./examples/uncertainty_waveform_td.png

Here we have chosen to downsample the posterior samples to 1000 samples
(:code:`_ = EOB.downsample(1000)`) and used 4 CPUs (:code:`multi_process=4`)
to speed up waveform generation.

We can also generate a waveform in the frequency domain with,

.. literalinclude:: ../../../../examples/gw/making_a_waveform_in_frequency_domain.py
   :language: python
   :linenos:

.. image:: ./examples/waveform_fd.png

For more details about the waveform generator in :code:`pesummary` see,

.. automodule:: pesummary.gw.waveform
    :members:
