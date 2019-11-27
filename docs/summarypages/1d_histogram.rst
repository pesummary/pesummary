=====================
The 1d Histogram page
=====================

There are 3 ways of viewing the marginalized posterior distributions with
PESummary. Either you view all posterior distributions on a single html page,
only view a select number of posteriors, or a single posterior distribution for
a chosen parameter (each are useful for different reasons).

For all of these cases, a single 1d marginalized posterior (either a histogram
or kde), an autocorrelation plot and a scatter plot of samples are produced.
By default the 1d marginalized posterior plot gives the median and 90%
confidence interval as a title and dashed lines respectively. If a prior is
given, and the `--include_prior` command line argument is passed, either a
histogram or a kde plot showing the prior is shown in grey. If an injection file
is passed, a single orange line showing the injected value will also be
plotted. An example 1d histogram is shown below:

.. image:: ./examples/1d_histogram.png

If you choose to use the `--gw` module for your analysis and specify the
`--kde_plot` command line argument, the 1d marginalized posterior will be
represented by a bounded 1d KDE. These KDE plots will be bounded according to
the following dictionary:

.. literalinclude:: ../../pesummary/gw/plots/bounds.py
   :language: python
   :lines: 19-65
   :linenos:

an example bounded 1d KDE is shown below:

.. image:: ./examples/1d_kde_plot.png

An autocorrelation plot is a commonly-used tool for checking correlations in
the data. If the posterior samples are randomly chosen, the autocorrelation
length should be near zero for any and all time-lag separations. If correlated
one or more of the autocorrelations will be significantly non-zero. An example
is shown below:

.. image:: ./examples/autocorrelation.png

An example scatter plot of samples is shown below:

.. image:: ./examples/scatter_plot.png 
