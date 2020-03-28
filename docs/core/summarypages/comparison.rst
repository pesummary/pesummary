===================
The Comparison page
===================

If more than one result file is passed to PESummary, comparison plots and
comparison pages are made by default (this can be turned off by passing the
`--disable_comparison` flag). Like with the
`1d histogram pages <1d_histogram.html>`_, there are 3 ways of viewing the
comparison plots. Either you view all posterior distributions on a single html
page, only view a select number of posteriors, or a single posterior
distribution for a chosen parameter.

For all these cases, a single 1d marginalized posterior comparison plot (either
histogram or kde), a comparison CDF plot and a comparison box plot are produced.
By default these will be colored with the `colorblind` seaborn palette, but
can be changed with the `--palette` command line argument or the `--colors`
command line argument. Each 1d posterior will also show the corresponding
90% confidence interval showing by dashed lines. An example is shown below:

.. image:: ./examples/comparison_1d_histogram.png

A comparison cumulative distribution function allows for an alternative
method for comparing the distributions. An example is shown below:

.. image:: ./examples/comparison_cdf.png

An example comparison box plot is shown below:

.. image:: ./examples/comparison_box.png
