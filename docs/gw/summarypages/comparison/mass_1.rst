=================================
The comparison 1d histogram pages
=================================

If more than one result file is passed to `pesummary`, various comparison
plots are produced. Comparison marginalized histograms can be seen on the
`1d histogram pages <https://pesummary.github.io/GW190412/html/Comparison_mass_1.html>`_.
Like the single analysis `1d histogram page <../IMRPhenomPv3HM/mass_1.html>`_,
we see a marginalized 1d histogram plot showing the posterior distribution for
all result files. By default these will be colored with the `colorblind` seaborn
palette, but can be changed with the `--palette` command line argument or the
`--colors` command line argument. Each 1d posterior will also show the
corresponding 90% symmetric credible interval shown by dashed lines.

A comparison cumulative distribution function allows for an alternative
method for comparing the distributions and is also provided. Finally a box plot
is also provided.

At the bottom of the page, we provide the
`KS <https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test>`_ and
`JS <https://en.wikipedia.org/wiki/Jensen–Shannon_divergence>`_ statistics, used
to compare two posterior distrubitons. Again, like the other tables on these
webpages, these may be exported to `csv` by clicking on the export to csv button.

If you wish to see how multiple marginalized posterior distributions differ
between the different runs, you may either visit the
`custom comparison page <https://pesummary.github.io/GW190412/html/Comparison_Custom.html>`_
or the `all comparison page <https://pesummary.github.io/GW190412/html/Comparison_All.html>`_.
The custom comparison page allows the user to select only the parameters they
are interested in. Like the `single analysis corner page <../IMRPhenomPv3HM/corner.html>`_,
the parameter are either selected via the side bar or typed manually in the
search bar. The all comparison page will show the comparison marginalized
1d histograms for all parameters that are common to all result files.
