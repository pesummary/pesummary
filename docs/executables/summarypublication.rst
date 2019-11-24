==================
summarypublication
==================

The `summarypublication` executable allows the user to generate plots that
appear in the `GWTC1`_ paper produced by the LIGO Virgo collaboration.

.. _GWTC1: https://arxiv.org/abs/1811.12907

To see help for this executable please run:

.. code-block:: console

    $ summarypublication --help

.. program-output:: summarypublication --help

Generating a 2d contour plot
----------------------------

PESummary has the ability to generate a 2d contour plot which contains bounded
2d KDEs for all result files passed (like Figure 4 in thr GWTC1 paper). The
bounds for each parameter is hardcoded in the
`pesummary.gw.plots.bounded_2d_kde` module:

.. literalinclude:: ../../pesummary/gw/plots/bounded_2d_kde.py
   :language: python
   :lines: 5-51
   :linenos:

In order to ensure that the lines are distinguishable from one another, you can
either choose a `seaborn` palette using the `--palette` command line argument or
the `--colors` command line argument to pass a list of colors, one for each
result file.

An example command line is below:

.. code-block:: console

    $ summarypublication --webdir ./ --samples GW150914_result.json \
                         --labels GW150914 --plot 2d_contour \
                         --parameters mass_1 mass_2

Generating a violin plot
------------------------

PESummary has the ability to generate violin plots for a  result files passed
(like Figure 5 in the GWTC1 paper). An example command line is below:

.. code-block:: console

    $ summarypublication --webdir ./ --samples GW150914_result.json \
                         --labels GW150914 --plot violin --parameters chi_eff

Generating a spin disk plot
---------------------------

PESummary has the ability to generate a spin disk plot for a given result file
(like figure 6 in the GWTC1 paper). An example command line is below:

.. code-block:: console

    $ summarypublication --webdir ./ --samples GW150914_result.json \
                         --labels GW150914 --plot spin_disk

`pesummary.gw.plots.publication`
--------------------------------

.. automodule:: pesummary.gw.plots.publication
    :members:
