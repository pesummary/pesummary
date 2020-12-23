===================================
Remaking plots that appear in GWTC1
===================================

Here we will go through step by step how to make all plots that appear in the
`GWTC1`_ paper produced by the LIGO Virgo collaboration. This involves first
downloading the data, and then using the
`summarypublication <../../core/cli/summarypublication.html>`_ executable
provided by PESummary.

.. _GWTC1: https://arxiv.org/abs/1811.12907

.. literalinclude:: ../../../../examples/gw/GWTC1_plots.sh
   :language: bash
   :lines: 1-28
   :linenos:

Now that we have all of the data and all of the variables setup, we can now
run the `summarypublication` executable and make a 2d bounded contour of the
mass_1 and mass_2 parameter space

.. literalinclude:: ../../../../examples/gw/GWTC1_plots.sh
    :language: bash
    :lines: 31-37
    :linenos: 
 
.. image:: ./examples/2d_contour_plot_mass_1_and_mass_2.png

Now we can produced a violin plot showing the variation in mass_ratio

.. literalinclude:: ../../../../examples/gw/GWTC1_plots.sh
    :language: bash
    :lines: 40-45
    :linenos:

.. image:: ./examples/violin_plot_mass_ratio.png

Now we can produce a 2d bounded contour of the theta_jn and luminosity_distance 
parameter space

.. literalinclude:: ../../../../examples/gw/GWTC1_plots.sh
    :language: bash
    :lines: 48-54
    :linenos:

.. image:: ./examples/2d_contour_plot_theta_jn_and_luminosity_distance.png

Now we can produce a 2d bounded contour of the luminosity_distance chirp_mass
parameter space

.. literalinclude:: ../../../../examples/gw/GWTC1_plots.sh
    :language: bash
    :lines: 57-63
    :linenos:

.. image:: ./examples/2d_contour_plot_luminosity_distance_and_chirp_mass.png
