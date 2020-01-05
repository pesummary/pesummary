====================
The Publication page
====================

If more than one result file, and the `--publication` command line option
passed, then plots that appear in the `GWTC1`_ paper produced by the LIGO Virgo
collaboration will also be generated thanks to the
`summarypublication <../executables/summarypublication.html>`_ executable. These
plots include a bounded 2d contour plot,

.. _GWTC1: https://arxiv.org/abs/1811.12907

.. image:: ./examples/contour_plot.png

a violin plot,

.. image:: ./examples/violin_plot.png

and a spin disk plot,

.. image:: ./examples/spin_disk.png

Of course, the colors and linestyles can be changed by simply specifying
the `--palette` command line option or the `--colors` and `--linestyles`
command line options.

..note ::
    If you find that the publication plots require further smoothing, then
    simply add the `--publication_kwargs gridsize:500` command line
    argument
