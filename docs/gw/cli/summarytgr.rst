==========
summarytgr
==========

The `summarytgr` executable is designed to post-process and generate webpages to
display results from analyses which test the General Theory of Relativity. To
see help for this executable please run:

.. code-block:: console

    $ summarytgr --help

.. program-output:: summarytgr --help

IMRCT
-----

This test checks for consistencies between the inspiral and postinspiral
estimates for the mass and spin of the final black hole in the binary black hole
merger. See `Ghosh et al 2018 <https://arxiv.org/abs/1704.06784>`_ for details.
The IMRCT test can be performed by passing the :code:`--test imrct` command line
argument.

.. note::

    This test requires samples from an inspiral only and postinspiral
    only parameter estimation analysis for one or more analyses.

You may tailor the postprocessing by passing a selection of kwargs specific
to the IMRCT postprocessing (:code:`--imrct_kwargs`). All kwargs are passed to
the :code:`imrct_deviation_parameters_from_final_mass_final_spin` function in
:code:`pesummary.gw.conversions.tgr`. For details about the allowed kwargs
see `the conversion docs <../Conversion.html#pesummary.gw.conversions.tgr.imrct_deviation_parameters_from_final_mass_final_spin>`_.

An example command line can be seen below:

.. code-block:: console

    $ summarytgr --webdir ./webpage --samples inspiral.dat postinspiral.dat \
                 --test imrct --labels inspiral postinspiral \
                 --imrct_kwargs N_bins:401 multi_process:16
