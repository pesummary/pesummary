=====================
summaryclassification
=====================

The `summaryclassification` executable allows the user to generate
source based classification probabilities given the samples in a GW specific
result file by interacting with the `pesummary.gw.pepredicates` and
`pesummary.gw.p_astro` modules and `PEPredicates`_ and `ligo.em-bright`_ packages.

.. _PEPredicates: https://will-farr.docs.ligo.org/pepredicates/
.. _ligo.em-bright: https://pypi.org/project/ligo.em-bright/

To see help for this executable please run:

.. code-block:: console

    $ summaryclassification --help

.. program-output:: summaryclassification --help

Generating classification probabilities
---------------------------------------

Below is an example of the output from summary `summaryclassification` on a
result file,

.. code-block:: console

    $ summaryclassification --webdir ./ --samples posterior_samples.hdf5 \
                            --labels GW150914
    $ ls ./
    GW150914_default_prior_pe_classification.json
    GW150914_population_prior_pe_classification.json
    GW150914_population_pepredicates_bar.png

`pesummary.gw.pepredicates`
---------------------------

.. autoclass:: pesummary.gw.pepredicates.PEPredicates
    :members:

`pesummary.gw.p_astro`
----------------------

.. autoclass:: pesummary.gw.p_astro.PAstro
    :members:
