========
Examples
========

Below are a series of example webpages and the command lines that have been
used to generate them.

Single result file
------------------

If you only have a single results file, then you can generate a summary page
with the following,

.. code-block:: console

   $ summarypages.py --email albert.einstein@ligo.org \
                     --webdir /home/albert.einstein/public_html/one_approximant \
                     --samples ./GW150914_result.h5 \
                     --gw

An example of this is shown
`here <https://docs.ligo.org/charlie.hoy/pesummary_examples/single/home.html>`_.

Multiple GW results files
-------------------------

If you have mutliple results files, then you can generate a single summary page
with the following,

.. code-block:: console

   $ summarypages.py --email albert.einstein@ligo.org \
                     --webdir /home/albert.einstein/public_html/two_approximants \
                     --samples ./IMRPhenomPv2/GW150914_result.h5 ./IMRPhenomP/GW150914_result.h5 \
                     --gw

An example of this is shown
`here <https://docs.ligo.org/charlie.hoy/pesummary_examples/double/home.html>`_.
